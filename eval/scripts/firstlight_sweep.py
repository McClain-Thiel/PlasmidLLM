#!/usr/bin/env python3
"""Phase 1: First-light sweep to pick optimal sampling config.

Generates ~50 plasmids per cell across a sampling grid, runs quick
pLannotate eval on a sample, and picks the best config.

Usage:
    python eval/scripts/firstlight_sweep.py \
        --model McClain/PlasmidLM-kmer6-GRPO-plannotate \
        --n-per-cell 50 \
        --output-dir eval/runs/firstlight_20260408
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def clean_dna(raw: str) -> str:
    seq = re.sub(r"<[^>]+>", "", raw.upper())
    return re.sub(r"[^ATGCN]", "", seq)


def generate_cell(model, tokenizer, prompts, batch_size, max_new_tokens,
                  temperature, top_k, top_p, device):
    """Generate sequences for one grid cell (model already loaded)."""
    all_results = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = list(prompts[batch_start:batch_start + batch_size])
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        with torch.inference_mode():
            outputs = model.generate_simple(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(outputs.shape[0]):
            raw = tokenizer.decode(outputs[i].tolist())
            sep = "<SEP>" if "<SEP>" in raw else "<SEQ>"
            dna_part = raw.split(sep, 1)[-1] if sep in raw else raw
            dna = clean_dna(dna_part)
            has_eos = "<EOS>" in raw
            all_results.append({
                "prompt": batch[i],
                "sequence": dna,
                "length": len(dna),
                "gc": (dna.count("G") + dna.count("C")) / max(len(dna), 1),
                "has_eos": has_eos,
            })
    return all_results


def quick_plannotate(fasta_path: str, output_dir: str, workers: int = 4) -> list[dict] | None:
    """Run pLannotate via conda subprocess, return results or None."""
    ann_script = Path(__file__).parent / "annotate.py"
    cmd = [
        "conda", "run", "-n", "plannotate", "--no-banner",
        sys.executable, str(ann_script),
        "--input", fasta_path,
        "--output-dir", output_dir,
        "--workers", str(workers),
        "--skip-prodigal", "--skip-dustmasker",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        results_path = Path(output_dir) / "plannotate_results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"  pLannotate failed: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="McClain/PlasmidLM-kmer6-GRPO-plannotate")
    parser.add_argument("--n-per-cell", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperatures", default="0.7,0.85,1.0")
    parser.add_argument("--top-ps", default="0.9,0.95")
    parser.add_argument("--rep-penalties", default="1.0,1.05")
    args = parser.parse_args()

    temps = [float(x) for x in args.temperatures.split(",")]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    rep_pens = [float(x) for x in args.rep_penalties.split(",")]
    grid = list(product(temps, top_ps, rep_pens))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"First-light sweep: {len(grid)} cells x {args.n_per_cell} seqs")
    print(f"Grid: temps={temps}, top_ps={top_ps}, rep_pens={rep_pens}")

    # Load model once
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<EOS>"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(args.device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Sample prompts (same set for all cells)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompt_df = pd.read_parquet("data/training_pairs_v4.parquet", columns=["prompt"])
    unique_prompts = prompt_df["prompt"].unique()
    sampled = np.random.choice(unique_prompts, size=min(args.n_per_cell, len(unique_prompts)), replace=False)

    results = []
    sweep_t0 = time.time()

    for temp, top_p, rep_pen in grid:
        cell_name = f"t{temp}_p{top_p}_r{rep_pen}"
        cell_dir = out / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {cell_name} ---")
        t0 = time.time()

        # Generate
        # Note: repetition_penalty not in generate_simple; it uses top_k+top_p
        gen_results = generate_cell(
            model, tokenizer, sampled, args.batch_size, 3000,
            temp, 50, top_p, args.device,
        )
        gen_time = time.time() - t0

        df = pd.DataFrame(gen_results)
        df.to_parquet(cell_dir / "generations.parquet")

        # Write FASTA
        with open(cell_dir / "generations.fasta", "w") as f:
            for i, row in df.iterrows():
                f.write(f">gen_{i}\n{row['sequence']}\n")

        # Fail-fast checks
        n_short = (df["length"] < 1000).sum()
        n_mono = 0
        for _, row in df.iterrows():
            seq = row["sequence"]
            if len(seq) > 0 and max(seq.count(b) for b in "ATGC") / len(seq) > 0.5:
                n_mono += 1

        # Quick pLannotate on 10 samples
        sample_fasta = cell_dir / "sample_10.fasta"
        with open(sample_fasta, "w") as f:
            for i, row in df.head(10).iterrows():
                f.write(f">gen_{i}\n{row['sequence']}\n")

        pn_results = quick_plannotate(str(sample_fasta), str(cell_dir), workers=4)
        median_coverage = None
        if pn_results:
            coverages = [
                r.get("coverage_bp", 0) / max(r.get("seq_length", 1), 1)
                for r in pn_results
            ]
            median_coverage = float(np.median(coverages))

        cell_result = {
            "cell": cell_name,
            "temperature": temp,
            "top_p": top_p,
            "repetition_penalty": rep_pen,
            "n_generated": len(df),
            "mean_length": float(df["length"].mean()),
            "median_length": float(df["length"].median()),
            "std_length": float(df["length"].std()),
            "mean_gc": float(df["gc"].mean()),
            "eos_rate": float(df["has_eos"].mean()),
            "n_under_1kb": int(n_short),
            "n_mononucleotide": int(n_mono),
            "median_plannotate_coverage": median_coverage,
            "gen_time_s": round(gen_time, 1),
        }
        results.append(cell_result)

        with open(cell_dir / "cell_summary.json", "w") as f:
            json.dump(cell_result, f, indent=2)

        cov_str = f"{median_coverage:.1%}" if median_coverage is not None else "?"
        print(f"  {len(df)} seqs in {gen_time:.0f}s | "
              f"len={df['length'].median():.0f} | gc={df['gc'].mean():.3f} | "
              f"eos={df['has_eos'].mean():.0%} | cov={cov_str} | "
              f"<1kb={n_short} mono={n_mono}")

    total_time = time.time() - sweep_t0

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "sweep_results.csv", index=False)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    cols = ["cell", "median_plannotate_coverage", "mean_length", "mean_gc", "eos_rate", "n_under_1kb", "n_mononucleotide"]
    avail_cols = [c for c in cols if c in results_df.columns]
    print(results_df[avail_cols].to_string(index=False))

    # Pick winner
    valid = results_df[results_df["n_mononucleotide"] <= 2].copy()
    if "median_plannotate_coverage" in valid.columns:
        valid = valid.dropna(subset=["median_plannotate_coverage"])
    if len(valid) > 0 and "median_plannotate_coverage" in valid.columns:
        valid = valid.sort_values("median_plannotate_coverage", ascending=False)
        winner = valid.iloc[0]["cell"]
        print(f"\nRecommended config: {winner}")
    else:
        winner = "manual review needed"
        print(f"\n{winner}")

    with open(out / "sweep_summary.json", "w") as f:
        json.dump({
            "total_time_min": round(total_time / 60, 1),
            "grid_size": len(grid),
            "n_per_cell": args.n_per_cell,
            "results": results,
            "recommendation": winner,
        }, f, indent=2)


if __name__ == "__main__":
    main()
