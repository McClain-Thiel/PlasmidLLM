#!/usr/bin/env python3
"""Phase 1: First-light sweep to pick optimal sampling config.

Generates ~50-100 plasmids per cell across a sampling grid:
  temperature × top_p × repetition_penalty

Runs stripped-down eval (length, GC, pLannotate coverage, fail-fast checks)
and picks the 1-2 best cells.

Usage:
    python eval/scripts/firstlight_sweep.py \
        --model McClain/PlasmidLM-kmer6-GRPO-plannotate \
        --n-per-cell 50 \
        --output-dir eval/runs/firstlight_20260408
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def run_cell(model: str, n: int, temp: float, top_p: float, rep_pen: float,
             output_dir: Path, seed: int, device: str) -> dict:
    """Generate and do fail-fast eval for one grid cell."""
    cell_name = f"t{temp}_p{top_p}_r{rep_pen}"
    cell_dir = output_dir / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cell: {cell_name} ({n} sequences)")
    print(f"{'='*60}")

    # Generate
    gen_cmd = [
        sys.executable, "eval/scripts/generate.py",
        "--model", model,
        "--n", str(n),
        "--temperature", str(temp),
        "--top-p", str(top_p),
        "--repetition-penalty", str(rep_pen),
        "--output-dir", str(cell_dir),
        "--seed", str(seed),
        "--device", device,
    ]
    result = subprocess.run(gen_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Generation FAILED:\n{result.stderr[-500:]}")
        return {"cell": cell_name, "error": result.stderr[-200:]}
    print(result.stdout[-300:])

    # Load results
    gen_df = pd.read_parquet(cell_dir / "generations.parquet")

    # Fail-fast checks
    n_all_n = (gen_df["sequence"].str.count("N") > gen_df["length"] * 0.5).sum()
    n_mono = 0
    for _, row in gen_df.iterrows():
        seq = row["sequence"]
        if len(seq) > 0:
            max_mono = max(seq.count(b) for b in "ATGC") / len(seq)
            if max_mono > 0.5:
                n_mono += 1
    n_short = (gen_df["length"] < 1000).sum()

    # Quick pLannotate on a sample (10 sequences for speed)
    sample_df = gen_df.head(10)
    coverage_frac = None
    try:
        sample_fasta = cell_dir / "sample.fasta"
        with open(sample_fasta, "w") as f:
            for i, row in sample_df.iterrows():
                f.write(f">gen_{i}\n{row['sequence']}\n")

        ann_cmd = [
            "conda", "run", "-n", "plannotate",
            sys.executable, "eval/scripts/annotate.py",
            "--input", str(sample_fasta),
            "--output-dir", str(cell_dir),
            "--workers", "4",
            "--skip-prodigal", "--skip-dustmasker",
        ]
        ann_result = subprocess.run(ann_cmd, capture_output=True, text=True, timeout=600)
        if ann_result.returncode == 0 and (cell_dir / "plannotate_results.json").exists():
            with open(cell_dir / "plannotate_results.json") as f:
                pn_results = json.load(f)
            coverages = []
            for r in pn_results:
                sl = r.get("seq_length", 1)
                cb = r.get("coverage_bp", 0)
                coverages.append(cb / max(sl, 1))
            coverage_frac = float(np.median(coverages)) if coverages else None
    except Exception as e:
        print(f"  pLannotate sample failed: {e}")

    cell_result = {
        "cell": cell_name,
        "temperature": temp,
        "top_p": top_p,
        "repetition_penalty": rep_pen,
        "n_generated": len(gen_df),
        "mean_length": float(gen_df["length"].mean()),
        "median_length": float(gen_df["length"].median()),
        "mean_gc": float(gen_df["gc"].mean()),
        "eos_rate": float(gen_df["has_eos"].mean()),
        "n_all_N": int(n_all_n),
        "n_mononucleotide": int(n_mono),
        "n_under_1kb": int(n_short),
        "median_plannotate_coverage": coverage_frac,
    }

    with open(cell_dir / "cell_summary.json", "w") as f:
        json.dump(cell_result, f, indent=2)

    return cell_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="McClain/PlasmidLM-kmer6-GRPO-plannotate")
    parser.add_argument("--n-per-cell", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    # Grid parameters
    parser.add_argument("--temperatures", default="0.7,0.85,1.0")
    parser.add_argument("--top-ps", default="0.9,0.95")
    parser.add_argument("--rep-penalties", default="1.0,1.05")
    args = parser.parse_args()

    temps = [float(x) for x in args.temperatures.split(",")]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    rep_pens = [float(x) for x in args.rep_penalties.split(",")]

    grid = list(product(temps, top_ps, rep_pens))
    total_seqs = len(grid) * args.n_per_cell

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"First-light sweep: {len(grid)} cells × {args.n_per_cell} = {total_seqs} sequences")
    print(f"Grid: temps={temps}, top_ps={top_ps}, rep_pens={rep_pens}")

    results = []
    for temp, top_p, rep_pen in grid:
        cell_result = run_cell(
            args.model, args.n_per_cell, temp, top_p, rep_pen,
            out, args.seed, args.device,
        )
        results.append(cell_result)

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "sweep_results.csv", index=False)

    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    # Pick winner: closest median pLannotate coverage to reference, no degenerate cells
    ref_metrics = pd.read_csv("eval/reference/addgene_reference_metrics.csv")
    # Addgene doesn't have plannotate coverage pre-computed, but we know real plasmids
    # should have high coverage. Pick cells with highest coverage that aren't degenerate.
    valid = results_df[
        (results_df.get("n_all_N", 0) == 0) &
        (results_df.get("n_mononucleotide", 0) <= 2)
    ].copy()

    if "median_plannotate_coverage" in valid.columns:
        valid = valid.dropna(subset=["median_plannotate_coverage"])
        if len(valid) > 0:
            valid = valid.sort_values("median_plannotate_coverage", ascending=False)
            print(f"\nTop cells by pLannotate coverage:")
            print(valid[["cell", "median_plannotate_coverage", "mean_length", "mean_gc", "eos_rate"]].head(3).to_string(index=False))

    with open(out / "sweep_summary.json", "w") as f:
        json.dump({
            "grid_size": len(grid),
            "n_per_cell": args.n_per_cell,
            "results": results,
            "recommendation": valid.iloc[0]["cell"] if len(valid) > 0 else "manual review needed",
        }, f, indent=2)


if __name__ == "__main__":
    main()
