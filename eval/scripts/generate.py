#!/usr/bin/env python3
"""Generate plasmid sequences from a PlasmidLM checkpoint.

Uses the model's optimized generate_simple() with KV cache.
Saves FASTA + parquet with per-sequence metadata.

Usage:
    python eval/scripts/generate.py \
        --model McClain/PlasmidLM-kmer6-GRPO-plannotate \
        --n 100 --temperature 0.85 --top-p 0.95 \
        --output-dir eval/runs/firstlight_20260408
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def clean_dna(raw: str) -> str:
    seq = re.sub(r"<[^>]+>", "", raw.upper())
    return re.sub(r"[^ATGCN]", "", seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="McClain/PlasmidLM-kmer6-GRPO-plannotate")
    parser.add_argument("--prompts", default="data/training_pairs_v4.parquet",
                        help="Parquet with 'prompt' column to sample from")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = vars(args).copy()
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model + tokenizer (trust_remote_code for custom PlasmidLM)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Custom PlasmidKmerTokenizer doesn't expose special tokens properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<EOS>"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<BOS>"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {n_params:.1f}M params on {args.device}")

    # Sample unique prompts
    prompt_df = pd.read_parquet(args.prompts, columns=["prompt"])
    unique_prompts = prompt_df["prompt"].unique()
    sampled = np.random.choice(
        unique_prompts, size=min(args.n, len(unique_prompts)), replace=False,
    )
    print(f"Sampled {len(sampled)} prompts from {len(unique_prompts)} unique")

    # Generate
    all_results = []
    t0 = time.time()

    for batch_start in range(0, len(sampled), args.batch_size):
        batch_prompts = list(sampled[batch_start:batch_start + args.batch_size])
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)

        with torch.inference_mode():
            outputs = model.generate_simple(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
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
                "prompt": batch_prompts[i],
                "sequence": dna,
                "length": len(dna),
                "gc": (dna.count("G") + dna.count("C")) / max(len(dna), 1),
                "has_eos": has_eos,
                "seed": args.seed,
            })

        done = len(all_results)
        elapsed = time.time() - t0
        batch_lens = [r["length"] for r in all_results[-len(batch_prompts):]]
        print(f"  [{done}/{len(sampled)}] {done/elapsed*60:.1f} seq/min, "
              f"batch avg = {np.mean(batch_lens):.0f} bp")

    total_time = time.time() - t0

    # Save
    df = pd.DataFrame(all_results)
    df.index.name = "gen_id"
    df.to_parquet(out / "generations.parquet")

    with open(out / "generations.fasta", "w") as f:
        for i, row in df.iterrows():
            f.write(f">gen_{i}\n{row['sequence']}\n")

    summary = {
        "n_generated": len(df),
        "mean_length": float(df["length"].mean()),
        "median_length": float(df["length"].median()),
        "mean_gc": float(df["gc"].mean()),
        "eos_rate": float(df["has_eos"].mean()),
        "pct_lt_1kb": float((df["length"] < 1000).mean()),
        "total_time_s": round(total_time, 1),
        "seqs_per_min": round(len(df) / (total_time / 60), 1),
    }
    with open(out / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(df)} sequences in {total_time:.1f}s")
    print(f"  Mean length: {summary['mean_length']:.0f} bp")
    print(f"  Mean GC: {summary['mean_gc']:.3f}")
    print(f"  EOS rate: {summary['eos_rate']:.1%}")
    print(f"  <1kb: {summary['pct_lt_1kb']:.1%}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
