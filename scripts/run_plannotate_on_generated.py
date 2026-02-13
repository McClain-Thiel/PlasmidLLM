#!/usr/bin/env python3
"""Run pLannotate on generated sequences from inference CSV."""

import csv
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import pandas as pd
from plannotate.annotate import annotate


def clean_sequence(seq: str) -> str:
    """Strip special tokens and whitespace, keep only DNA bases."""
    # Remove common special tokens
    for tok in ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<UNK>"]:
        seq = seq.replace(tok, "")
    # Keep only ATCGN (upper and lower)
    return "".join(c for c in seq if c.upper() in "ATCGN")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "mamba_inference_full.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "plannotate_results"

    os.makedirs(output_dir, exist_ok=True)

    # Read inference CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from {csv_path}")

    summaries = []

    for i, row in enumerate(rows):
        idx = row["idx"]
        generated = clean_sequence(row["generated"])
        true_seq = clean_sequence(row["true_completion"])

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(rows)}] Sample idx={idx}")
        print(f"  Generated length: {len(generated)} bp")
        print(f"  True length:      {len(true_seq)} bp")

        if len(generated) < 100:
            print("  SKIP: generated sequence too short")
            continue

        # Annotate generated sequence
        try:
            t0 = time.time()
            gen_df = annotate(generated, is_detailed=True)
            gen_time = time.time() - t0

            gen_csv = os.path.join(output_dir, f"gen_{idx}.csv")
            gen_df.to_csv(gen_csv, index=False)

            n_features = len(gen_df)
            feature_types = gen_df["Feature"].tolist() if "Feature" in gen_df.columns else []
            print(f"  Generated: {n_features} features found in {gen_time:.1f}s")
            if n_features > 0:
                print(f"  Features: {', '.join(feature_types[:10])}")

        except Exception as e:
            print(f"  ERROR annotating generated: {e}")
            n_features = 0
            feature_types = []

        # Annotate true sequence for comparison
        try:
            t0 = time.time()
            true_df = annotate(true_seq, is_detailed=True)
            true_time = time.time() - t0

            true_csv = os.path.join(output_dir, f"true_{idx}.csv")
            true_df.to_csv(true_csv, index=False)

            n_true_features = len(true_df)
            true_feature_types = true_df["Feature"].tolist() if "Feature" in true_df.columns else []
            print(f"  True:      {n_true_features} features found in {true_time:.1f}s")
            if n_true_features > 0:
                print(f"  Features: {', '.join(true_feature_types[:10])}")

        except Exception as e:
            print(f"  ERROR annotating true: {e}")
            n_true_features = 0
            true_feature_types = []

        summaries.append({
            "idx": idx,
            "gen_len": len(generated),
            "true_len": len(true_seq),
            "gen_features": n_features,
            "true_features": n_true_features,
            "gen_feature_list": "; ".join(feature_types),
            "true_feature_list": "; ".join(true_feature_types),
        })

    # Write summary
    summary_path = os.path.join(output_dir, "summary.csv")
    if summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\n\nSummary written to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'idx':>6} {'gen_bp':>8} {'true_bp':>8} {'gen_feat':>10} {'true_feat':>10}")
    for s in summaries:
        print(f"{s['idx']:>6} {s['gen_len']:>8} {s['true_len']:>8} {s['gen_features']:>10} {s['true_features']:>10}")

    total_gen = sum(s["gen_features"] for s in summaries)
    total_true = sum(s["true_features"] for s in summaries)
    print(f"\nTotal features: generated={total_gen}, true={total_true}")


if __name__ == "__main__":
    main()
