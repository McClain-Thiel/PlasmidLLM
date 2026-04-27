#!/usr/bin/env python3
"""Build a matched_pairs.parquet for the base run that mirrors the GRPO Evo2 set.

The GRPO run was Evo2-scored on a 200-pair sample (`runs/grpo_plannotate_full_*/evo2/
evo2_scores.parquet`). To get an apples-to-apples base-vs-GRPO Evo2 comparison we need
to score the same 200 prompt indices with the base model's generation, paired against
the same real Addgene plasmid that was used as the matched-real reference.

This script:
  - Reads the GRPO matched_pairs.parquet (1000 rows, gen_id 0..999, prompt, gen_sequence,
    real_plasmid_id, real_sequence, ...).
  - Reads the GRPO evo2_scores.parquet to discover which 200 gen_ids were scored.
  - Reads the base run's generations.parquet (1000 rows; index aligned to GRPO via
    identical prompt order).
  - Builds a 200-row parquet whose `gen_sequence` / `gen_length` come from the base
    model and whose `real_*` columns are unchanged from GRPO's pairing — so each row
    has (base_gen, real) for the same 200 prompts GRPO was scored on.

Usage:
  python eval/scripts/build_base_matched_pairs.py \\
      --grpo-pairs runs/grpo_plannotate_full_20260408/matched_pairs.parquet \\
      --grpo-evo2 runs/grpo_plannotate_full_20260408/evo2/evo2_scores.parquet \\
      --base-generations runs/base_kmer6_matched_20260420_1303/generations.parquet \\
      --output runs/base_kmer6_matched_20260420_1303/matched_pairs_evo2.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-pairs", type=Path, required=True)
    parser.add_argument("--grpo-evo2", type=Path, required=True,
                        help="evo2_scores.parquet from the GRPO Evo2 run")
    parser.add_argument("--base-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grpo = pd.read_parquet(args.grpo_pairs)
    evo2 = pd.read_parquet(args.grpo_evo2)
    base_gen = pd.read_parquet(args.base_generations)

    if len(grpo) != len(base_gen):
        raise ValueError(
            f"grpo_pairs ({len(grpo)}) and base_generations ({len(base_gen)}) "
            "have different lengths; expected 1:1 prompt alignment"
        )

    # Sanity: same prompt order on both sides.
    if not (grpo["prompt"].values == base_gen["prompt"].values).all():
        raise ValueError(
            "prompt order mismatch between grpo_pairs and base_generations — "
            "Evo2 comparison would not be apples-to-apples"
        )

    # gen_id in grpo is 0..N-1, identical to row index in base_gen.
    keep_ids = sorted(int(x) for x in evo2["gen_id"].unique())
    keep = grpo[grpo["gen_id"].isin(keep_ids)].copy().reset_index(drop=True)
    if len(keep) != len(keep_ids):
        raise ValueError(
            f"some gen_ids in evo2 ({len(keep_ids)}) not present in grpo_pairs "
            f"({len(keep)} matched). Cannot continue."
        )

    # Replace gen_sequence / gen_length with base model's output for the same prompt.
    base_indexed = base_gen.iloc[keep_ids].reset_index(drop=True)
    keep["gen_sequence"] = base_indexed["sequence"].values
    keep["gen_length"] = base_indexed["length"].values

    # Sanity: prompt still matches per row
    keep_prompts = keep["prompt"].values
    base_prompts = base_indexed["prompt"].values
    if not (keep_prompts == base_prompts).all():
        raise ValueError("post-replace prompt mismatch — internal bug")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(args.output, index=False)

    print(f"wrote {len(keep)} pairs to {args.output}")
    print(f"  base mean gen_length: {keep['gen_length'].mean():.0f} bp")
    print(f"  real mean length:     {keep['real_length'].mean():.0f} bp")


if __name__ == "__main__":
    main()
