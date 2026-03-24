#!/usr/bin/env python
"""Generate single-token (or N-token) curriculum data for BinaryMotifScorer training.

Reads training_pairs_v4.parquet and explodes each row into one row per hard token,
so the model learns individual motif→sequence mappings before tackling multi-token prompts.

Usage:
    python scripts/generate_curriculum_data.py                          # default: 1-tok
    python scripts/generate_curriculum_data.py --max-tokens 2           # 2-tok combos
    python scripts/generate_curriculum_data.py --input path/to/data.parquet
"""

import argparse
import random
import re
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import pandas as pd

HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}


def extract_hard_tokens(prompt: str) -> list[str]:
    """Extract hard annotation tokens from a prompt string."""
    tokens = re.findall(r"<[^>]+>", prompt)
    return [t for t in tokens if any(t.strip("<>").startswith(p) for p in HARD_PREFIXES)]


def main():
    parser = argparse.ArgumentParser(description="Generate curriculum training data")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "training_pairs_v4.parquet"),
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file (default: auto-named based on --max-tokens)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Max number of hard tokens per prompt (default: 1)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Min number of hard tokens per prompt (default: same as max-tokens for exact N-tok)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap total rows (sample after dedup). Useful for large N-tok stages.",
    )
    args = parser.parse_args()

    if args.min_tokens is None:
        args.min_tokens = args.max_tokens

    output_path = args.output or str(
        Path(__file__).resolve().parent.parent
        / "data"
        / f"training_pairs_binary_motif_{args.max_tokens}tok.parquet"
    )

    print(f"Reading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Determine prompt column
    if "prompt" in df.columns:
        prompt_col = "prompt"
    elif "token_prompt" in df.columns:
        prompt_col = "token_prompt"
    else:
        print(f"ERROR: No prompt column found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Determine sequence column
    if "sequence" in df.columns:
        seq_col = "sequence"
    elif "token_completion" in df.columns:
        seq_col = "token_completion"
    else:
        print(f"ERROR: No sequence column found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # For large N, estimate total combos to decide whether to enumerate or sample
    max_combos_per_row = args.max_rows // max(len(df), 1) + 1 if args.max_rows else None

    rng = random.Random(42)
    rows = []
    for _, row in df.iterrows():
        prompt_str = row[prompt_col]
        sequence = row[seq_col]
        hard_tokens = extract_hard_tokens(prompt_str)
        if not hard_tokens:
            continue

        for n in range(args.min_tokens, min(args.max_tokens, len(hard_tokens)) + 1):
            total_combos = comb(len(hard_tokens), n)

            # If total combos is small or no cap, enumerate all
            if max_combos_per_row is None or total_combos <= max_combos_per_row:
                combos = combinations(hard_tokens, n)
            else:
                # Random sample: pick random subsets instead of enumerating all
                seen = set()
                sampled = []
                for _ in range(max_combos_per_row * 2):  # oversample to handle dupes
                    combo = tuple(sorted(rng.sample(hard_tokens, n)))
                    if combo not in seen:
                        seen.add(combo)
                        sampled.append(combo)
                    if len(sampled) >= max_combos_per_row:
                        break
                combos = sampled

            for combo in combos:
                new_prompt = "<BOS>" + "".join(combo) + "<SEQ>"
                rows.append({
                    "prompt": new_prompt,
                    "sequence": sequence,
                    "has_hard_tokens": True,
                    "reward_motifs": prompt_str,
                })

    out_df = pd.DataFrame(rows)

    # Deduplicate: same prompt + sequence is redundant
    before = len(out_df)
    out_df = out_df.drop_duplicates(subset=["prompt", "sequence"])
    after = len(out_df)
    if before != after:
        print(f"  Deduplicated: {before} → {after} rows")

    # Cap total rows if requested
    if args.max_rows and len(out_df) > args.max_rows:
        out_df = out_df.sample(n=args.max_rows, random_state=42)
        print(f"  Sampled down to {args.max_rows} rows")

    # Stats
    unique_prompts = out_df["prompt"].nunique()
    token_counts = out_df["prompt"].apply(lambda p: len(extract_hard_tokens(p)))
    print(f"\nOutput: {len(out_df)} rows, {unique_prompts} unique prompts")
    print(f"Tokens per prompt: min={token_counts.min()}, max={token_counts.max()}, "
          f"mean={token_counts.mean():.1f}")

    # Per-token distribution
    all_tokens = []
    for p in out_df["prompt"]:
        all_tokens.extend(extract_hard_tokens(p))
    token_series = pd.Series(all_tokens)
    print(f"\n{len(token_series.unique())} unique hard tokens:")
    for tok, count in token_series.value_counts().head(20).items():
        print(f"  {tok}: {count}")
    if len(token_series.unique()) > 20:
        print(f"  ... and {len(token_series.unique()) - 20} more")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
