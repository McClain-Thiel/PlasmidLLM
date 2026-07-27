#!/usr/bin/env python3
"""Build a source-plasmid-ID-disjoint benchmark from Addgene plasmids.

Written for the NeurIPS 2026 rebuttal (meta-review point 2; Dok5 W5/Q1; pSXh W6).

Why this exists: the benchmark reported in the paper was sampled from
`training_pairs_v4.parquet`, the pretraining corpus (`eval/scripts/generate.py:86-95` does
no holdout filtering). The designated `holdout_full_prompts.parquet` would not have helped
either — all 22,392 of its distinct sequences also appear in the training corpus, and
20,562 of the 20,588 plasmids behind it were in pretraining. It is a pair-level reshuffle
of the same plasmids. A source-ID-disjoint benchmark therefore has to be built from the
larger sequence table and checked against the pretraining IDs.

It can be: `full_sequences.parquet` holds 141,093 distinct plasmids against the 20,644 used
in pretraining, leaving ~120k unseen.

Prompt construction mirrors the training pipeline, restricted to the parts that reproduce
exactly:

  hard tokens  pLannotate Feature -> token via the repo's own `feature_to_category_token`,
               with the registry's documented quality filter (percmatch >= 95, non-fragment).
               Requires a FRESH pLannotate run — validated against known training prompts,
               deriving these from the archived `annotations.parquet` gives 99.9% precision
               but only 76.4% recall, because that dump covers 14.6k of 20.6k training
               plasmids. Using it would systematically under-request tokens.
  COPY_*       from `metadata.plasmid_copy`. Validated: reproduces the COPY token exactly
               for 100.0% of all 20,644 training plasmids.

VEC_*, SP_* and BB_* are deliberately omitted: no rule reproduces them reliably (VEC via the
`splits` tags reaches only 76.2%, and `splits` does not exist for unseen plasmids; a
backbone-slug rule for BB_* matches 8.8%). Guessing them would put wrong tokens in the
prompt, which is worse than leaving them out.

Consequence to report honestly: ~86.7% of training prompts carry a VEC_* token, so these
prompts are mildly out-of-distribution and absolute rates should be read as a LOWER BOUND.
Both checkpoints receive identical prompts, so the base-vs-GRPO delta — the paper's actual
claim — is unaffected. Faithfulness is defined over hard tokens only in any case, so the
omitted classes are outside the metric by construction.

This excludes exact source plasmid IDs, not close sequence relatives. The resulting panel
must not be called sequence-cluster-disjoint.

Usage:
  python build_benchmark.py \\
      --plannotate heldout_ann/plannotate_results.json \\
      --metadata metadata.parquet \\
      --train-pairs training_pairs_v4.parquet \\
      --pool heldout_2000.csv \\
      --n 1000 --out disjoint_benchmark.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from common import detect, load_annotations

# Order tokens the way training prompts do, so the prompt string is in-distribution.
CATEGORY_ORDER = ["VEC", "SP", "COPY", "AMR", "ORI", "PROM", "ELEM", "REPORTER", "TAG"]
COPY_MAP = {"High Copy": "<COPY_HIGH>", "Low Copy": "<COPY_LOW>"}


def category_of(token: str) -> str:
    return token.strip("<>").split("_")[0]


def render(tokens: set[str]) -> str:
    ordered = sorted(
        tokens,
        key=lambda t: (
            CATEGORY_ORDER.index(category_of(t)) if category_of(t) in CATEGORY_ORDER else 99,
            t,
        ),
    )
    return "<BOS>" + "".join(ordered) + "<SEQ>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plannotate", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument(
        "--train-pairs",
        type=Path,
        required=True,
        help="Pretraining pairs used to verify that every source plasmid ID is absent",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        required=True,
        help="CSV of the annotated plasmids: plasmid_id, sequence, sequence_length",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    annotations = load_annotations(args.plannotate)
    print(f"annotated sequences: {len(annotations)}")

    pool = (
        duckdb.connect()
        .execute(
            """
        SELECT CAST(plasmid_id AS VARCHAR) AS plasmid_id, sequence, sequence_length
        FROM read_csv_auto(?)
        """,
            [str(args.pool)],
        )
        .df()
    )
    if pool["plasmid_id"].duplicated().any():
        duplicated = pool.loc[pool["plasmid_id"].duplicated(), "plasmid_id"].tolist()
        raise ValueError(f"Candidate pool has duplicate plasmid IDs: {duplicated[:5]}")
    train_ids = {
        row[0]
        for row in duckdb.connect()
        .execute(
            """
            SELECT DISTINCT CAST(plasmid_id AS VARCHAR)
            FROM read_parquet(?)
            """,
            [str(args.train_pairs)],
        )
        .fetchall()
    }
    overlap = sorted(set(pool["plasmid_id"]) & train_ids)
    if overlap:
        raise ValueError(
            f"Candidate pool contains {len(overlap)} pretraining plasmid IDs; first={overlap[:5]}"
        )
    # annotate.py keys results by FASTA record id, which is the plasmid_id here.
    ids = list(annotations)

    copy_of = (
        duckdb.connect()
        .execute(
            """
        SELECT CAST(id AS VARCHAR) AS id, ANY_VALUE(plasmid_copy) AS c
        FROM read_parquet(?)
        GROUP BY 1
        """,
            [str(args.metadata)],
        )
        .df()
    )
    copy_of = {str(r.id): COPY_MAP.get(r.c) for r in copy_of.itertuples()}

    rows = []
    for pid in ids:
        hard = detect(annotations[pid])
        if not hard:
            continue  # faithfulness is undefined without a requested hard token
        toks = set(hard)
        if copy_of.get(pid):
            toks.add(copy_of[pid])
        rows.append(
            {
                "plasmid_id": pid,
                "prompt": render(toks),
                "n_hard_tokens": len(hard),
            }
        )

    df = pd.DataFrame(rows)
    print(f"plasmids with >=1 hard token: {len(df)}")
    if len(df) < args.n:
        raise ValueError(f"Requested {args.n} prompts, but only {len(df)} are eligible")
    df = (
        df.sample(n=args.n, random_state=args.seed).sort_values("plasmid_id").reset_index(drop=True)
    )

    src = pool.set_index("plasmid_id")
    df["source_sequence"] = df.plasmid_id.map(src.sequence)
    df["source_length"] = df.plasmid_id.map(src.sequence_length)
    if df["source_sequence"].isna().any():
        missing = df.loc[df["source_sequence"].isna(), "plasmid_id"].tolist()
        raise ValueError(f"Missing source sequences for plasmid IDs: {missing[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {len(df)} prompts -> {args.out}")
    print(f"  mean hard tokens/prompt   : {df.n_hard_tokens.mean():.2f}")
    print(f"  median source plasmid (bp): {df.source_length.median():.0f}")
    print(f"  distinct prompts          : {df.prompt.nunique()}")
    print(f"\nexample: {df.prompt.iloc[0]}")


if __name__ == "__main__":
    main()
