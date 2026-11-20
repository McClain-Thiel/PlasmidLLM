#!/usr/bin/env python3
"""Build source-ID benchmarks or freeze an independently audited cluster test panel.

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
prompts change the conditioning distribution. The direction of that effect is unknown.
Identical prompts support a matched checkpoint comparison on this panel; neither absolute
rates nor the delta necessarily transfer to the original conditioning distribution.
Faithfulness is defined over hard tokens, which excludes the omitted classes.

The source-ID mode excludes exact plasmid IDs, not close sequence relatives. It must not
be called sequence-cluster-disjoint. The separate --cluster-split mode requires an audit
matching all three partition checksums, and uses only the test partition.

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
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

try:
    from experiments.neurips_rebuttal.common import TOKEN_RE, detect, load_annotations
except ImportError:
    from common import TOKEN_RE, detect, load_annotations

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


def build_cluster_panel(
    split_dir: Path, output: Path, n: int = 1000, minority_n: int = 100, seed: int = 42,
) -> dict:
    """Freeze separate overall/minority panels without selecting on model outcomes."""
    from pretraining.build_cluster_split import MINORITY_TOKENS, sha256_file

    if min(n, minority_n) < 1:
        raise ValueError("Panel limits must be positive")
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Panel output already exists: {output} or {report_path}")
    audit_path = split_dir / "audit/report.json"
    audit = json.loads(audit_path.read_text())
    if audit["passed"] is not True:
        raise ValueError("Cluster partition audit did not pass")
    for partition in ("train", "validation", "test"):
        if sha256_file(split_dir / f"{partition}.parquet") != audit["partition_sha256"][partition]:
            raise ValueError(f"Audit does not match {partition} data")
    rows = pd.read_parquet(split_dir / "test.parquet")
    required = {"cluster_id", "sequence_id", "plasmid_id", "prompt", "sequence"}
    if not required <= set(rows):
        raise ValueError(f"Test partition lacks columns: {sorted(required - set(rows))}")
    if rows[list(required)].isna().any().any():
        raise ValueError("Test partition contains missing identifiers, prompts or sequences")
    rows["n_hard_tokens"] = rows.prompt.map(lambda p: len(set(TOKEN_RE.findall(p))))
    rows = rows[rows.n_hard_tokens > 0].copy()
    if rows.empty:
        raise ValueError("No hard-token requests in the test partition")
    rows["plasmid_id"] = rows.plasmid_id.astype(str)
    rows["n_prompt_tokens"] = rows.prompt.map(lambda p: len(set(re.findall(r"<[^>]+>", p))))
    # Multiple prompt subsets must not turn the same source into extra evaluation units.
    rows = rows.sort_values(
        ["sequence_id", "n_hard_tokens", "n_prompt_tokens", "prompt", "plasmid_id"],
        ascending=[True, False, False, True, True],
    ).drop_duplicates("sequence_id")
    rows["sample_id"] = [hashlib.sha256(f"{r.sequence_id}:{r.prompt}".encode()).hexdigest()
                         for r in rows.itertuples()]
    rows["cluster_rank"] = rows.cluster_id.map(
        lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    rows["sample_rank"] = rows.sample_id.map(
        lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    rows = rows.sort_values(["cluster_rank", "sample_rank"])
    main = rows.drop_duplicates("cluster_id").head(n)
    selected = set(main.sample_id)
    minority_members = {}
    counts = {}
    for token in MINORITY_TOKENS:
        eligible = rows[rows.prompt.str.contains(token, regex=False)].drop_duplicates("cluster_id")
        panel = eligible.head(minority_n)
        minority_members[token] = set(panel.sample_id)
        selected.update(panel.sample_id)
        counts[token] = dict(eligible_clusters=len(eligible), selected_clusters=len(panel),
                             requested_clusters=minority_n, below_target=len(panel) < minority_n)
    panel = rows[rows.sample_id.isin(selected)].copy()
    panel["panel_main"] = panel.sample_id.isin(set(main.sample_id))
    panel["minority_panels"] = panel.sample_id.map(
        lambda s: [token for token, members in minority_members.items() if s in members])
    panel = panel.rename(columns={"sequence": "source_sequence"})
    panel["source_length"] = panel.source_sequence.str.len()
    panel = panel.drop(columns=["cluster_rank", "sample_rank", "n_prompt_tokens"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        panel.to_parquet(f, index=False)
    report = dict(
        seed=seed, rows=len(panel), source_clusters=panel.cluster_id.nunique(),
        main_clusters=len(main), main_requested_clusters=n, minority_panels=counts,
        selection="One full prompt per sequence; hash-selected example per cluster in each panel",
        aggregation="Report main and minority panels separately; bootstrap source clusters",
        partition_sha256=audit["partition_sha256"], audit_sha256=sha256_file(audit_path),
        panel_sha256=sha256_file(output),
    )
    with report_path.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-split", type=Path, help="Audited split; use its test partition")
    ap.add_argument("--minority-n", type=int, default=100)
    ap.add_argument("--plannotate", type=Path)
    ap.add_argument("--metadata", type=Path)
    ap.add_argument(
        "--train-pairs",
        type=Path,
        help="Pretraining pairs used to verify that every source plasmid ID is absent",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        help="CSV of the annotated plasmids: plasmid_id, sequence, sequence_length",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    legacy_inputs = (args.plannotate, args.metadata, args.train_pairs, args.pool)
    if args.cluster_split:
        if any(path is not None for path in legacy_inputs):
            ap.error("--cluster-split cannot be combined with source-ID benchmark inputs")
        print(json.dumps(build_cluster_panel(
            args.cluster_split, args.out, args.n, args.minority_n, args.seed,
        ), indent=2))
        return
    if any(path is None for path in legacy_inputs):
        ap.error("Provide --cluster-split or all of --plannotate, --metadata, --train-pairs, --pool")

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
