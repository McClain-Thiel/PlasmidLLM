#!/usr/bin/env python3
"""Build prompts for component combinations that exist in no training plasmid.

The disjoint benchmark (`build_benchmark.py`) derives its prompts from real held-out
plasmids, so every spec it asks for is satisfiable by *something* that already exists —
which is why retrieval scores ~95% on it. Only 6% of those prompts demand a combination
absent from the training corpus. That benchmark therefore cannot distinguish design from
lookup, and it is the wrong instrument for the paper's actual claim.

This builds the right one. Take a real held-out plasmid's component set and substitute one
component with another from the same category, keeping the result absent from every
training plasmid. That is the everyday cloning request — "this vector, but chloramphenicol
instead of ampicillin" — and it is the regime where retrieval is capped below full
satisfaction by construction.

Substitutes are drawn only from components that are individually well-supported
(>= --min-support occurrences in training). The paper already establishes that tokens
below roughly 100 occurrences are outside the model's actionable vocabulary, so drawing
rare substitutes would engineer a failure rather than test composition: we want both parts
individually learnable and only their *combination* new.

Sanity constraint: substitution stays within category (ORI for ORI, AMR for AMR), so the
result remains a coherent design rather than a plasmid with two origins and no marker.

Usage:
  python novel_combination.py --plannotate heldout_ann/plannotate_results.json \\
      --train-pairs training_pairs_v4.parquet --metadata metadata.parquet \\
      --pool heldout_2000.csv --n 500 --out novel_benchmark.parquet
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import duckdb
import pandas as pd

from build_benchmark import COPY_MAP, render
from common import TOKEN_RE, category_of, detect, load_annotations

SUBSTITUTABLE = ["AMR", "ORI", "REPORTER", "PROM", "ELEM"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plannotate", type=Path, required=True)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--allow-substitutes", type=Path, default=None,
                    help="file of allowed substitute tokens, one per line (producible-only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # training component sets + per-token support
    index: dict[int, set[str]] = collections.defaultdict(set)
    support: collections.Counter = collections.Counter()
    for pid, prompt in duckdb.sql(
        f"select plasmid_id, prompt from '{args.train_pairs}'").fetchall():
        toks = set(TOKEN_RE.findall(prompt))
        index[pid] |= toks
    for toks in index.values():
        support.update(toks)
    train_sets = [frozenset(v) for v in index.values() if v]
    print(f"training plasmids: {len(train_sets):,}")

    well_supported = {t for t, c in support.items() if c >= args.min_support}
    allow = None
    if args.allow_substitutes:
        allow = set(args.allow_substitutes.read_text().split())
    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for t in sorted(well_supported):
        if category_of(t) in SUBSTITUTABLE and (allow is None or t in allow):
            by_cat[category_of(t)].append(t)
    print(f"well-supported substitutes (>={args.min_support} plasmids): "
          + ", ".join(f"{c}:{len(v)}" for c, v in sorted(by_cat.items())))

    ann = load_annotations(args.plannotate)
    copy_of = {
        str(r.id): COPY_MAP.get(r.c) for r in duckdb.sql(
            f"select id, any_value(plasmid_copy) c from '{args.metadata}' group by 1"
        ).df().itertuples()
    }

    def unavailable(req: set[str]) -> bool:
        """No single training plasmid supplies every requested component."""
        return not any(req <= t for t in train_sets)

    rows = []
    for pid, feats in ann.items():
        base = detect(feats)
        if len(base) < 3:
            continue
        cats = [c for c in SUBSTITUTABLE
                if any(category_of(t) == c for t in base) and by_cat.get(c)]
        rng.shuffle(cats)
        for cat in cats:
            present = [t for t in base if category_of(t) == cat]
            old = rng.choice(present)
            options = [t for t in by_cat[cat] if t not in base]
            rng.shuffle(options)
            for new in options:
                cand = (base - {old}) | {new}
                if unavailable(cand):
                    toks = set(cand)
                    if copy_of.get(pid):
                        toks.add(copy_of[pid])
                    rows.append({
                        "plasmid_id": pid, "prompt": render(toks),
                        "n_hard_tokens": len(cand),
                        "substituted_out": old, "substituted_in": new,
                        "support_in": support[new],
                    })
                    break
            if rows and rows[-1]["plasmid_id"] == pid:
                break

    df = pd.DataFrame(rows)
    print(f"\nplasmids yielding a novel combination: {len(df)}")
    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=args.seed)
    df = df.sort_values("plasmid_id").reset_index(drop=True)

    pool = duckdb.sql(
        f"select plasmid_id, sequence_length from '{args.pool}'").df()
    pool["plasmid_id"] = pool.plasmid_id.astype(str)
    df["source_length"] = df.plasmid_id.map(pool.set_index("plasmid_id").sequence_length)
    df.to_parquet(args.out, index=False)

    print(f"wrote {len(df)} novel-combination prompts -> {args.out}")
    print(f"  mean hard tokens/prompt : {df.n_hard_tokens.mean():.2f}")
    print(f"  median support of the substituted-in token: "
          f"{df.support_in.median():.0f} training plasmids")
    print("\n  substitutions by category:")
    for c, k in collections.Counter(
            category_of(t) for t in df.substituted_in).most_common():
        print(f"    {c:<10}{k:>5}")
    print(f"\nBy construction no training plasmid supplies every requested component, so "
          f"retrieval\ncannot reach F=1.0 on any of these prompts.")


if __name__ == "__main__":
    main()
