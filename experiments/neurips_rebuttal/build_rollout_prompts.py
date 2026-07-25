#!/usr/bin/env python3
"""Build a GRPO rollout prompt set that includes novel component combinations.

The F16 fix hypothesis: the model reverts to the training co-occurrence because it has never
seen a novel combination during post-training — rollout prompts come from training_pairs_v4,
where the co-occurring marker always *is* the requested one, so the recall reward never has
occasion to push the substitute over the prior. Give it novel combinations to roll out on and
the recall reward optimizes exactly the failing behaviour (produce the requested substitute).

No reference sequence is needed: the reward is registry-BLAST on the generation against the
requested token set, so a rollout prompt is just a token set. We mix:
  - natural prompts (a sample of real training prompts) so learned behaviour is retained
  - novel prompts (substitute one well-supported component, verified absent from every
    training plasmid) so the model is trained on the composition it currently fails

Output is a parquet with a `prompt` column — all the runner's _load_prompts needs.

Usage:
  python build_rollout_prompts.py \\
      --train-pairs training_pairs_v4.parquet --metadata metadata.parquet \\
      --n-novel 4000 --n-natural 4000 --out rollout_novel_aug.parquet
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import duckdb
import pandas as pd

from build_benchmark import COPY_MAP, render
from common import TOKEN_RE, category_of
from novel_combination import SUBSTITUTABLE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--n-novel", type=int, default=4000)
    ap.add_argument("--n-natural", type=int, default=4000)
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--allow-substitutes", type=Path, default=None,
                    help="file of allowed substitute tokens, one per line (producible-only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    index: dict[int, set[str]] = collections.defaultdict(set)
    support: collections.Counter = collections.Counter()
    natural_prompts = []
    for pid, prompt in duckdb.sql(
        f"select plasmid_id, prompt from '{args.train_pairs}'").fetchall():
        toks = set(TOKEN_RE.findall(prompt))
        index[pid] |= toks
        natural_prompts.append(prompt)
    for toks in index.values():
        support.update(toks)
    train_sets = [frozenset(v) for v in index.values() if v]

    allow = None
    if args.allow_substitutes:
        allow = set(args.allow_substitutes.read_text().split())
    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for t, c in support.items():
        if c >= args.min_support and category_of(t) in SUBSTITUTABLE and (allow is None or t in allow):
            by_cat[category_of(t)].append(t)

    copy_of = {str(r.id): COPY_MAP.get(r.c) for r in duckdb.sql(
        f"select id, any_value(plasmid_copy) c from '{args.metadata}' group by 1"
    ).df().itertuples()}

    def novel(req: set[str]) -> bool:
        return not any(req <= t for t in train_sets)

    novel_rows = []
    pids = [p for p, v in index.items() if len(v) >= 3]
    rng.shuffle(pids)
    for pid in pids:
        if len(novel_rows) >= args.n_novel:
            break
        base = set(index[pid])
        cats = [c for c in SUBSTITUTABLE
                if any(category_of(t) == c for t in base) and by_cat.get(c)]
        rng.shuffle(cats)
        for cat in cats:
            old = rng.choice([t for t in base if category_of(t) == cat])
            for new in rng.sample(by_cat[cat], len(by_cat[cat])):
                if new in base:
                    continue
                cand = (base - {old}) | {new}
                if novel(cand):
                    toks = set(cand)
                    if copy_of.get(str(pid)):
                        toks.add(copy_of[str(pid)])
                    novel_rows.append(render(toks))
                    break
            else:
                continue
            break

    natural = rng.sample(natural_prompts, min(args.n_natural, len(natural_prompts)))
    rows = [{"prompt": p, "kind": "novel"} for p in novel_rows] + \
           [{"prompt": p, "kind": "natural"} for p in natural]
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)

    print(f"wrote {len(df)} rollout prompts -> {args.out}")
    print(f"  novel:   {sum(df.kind=='novel')}")
    print(f"  natural: {sum(df.kind=='natural')}")
    print(f"  mean hard tokens: "
          f"{df.prompt.map(lambda p: len(TOKEN_RE.findall(p))).mean():.2f}")


if __name__ == "__main__":
    main()
