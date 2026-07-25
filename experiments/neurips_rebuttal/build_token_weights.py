#!/usr/bin/env python3
"""Inverse-frequency token weights for the rarity-weighted GRPO reward (attempt 4, F21 fix).

Weight w_t = clip(1 / base_rate(t), 1, cap), base_rate(t) = fraction of training plasmids
containing token t. Common backbone tokens (ORI_COLE1 ~94% -> ~1.0) barely count extra;
rare requested components (ZEOCIN ~4% -> cap) dominate the recall term, giving the specific
gradient a uniform recall reward lacks. Consumed by PlannotateScorer(token_weights_path=...).

  python build_token_weights.py --train-pairs training_pairs_v4.parquet --cap 12 --out token_weights.json
"""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path
import duckdb

TOK = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--cap", type=float, default=12.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    idx: dict[int, set[str]] = collections.defaultdict(set)
    for pid, prompt in duckdb.sql(
        f"select plasmid_id, prompt from '{args.train_pairs}'").fetchall():
        idx[pid] |= set(TOK.findall(prompt))
    n = len(idx)
    freq = collections.Counter()
    for toks in idx.values():
        freq.update(toks)
    weights = {t: round(max(1.0, min(n / c, args.cap)), 3) for t, c in freq.items()}
    args.out.write_text(json.dumps(weights, indent=1))
    print(f"wrote {len(weights)} token weights (cap={args.cap}) -> {args.out}")

if __name__ == "__main__":
    main()
