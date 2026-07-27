#!/usr/bin/env python3
"""Best-of-K useful rate, oracle ranker (the paper's protocol).

K draws per prompt; keep the candidate maximising n_found, ties broken viable-then-faithful.
useful = viable AND faithful@0.5 on the kept candidate. Pass the single-shot annotation as
the first --draws entry, then the extra-seed annotations.

Usage:
  python score_bestofk.py --benchmark disjoint_benchmark.parquet \\
      --draws base=dj_base.json,bok_base_s43.json,bok_base_s44.json,bok_base_s45.json \\
      --draws grpo=dj_grpo.json,bok_grpo_s43.json,bok_grpo_s44.json,bok_grpo_s45.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from common import TOKEN_RE, detect, is_viable, load_annotations, wilson


def candidate(feats: list[dict], req: set[str]) -> tuple[bool, float, int]:
    det = detect(feats)
    n_found = len(req & det)
    return (is_viable(feats), n_found / len(req) if req else 0.0, n_found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument(
        "--draws", action="append", required=True, metavar="LABEL=draw1.json,draw2.json,..."
    )
    ap.add_argument("--ks", default="1,2,4")
    args = ap.parse_args()

    requested = [
        set(TOKEN_RE.findall(r[0]))
        for r in duckdb.sql(f"select prompt from '{args.benchmark}'").fetchall()
    ]
    ks = [int(k) for k in args.ks.split(",")]

    for spec in args.draws:
        label, separator, paths = spec.partition("=")
        if not separator or not label or not paths:
            raise ValueError(f"Invalid --draws value: {spec!r}")
        annotation_paths = paths.split(",")
        anns = [load_annotations(path) for path in annotation_paths]
        print(f"\n{label}:")
        for K in ks:
            if K > len(anns):
                raise ValueError(
                    f"best-of-{K} requested for {label!r}, but only "
                    f"{len(anns)} annotation panels were supplied"
                )
            useful = n = 0
            for i, req in enumerate(requested):
                if not req:
                    raise ValueError(f"Benchmark prompt {i} has no hard tokens")
                missing = [
                    annotation_paths[j]
                    for j, annotations in enumerate(anns[:K])
                    if f"gen_{i}" not in annotations
                ]
                if missing:
                    raise ValueError(f"Missing gen_{i} from annotation panels: {missing}")
                cands = [candidate(annotations[f"gen_{i}"], req) for annotations in anns[:K]]
                n += 1
                best = max(cands, key=lambda c: (c[2], c[0], c[1]))  # oracle ranker
                useful += best[0] and best[1] >= 0.5
            lo, hi = wilson(useful, n)
            print(f"  best-of-{K}: useful {useful / n * 100:5.1f}%  [{lo:.1f}, {hi:.1f}]  (n={n})")


if __name__ == "__main__":
    main()
