#!/usr/bin/env python3
"""Nearest-neighbour retrieval baseline: how well does "just return the closest training
plasmid" do on the same prompts?

Written for the NeurIPS 2026 rebuttal (Dok5 W2; pSXh W4; meta-review point 3).

This is the null the paper is missing. Dok5: "For a 19.3M model trained on ~20K unique
plasmids, the natural null is nearest-neighbour retrieval by prompt token-set Jaccard from
the training set, returning the retrieved plasmid." If retrieval matches the model on the
useful-plasmid rate, the generative route adds nothing measurable and the contribution is
in trouble regardless of how the model scores.

Method. Index every training plasmid by the union of hard tokens across its prompts in
`training_pairs_v4.parquet` — those prompts are annotation subsets, so the union is that
plasmid's annotated component set. For each benchmark prompt, retrieve the training plasmid
maximising Jaccard against the requested set, then score the *retrieved plasmid* against the
*requested* prompt exactly as a generation would be scored:

    faithfulness F = |requested ∩ retrieved| / |requested|

Viability is not recomputed per retrieved plasmid: these are real deposited Addgene
plasmids, and the paper measures 95.0% viability on a real-plasmid panel under the same
rule. Applying that constant is generous to the baseline — deliberately, since a null should
be steelmanned rather than handicapped.

The retrieval scores reported are therefore an upper bound on what the baseline delivers
biologically: a retrieved plasmid is a *different* construct that happens to share
components, not a construct built to the specification.

Usage:
  python eval/scripts/retrieval_baseline.py \\
      --train-pairs training_pairs_v4.parquet \\
      --benchmark disjoint_benchmark.parquet
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
from pathlib import Path

import duckdb

TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")
REAL_PLASMID_VIABILITY = 0.950  # paper, real-Addgene panel, same viability rule


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="faithfulness threshold T for the useful-plasmid rate")
    args = ap.parse_args()

    rows = duckdb.sql(
        f"select plasmid_id, prompt from '{args.train_pairs}'").fetchall()
    index: dict[int, set[str]] = collections.defaultdict(set)
    for pid, prompt in rows:
        index[pid] |= set(TOKEN_RE.findall(prompt))
    index = {k: v for k, v in index.items() if v}
    print(f"indexed training plasmids: {len(index):,}")

    # Invert to a token -> plasmid postings list so each query only scores candidates
    # sharing at least one token, instead of all 20k plasmids.
    postings: dict[str, set[int]] = collections.defaultdict(set)
    for pid, toks in index.items():
        for t in toks:
            postings[t].add(pid)

    queries = [r[0] for r in duckdb.sql(
        f"select prompt from '{args.benchmark}'").fetchall()]

    faiths, jaccards, exact = [], [], 0
    for q in queries:
        req = set(TOKEN_RE.findall(q))
        if not req:
            continue
        cands: set[int] = set()
        for t in req:
            cands |= postings.get(t, set())
        best_j, best_tok = 0.0, set()
        for pid in cands:
            toks = index[pid]
            j = len(req & toks) / len(req | toks)
            if j > best_j:
                best_j, best_tok = j, toks
        faiths.append(len(req & best_tok) / len(req))
        jaccards.append(best_j)
        exact += best_j == 1.0

    n = len(faiths)
    passed = sum(1 for f in faiths if f >= args.threshold)
    useful = passed * REAL_PLASMID_VIABILITY
    lo, hi = wilson(passed, n)

    print(f"\n=== retrieval baseline (max prompt-token Jaccard over training set) ===")
    print(f"  benchmark prompts                 : {n}")
    print(f"  mean best Jaccard                 : {statistics.mean(jaccards):.3f}")
    print(f"  exact token-set match available   : {exact}/{n} ({exact/n*100:.1f}%)")
    print(f"  mean faithfulness of retrieved     : {statistics.mean(faiths)*100:.1f}%")
    print(f"  faithful at T={args.threshold}                  : "
          f"{passed}/{n} = {passed/n*100:.1f}%  (Wilson 95% [{lo*100:.1f}, {hi*100:.1f}])")
    print(f"  implied USEFUL rate (x{REAL_PLASMID_VIABILITY:.3f} viability): "
          f"{useful/n*100:.1f}%")
    print("\n  Note: assumes retrieved real plasmids inherit the paper's 95.0% real-panel")
    print("  viability. Generous to the baseline by construction.")


if __name__ == "__main__":
    main()
