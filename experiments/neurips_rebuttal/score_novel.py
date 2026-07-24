#!/usr/bin/env python3
"""Score the novel-combination benchmark. T=1.0 is the point.

On these prompts no training plasmid supplies the full requested set, so retrieval is capped
at F=1.0 -> ~0% by construction. F16: the model is also ~0% at full satisfaction, and when
asked for the substituted-in component produces it only ~5% of the time while reverting to
the removed component ~40% of the time. Reports the substituted-token diagnostic too.

Usage:
  python score_novel.py --benchmark novel_benchmark.parquet \\
      --train-pairs training_pairs_v4.parquet \\
      --panel base=nov_base.json --panel grpo=nov_grpo.json
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import duckdb

from common import (
    TOKEN_RE,
    detect,
    is_viable,
    load_annotations,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilson,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True,
                    metavar="LABEL=plannotate_results.json")
    args = ap.parse_args()

    bm = duckdb.sql(
        f"select prompt, substituted_in, substituted_out from '{args.benchmark}'").df()
    requested = [set(TOKEN_RE.findall(p)) for p in bm.prompt]

    # retrieval ceiling: best faithfulness available from any single training plasmid
    index: dict[int, set[str]] = collections.defaultdict(set)
    for pid, p in duckdb.sql(
        f"select plasmid_id, prompt from '{args.train_pairs}'").fetchall():
        index[pid] |= set(TOKEN_RE.findall(p))
    index = {k: v for k, v in index.items() if v}
    postings: dict[str, set[int]] = collections.defaultdict(set)
    for pid, toks in index.items():
        for t in toks:
            postings[t].add(pid)

    def retr_faith(req: set[str]) -> float:
        cands: set[int] = set()
        for t in req:
            cands |= postings.get(t, set())
        return max((len(req & index[p]) / len(req) for p in cands), default=0.0)

    rf = [retr_faith(req) for req in requested]

    print(f"novel-combination benchmark, n={len(requested)}\n")
    print(f"{'panel':<20}{'viable':>9}{'F>=0.5':>9}{'F=1.0':>9}{'  Wilson (F=1.0)':>18}")
    print("-" * 65)
    print(f"{'retrieval (ceiling)':<20}{'--':>9}"
          f"{sum(1 for f in rf if f >= 0.5)/len(rf)*100:>8.1f}%"
          f"{sum(1 for f in rf if f >= 1.0)/len(rf)*100:>8.1f}%   (capped by construction)")

    panels = {}
    for spec in args.panel:
        label, path = spec.split("=", 1)
        ann = load_annotations(path)
        rows = []
        for i, req in enumerate(requested):
            f = ann.get(f"gen_{i}")
            if f is None or not req:
                continue
            det = detect(f)
            rows.append((is_viable(f), len(req & det) / len(req), req <= det, det))
        panels[label] = (rows, ann)
        n = len(rows)
        lo1, hi1 = wilson(sum(1 for *_, full, _ in rows if full), n)
        print(f"{label:<20}{sum(v for v, *_ in rows)/n*100:>8.1f}%"
              f"{sum(1 for _, fa, *_ in rows if fa >= 0.5)/n*100:>8.1f}%"
              f"{sum(1 for _, _, full, _ in rows if full)/n*100:>8.1f}%"
              f"{f'  [{lo1:.1f}, {hi1:.1f}]':>18}")

    # substituted-token diagnostic
    print(f"\n{'panel':<20}{'sub-IN recovered':>18}{'reverted to sub-OUT':>22}")
    for label, (rows, ann) in panels.items():
        got_in = got_out = n = 0
        for i, (new, old) in enumerate(zip(bm.substituted_in, bm.substituted_out)):
            f = ann.get(f"gen_{i}")
            if f is None:
                continue
            n += 1
            det = detect(f)
            got_in += new in det
            got_out += old in det
        print(f"{label:<20}{got_in/n*100:>17.1f}%{got_out/n*100:>21.1f}%")

    labels = [s.split("=", 1)[0] for s in args.panel]
    if len(labels) == 2:
        a, b = labels
        ua = [v and fa >= 0.5 for v, fa, *_ in panels[a][0]]
        ub = [v and fa >= 0.5 for v, fa, *_ in panels[b][0]]
        _, _, p = mcnemar_exact(ua, ub)
        lo, hi = paired_bootstrap_ci(ua, ub)
        print(f"\nuseful@T=0.5  {a} {sum(ua)/len(ua)*100:.1f}% -> {b} {sum(ub)/len(ub)*100:.1f}%"
              f"  delta {(sum(ub)-sum(ua))/len(ua)*100:+.1f}pp  CI[{lo:+.1f},{hi:+.1f}]  p={p:.3g}")


if __name__ == "__main__":
    main()
