#!/usr/bin/env python3
"""Prompt-conditional lift: the metric that actually measures prompt control.

The useful-plasmid rate cannot separate control from base rate — it is matched by retrieval
(F13) and by unconditional generation (F17), because common plasmid components appear
regardless of the prompt. Lift fixes that. For each component t:

    lift(t) = P(t detected | t requested) − P(t detected | t NOT requested)

A model with no control (retrieval, an unconditional generator) has lift ≈ 0 for every
component by construction — its output does not depend on the prompt. A model that follows
the prompt has positive lift wherever it has learned that component. On the disjoint
benchmark PlasmidLM-GRPO shows mean lift +7.8 pp with 18/41 components above +10 pp
(AMR_ZEOCIN +34, PROM_CMV +21, ORI_F1 +20), and ≈0 for the rare tail.

This is the controllability metric to report in place of the saturated useful rate: it is
not matched by the trivial baselines, it quantifies control per component, and it maps
exactly where prompting works and fails.

Usage:
  python control_lift.py --benchmark disjoint_benchmark.parquet \\
      --panel grpo=gen_grpo/plannotate_results.json \\
      --panel base=gen_base/plannotate_results.json --min-count 20
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import duckdb

from common import TOKEN_RE, detect, load_annotations


def lifts(ann: dict, requested: list[set[str]], min_count: int):
    det = [detect(ann.get(f"gen_{i}", [])) for i in range(len(requested))]
    vocab = set().union(*requested) if requested else set()
    rows = []
    for t in vocab:
        req = [i for i, r in enumerate(requested) if t in r]
        non = [i for i, r in enumerate(requested) if t not in r]
        if len(req) < min_count or len(non) < min_count:
            continue
        p_req = sum(t in det[i] for i in req) / len(req)
        p_non = sum(t in det[i] for i in non) / len(non)
        rows.append((t, len(req), p_req, p_non, p_req - p_non))
    rows.sort(key=lambda x: -x[4])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True,
                    metavar="LABEL=plannotate_results.json")
    ap.add_argument("--min-count", type=int, default=20,
                    help="min prompts requesting / not-requesting a component to include it")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    requested = [set(TOKEN_RE.findall(r[0]))
                 for r in duckdb.sql(
                     f"select prompt from '{args.benchmark}'").fetchall()]

    for spec in args.panel:
        label, path = spec.split("=", 1)
        rows = lifts(load_annotations(path), requested, args.min_count)
        if not rows:
            print(f"\n{label}: no components with >= {args.min_count} in both groups")
            continue
        print(f"\n=== {label}: lift = P(present|requested) - P(present|not requested) ===")
        print(f"{'component':<24}{'n_req':>6}{'P|req':>8}{'P|not':>8}{'lift':>8}")
        for t, n, pr, pn, l in rows[:args.top]:
            print(f"{t:<24}{n:>6}{pr*100:>7.0f}%{pn*100:>7.0f}%{l*100:>+7.0f}")
        if len(rows) > args.top:
            print("  ...")
            for t, n, pr, pn, l in rows[-3:]:
                print(f"{t:<24}{n:>6}{pr*100:>7.0f}%{pn*100:>7.0f}%{l*100:>+7.0f}")
        mean = statistics.mean(x[4] for x in rows)
        print(f"  mean lift over {len(rows)} components: {mean*100:+.1f} pp   "
              f"| lift>+10pp: {sum(1 for x in rows if x[4] > 0.1)}/{len(rows)}   "
              f"| lift<=0: {sum(1 for x in rows if x[4] <= 0)}/{len(rows)}")


if __name__ == "__main__":
    main()
