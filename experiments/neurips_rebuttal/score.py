#!/usr/bin/env python3
"""Score annotated runs against the prompts that produced them.

Two views of the same annotations, because the paper only ever reported the first:

  useful     viable x faithful, the paper's primary metric. Reported decomposed —
             retrieval inherits real-plasmid viability for free, so a single useful
             number hides whether a gap is a viability gap or a faithfulness gap.
             (It was both: viability 0.65x, faithfulness 0.41x.)

  precision  what fraction of detected components were actually asked for, and how
             many unrequested ones came along. The GRPO reward is recall-only
             (post_training/scorers/plannotate.py) so nothing penalises extra
             origins or resistance markers. pSXh W2/Q2 and meta-review point 1.

Both faithfulness thresholds are printed: T=0.5 is the paper's, T=1.0 is "every
requested component present", which is the bar a designer actually cares about and the
one where retrieval's advantage is capped.

Usage:
  python score.py --benchmark disjoint_benchmark.parquet \\
      --panel base=gen_base/plannotate_results.json \\
      --panel grpo=gen_grpo/plannotate_results.json [--precision] [--paired base,grpo]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import duckdb

from common import (
    CATEGORIES,
    TOKEN_RE,
    category_of,
    detect,
    is_viable,
    load_annotations,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilson,
)


def load_requested(benchmark: Path) -> list[set[str]]:
    prompts = [r[0] for r in duckdb.sql(f"select prompt from '{benchmark}'").fetchall()]
    return [set(TOKEN_RE.findall(p)) for p in prompts]


def score_panel(ann: dict, requested: list[set[str]], quality_filter: bool = True):
    """Per-prompt (viable, faithfulness) aligned to the benchmark order."""
    missing = [f"gen_{i}" for i in range(len(requested)) if f"gen_{i}" not in ann]
    if missing:
        raise ValueError(
            f"Annotation panel is missing {len(missing)} sequences; first={missing[:5]}"
        )
    empty_requests = [i for i, req in enumerate(requested) if not req]
    if empty_requests:
        raise ValueError(
            f"Benchmark contains {len(empty_requests)} prompts without hard tokens; "
            f"first={empty_requests[:5]}"
        )

    out = []
    for i, req in enumerate(requested):
        feats = ann[f"gen_{i}"]
        det = detect(feats, quality_filter)
        out.append((is_viable(feats, quality_filter), len(req & det) / len(req), req, det))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument(
        "--panel", action="append", required=True, metavar="LABEL=plannotate_results.json"
    )
    ap.add_argument(
        "--precision", action="store_true", help="also report unrequested-component counts"
    )
    ap.add_argument(
        "--paired", metavar="A,B", help="McNemar + bootstrap CI on useful rate between two panels"
    )
    ap.add_argument("--no-quality-filter", action="store_true")
    args = ap.parse_args()

    qf = not args.no_quality_filter
    requested = load_requested(args.benchmark)
    print(
        f"benchmark prompts: {len(requested)}   "
        f"mean requested hard tokens: "
        f"{statistics.mean(len(r) for r in requested):.2f}   "
        f"quality filter: {'on' if qf else 'OFF'}\n"
    )

    panels: dict[str, list] = {}
    hdr = (
        f"{'panel':<14}{'n':>6}{'viable':>9}{'mean F':>9}"
        f"{'F>=0.5':>9}{'F=1.0':>8}{'USEFUL':>9}{'  Wilson 95%':>18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for spec in args.panel:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Invalid --panel value: {spec!r}")
        if label in panels:
            raise ValueError(f"Duplicate panel label: {label}")
        rows = score_panel(load_annotations(path), requested, qf)
        panels[label] = rows
        n = len(rows)
        viable = sum(1 for v, *_ in rows if v)
        f05 = sum(1 for _, f, *_ in rows if f >= 0.5)
        f10 = sum(1 for _, f, *_ in rows if f >= 1.0)
        useful = sum(1 for v, f, *_ in rows if v and f >= 0.5)
        lo, hi = wilson(useful, n)
        print(
            f"{label:<14}{n:>6}{viable / n * 100:>8.1f}%"
            f"{statistics.mean(f for _, f, *_ in rows) * 100:>8.1f}%"
            f"{f05 / n * 100:>8.1f}%{f10 / n * 100:>7.1f}%{useful / n * 100:>8.1f}%"
            f"{f'  [{lo:.1f}, {hi:.1f}]':>18}"
        )

    if args.paired:
        a, b = args.paired.split(",")
        missing_labels = [label for label in (a, b) if label not in panels]
        if missing_labels:
            raise ValueError(f"Unknown paired panel labels: {missing_labels}")
        ua = [v and f >= 0.5 for v, f, *_ in panels[a]]
        ub = [v and f >= 0.5 for v, f, *_ in panels[b]]
        a_only, b_only, p = mcnemar_exact(ua, ub)
        lo, hi = paired_bootstrap_ci(ua, ub)
        print(
            f"\npaired {b} vs {a}: delta = {(sum(ub) - sum(ua)) / len(ua) * 100:+.1f} pp"
            f"   bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]"
        )
        print(f"  McNemar discordant {a}-only={a_only} {b}-only={b_only}, exact p={p:.3g}")

    if args.precision:
        print(
            f"\n{'panel':<14}{'precision':>11}{'extra/seq':>11}{'>=1 extra':>11}"
            f"   unrequested per 100 seqs, by category"
        )
        print("-" * 88)
        for label, rows in panels.items():
            precs = [len(req & det) / len(det) for _, _, req, det in rows if det]
            if not precs:
                raise ValueError(f"Panel {label!r} has no detected hard tokens")
            extras = [len(det - req) for _, _, req, det in rows]
            bycat = {c: 0 for c in CATEGORIES}
            for _, _, req, det in rows:
                for t in det - req:
                    if category_of(t) in bycat:
                        bycat[category_of(t)] += 1
            n = len(rows)
            cats = " ".join(f"{c}:{v / n * 100:.0f}" for c, v in bycat.items())
            print(
                f"{label:<14}{statistics.mean(precs) * 100:>10.1f}%"
                f"{statistics.mean(extras):>11.2f}"
                f"{sum(1 for e in extras if e) / n * 100:>10.1f}%   {cats}"
            )

    print("\nviability: origin AND selection marker AND (promoter OR CDS)")
    print("faithfulness F: fraction of requested hard tokens detected")


if __name__ == "__main__":
    main()
