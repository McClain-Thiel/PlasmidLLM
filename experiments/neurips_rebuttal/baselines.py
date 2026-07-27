#!/usr/bin/env python3
"""The two nulls the paper is missing, and they fail in opposite directions.

  retrieval        Return the training plasmid whose component set best matches the
                   request (Dok5 W2, pSXh W4). Satisfies the spec, invents nothing —
                   it hands you a plasmid that already exists.

  completion-only  Every other plasmid LM (PlasmidGPT/Shao, PlasmidGPT-SFT/Cunningham,
                   PlasmidGPT-GRPO) generates novel sequence but has no interface for
                   saying what to make. Scoring unconditional samples against the same
                   specs measures the base rate an unsteerable generator achieves — i.e.
                   exactly what promptability is worth.

Between them they bracket PlasmidLM: retrieval is specifiable but not novel,
completion-only is novel but not specifiable.

`retrieval` uses `--annotations` when given, scoring retrieved plasmids through the same
pLannotate path the model faces. Without it, it falls back to each plasmid's recorded
annotation, which flatters the baseline — the two agreed closely in practice (95.7% vs
96.2% mean faithfulness), but prefer the annotated form for anything reported.

Usage:
  python baselines.py retrieval --train-pairs training_pairs_v4.parquet \\
      --benchmark disjoint_benchmark.parquet [--annotations retr/plannotate_results.json]
  python baselines.py completion --benchmark disjoint_benchmark.parquet \\
      --panel shao=pg_base/plannotate_results.json --seed 0
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
from pathlib import Path

import duckdb

from common import TOKEN_RE, detect, is_viable, load_annotations, wilson

# Real deposited plasmids; the paper measures 95.0% viability on a real-plasmid panel
# under the same rule. Applying it is generous to the baseline, which is the point.
REAL_PLASMID_VIABILITY = 0.950


def build_index(train_pairs: Path) -> dict[int, set[str]]:
    """plasmid_id -> union of hard tokens over its prompts (its annotated component set)."""
    index: dict[int, set[str]] = collections.defaultdict(set)
    for pid, prompt in duckdb.sql(f"select plasmid_id, prompt from '{train_pairs}'").fetchall():
        index[pid] |= set(TOKEN_RE.findall(prompt))
    return {k: v for k, v in index.items() if v}


def cmd_retrieval(args) -> None:
    index = build_index(args.train_pairs)
    postings: dict[str, set[int]] = collections.defaultdict(set)
    for pid, toks in index.items():
        for t in toks:
            postings[t].add(pid)
    print(f"indexed training plasmids: {len(index):,}")

    prompts = [r[0] for r in duckdb.sql(f"select prompt from '{args.benchmark}'").fetchall()]
    ann = load_annotations(args.annotations) if args.annotations else None

    faiths, full, picks, direct_viability = [], 0, [], []
    for q in prompts:
        req = set(TOKEN_RE.findall(q))
        if not req:
            raise ValueError("Every benchmark prompt must contain at least one hard token")
        candidates: set[int] = set()
        for t in req:
            candidates |= postings.get(t, set())
        if not candidates:
            raise ValueError(f"No retrieval candidate for requested tokens: {sorted(req)}")
        best_pid = max(
            candidates,
            key=lambda pid: (
                len(req & index[pid]) / len(req | index[pid]),
                -pid,
            ),
        )
        picks.append(best_pid)
        if ann is not None:
            if str(best_pid) not in ann:
                raise ValueError(f"Missing direct annotation for retrieved plasmid {best_pid}")
            features = ann[str(best_pid)]
            got = detect(features)
            direct_viability.append(is_viable(features))
        else:
            got = index[best_pid]
        f = len(req & got) / len(req)
        faiths.append(f)
        full += req <= got

    n = len(faiths)
    k = sum(1 for f in faiths if f >= 0.5)
    lo, hi = wilson(k, n)
    src = "same-pipeline annotations" if ann else "recorded annotations (optimistic)"
    print(f"\n=== retrieval baseline — scored via {src} ===")
    print(f"  prompts                  : {n}")
    print(f"  mean faithfulness        : {statistics.mean(faiths) * 100:.1f}%")
    print(f"  F >= 0.5                 : {k / n * 100:.1f}%  [{lo:.1f}, {hi:.1f}]")
    print(f"  F = 1.0 (full spec)      : {full / n * 100:.1f}%")
    if ann is not None:
        useful = sum(viable and faith >= 0.5 for viable, faith in zip(direct_viability, faiths))
        print(f"  viability                : {statistics.mean(direct_viability) * 100:.1f}%")
        print(f"  useful                   : {useful / n * 100:.1f}%")
    else:
        print(
            f"  implied useful (x{REAL_PLASMID_VIABILITY:.2f})  : "
            f"{k / n * REAL_PLASMID_VIABILITY * 100:.1f}%"
        )
    if args.out:
        Path(args.out).write_text("\n".join(str(p) for p in picks))
        print(f"  retrieved ids -> {args.out}")


def cmd_completion(args) -> None:
    """Score unconditional generations against specs they were never given.

    Pairing is random, not oracle: prompt i is judged against a randomly drawn sample.
    That is the honest estimate of what you get when the model cannot be steered. An
    oracle pairing (best sample per prompt) would measure something else — the best of K
    unconditional draws — and is not what a user could do without our annotation pipeline.
    """
    prompts = [r[0] for r in duckdb.sql(f"select prompt from '{args.benchmark}'").fetchall()]
    requested = [set(TOKEN_RE.findall(p)) for p in prompts]
    rng = random.Random(args.seed)

    hdr = (
        f"{'panel':<20}{'n gen':>7}{'viable':>9}{'mean F':>9}"
        f"{'F>=0.5':>9}{'F=1.0':>8}{'USEFUL':>9}{'  Wilson 95%':>18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for spec in args.panel:
        label, path = spec.split("=", 1)
        ann = load_annotations(path)
        ids = sorted(ann)
        viable_frac = sum(1 for i in ids if is_viable(ann[i])) / len(ids)

        faiths, useful = [], 0
        for req in requested:
            if not req:
                continue
            sid = ids[rng.randrange(len(ids))]
            feats = ann[sid]
            f = len(req & detect(feats)) / len(req)
            faiths.append(f)
            useful += is_viable(feats) and f >= 0.5
        n = len(faiths)
        k = sum(1 for f in faiths if f >= 0.5)
        full = sum(1 for f in faiths if f >= 1.0)
        lo, hi = wilson(useful, n)
        print(
            f"{label:<20}{len(ids):>7}{viable_frac * 100:>8.1f}%"
            f"{statistics.mean(faiths) * 100:>8.1f}%{k / n * 100:>8.1f}%"
            f"{full / n * 100:>7.1f}%{useful / n * 100:>8.1f}%"
            f"{f'  [{lo:.1f}, {hi:.1f}]':>18}"
        )
    print("\nUnconditional samples scored against specs they never received:")
    print("the base rate a non-steerable generator hits by chance.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval")
    r.add_argument("--train-pairs", type=Path, required=True)
    r.add_argument("--benchmark", type=Path, required=True)
    r.add_argument("--annotations", type=Path)
    r.add_argument("--out", type=Path, help="write retrieved plasmid ids here")
    r.set_defaults(func=cmd_retrieval)

    c = sub.add_parser("completion")
    c.add_argument("--benchmark", type=Path, required=True)
    c.add_argument(
        "--panel", action="append", required=True, metavar="LABEL=plannotate_results.json"
    )
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_completion)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
