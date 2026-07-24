#!/usr/bin/env python3
"""Precision / specificity of prompt following: how often do generations contain
components the prompt did NOT request?

Written for the NeurIPS 2026 rebuttal (meta-review point 1; pSXh W2/Q2).

The paper's faithfulness metric is recall-only — fraction of *requested* hard tokens
recovered — and the GRPO reward is likewise recall-only
(`post_training/scorers/plannotate.py`: composite = geo_mean(found quality) x recall
penalty, with no term for unrequested components). So a generation that satisfies the
prompt and also carries three extra origins and two extra resistance markers scores
identically to a clean one. This script measures what that misses.

Detected components come from the run's `plannotate_results.json`; the Feature -> token
mapping is the repo's own `feature_to_category_token`, so "detected" and "requested" are
expressed in the same vocabulary.

Usage:
  python eval/scripts/prompt_precision.py \\
      --annotations runs/grpo_plannotate_full_20260408/plannotate_results.json \\
      --generations runs/grpo_plannotate_full_20260408/generations.parquet \\
      --label post-GRPO
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402
from scripts.build_motif_registry import feature_to_category_token  # noqa: E402

TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")
CATEGORIES = ["ORI", "AMR", "PROM", "REPORTER", "ELEM", "TAG"]

# Mirrors the registry build (scripts/build_motif_registry.py): a hit only counts as a
# present component if it is a near-full-length, non-fragment match. Without this a
# 28%-coverage partial CDS would count as "an extra resistance marker".
MIN_PERCMATCH = 95.0


def category_of(token: str) -> str:
    return token.strip("<>").split("_")[0]


def detected_tokens(features: list[dict], quality_filter: bool) -> set[str]:
    out = set()
    for f in features:
        if quality_filter:
            if (f.get("percmatch") or 0) < MIN_PERCMATCH or f.get("fragment"):
                continue
        m = feature_to_category_token(str(f.get("Feature", "")))
        if m:
            out.add(m[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-quality-filter", action="store_true")
    args = ap.parse_args()

    ann = json.loads(args.annotations.read_text())
    by_id = {a["seq_id"]: a.get("features", []) for a in ann}

    prompts = [r[0] for r in duckdb.sql(
        f"select prompt from '{args.generations}'").fetchall()]

    qf = not args.no_quality_filter
    recalls, precisions, extras = [], [], []
    extra_by_cat: collections.Counter = collections.Counter()
    any_extra_by_cat: collections.Counter = collections.Counter()
    n = 0

    for i, prompt in enumerate(prompts):
        feats = by_id.get(f"gen_{i}")
        if feats is None:
            continue
        req = set(TOKEN_RE.findall(prompt))
        if not req:
            continue
        det = detected_tokens(feats, qf)
        n += 1

        hit = req & det
        extra = det - req
        recalls.append(len(hit) / len(req))
        precisions.append(len(hit) / len(det) if det else float("nan"))
        extras.append(len(extra))
        for t in extra:
            extra_by_cat[category_of(t)] += 1
        for c in {category_of(t) for t in extra}:
            any_extra_by_cat[c] += 1

    valid_prec = [p for p in precisions if p == p]
    print(f"\n=== {args.label} — n={n} prompts"
          f" (quality filter: {'percmatch>=95, no fragments' if qf else 'OFF'}) ===")
    print(f"  mean recall  (paper's faithfulness) : {statistics.mean(recalls)*100:5.1f}%")
    print(f"  mean precision (NOT in paper)       : {statistics.mean(valid_prec)*100:5.1f}%")
    print(f"  mean unrequested components / seq   : {statistics.mean(extras):5.2f}")
    print(f"  median unrequested components / seq : {statistics.median(extras):5.1f}")
    print(f"  generations with >=1 unrequested    : "
          f"{sum(1 for e in extras if e)/n*100:5.1f}%")

    print(f"\n  {'category':<12}{'total extra':>13}{'% seqs with >=1 extra':>24}")
    for c in CATEGORIES:
        print(f"  {c:<12}{extra_by_cat[c]:>13}{any_extra_by_cat[c]/n*100:>23.1f}%")


if __name__ == "__main__":
    main()
