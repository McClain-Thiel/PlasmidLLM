#!/usr/bin/env python3
"""Score generations on the plasmid-disjoint benchmark, decomposed as useful = viable x faithful.

Written for the NeurIPS 2026 rebuttal (M2.0 — the honest re-run behind meta-review point 2).

Every panel here — supervised base, post-GRPO, and the retrieval null — goes through one
identical path: pLannotate Feature calls -> `feature_to_category_token` -> `percmatch >= 95`
and non-fragment. No detector is crossed and no number is spliced in from the paper.

Report the decomposition, not the single useful rate. Retrieval inherits real-plasmid
viability (~95%) for free, so if generated sequences are viable only ~60% of the time the
model is capped near ~60% useful *however good faithfulness gets* — meaning the gap to
retrieval is a viability gap, not a faithfulness gap. One number hides that; three show it.

Viability rule is the paper's: origin AND selection marker AND (promoter OR CDS) detected.

Usage:
  python eval/scripts/score_disjoint.py \\
      --benchmark disjoint_benchmark.parquet \\
      --panel base=gen_base/plannotate_results.json \\
      --panel grpo=gen_grpo/plannotate_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402
from scripts.build_motif_registry import (  # noqa: E402
    MIN_PERCMATCH,
    feature_to_category_token,
)

TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")

ORIGIN_TYPES = {"rep_origin"}
PROMOTER_TYPES = {"promoter"}
CDS_TYPES = {"CDS"}
# Selection markers are identified by token category rather than pLannotate Type, since
# a resistance gene is annotated as a plain CDS.
MARKER_PREFIX = "<AMR_"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def analyse(features: list[dict]) -> tuple[set[str], bool]:
    """Return (detected hard tokens, viable) under the paper's viability rule."""
    toks, types = set(), set()
    for f in features:
        if (f.get("percmatch") or 0) < MIN_PERCMATCH or f.get("fragment"):
            continue
        types.add(f.get("Type"))
        m = feature_to_category_token(str(f.get("Feature", "")))
        if m:
            toks.add(m[0])
    has_ori = bool(types & ORIGIN_TYPES)
    has_marker = any(t.startswith(MARKER_PREFIX) for t in toks)
    has_prom_or_cds = bool(types & (PROMOTER_TYPES | CDS_TYPES))
    return toks, (has_ori and has_marker and has_prom_or_cds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True,
                    metavar="LABEL=plannotate_results.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    prompts = [r[0] for r in duckdb.sql(
        f"select prompt from '{args.benchmark}'").fetchall()]
    requested = [set(TOKEN_RE.findall(p)) for p in prompts]

    print(f"benchmark prompts: {len(prompts)}   "
          f"mean requested hard tokens: "
          f"{statistics.mean(len(r) for r in requested):.2f}\n")
    hdr = (f"{'panel':<16}{'n':>6}{'viable':>10}{'mean F':>9}"
           f"{'faithful':>10}{'USEFUL':>9}{'  Wilson 95% (useful)':>22}")
    print(hdr)
    print("-" * len(hdr))

    for spec in args.panel:
        label, path = spec.split("=", 1)
        ann = json.loads(Path(path).read_text())
        by_id = {str(a["seq_id"]): a.get("features", []) for a in ann}

        n = viable = faithful = useful = 0
        fs = []
        for i, req in enumerate(requested):
            feats = by_id.get(f"gen_{i}")
            if feats is None or not req:
                continue
            n += 1
            det, viab = analyse(feats)
            f = len(req & det) / len(req)
            fs.append(f)
            viable += viab
            faithful += f >= args.threshold
            useful += viab and f >= args.threshold

        lo, hi = wilson(useful, n)
        print(f"{label:<16}{n:>6}{viable/n*100:>9.1f}%{statistics.mean(fs)*100:>8.1f}%"
              f"{faithful/n*100:>9.1f}%{useful/n*100:>8.1f}%"
              f"{f'  [{lo:.1f}, {hi:.1f}]':>22}")

    print(f"\nviability rule: origin AND selection marker AND (promoter OR CDS), "
          f"percmatch>={MIN_PERCMATCH:.0f}, non-fragment")
    print(f"faithful: fraction of requested hard tokens detected >= T={args.threshold}")


if __name__ == "__main__":
    main()
