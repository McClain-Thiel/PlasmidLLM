"""Shared pieces for the NeurIPS 2026 rebuttal experiments.

Everything in here exists so that every panel we compare — supervised base, post-GRPO,
retrieval, completion-only baselines, real plasmids — goes through one identical scoring
path. The reviewers' central objection is that reward, metric and ranker were entangled;
the least we can do in response is not introduce a second inconsistency of our own.

Detection path (fixed, used everywhere):
    pLannotate Feature -> feature_to_category_token() -> percmatch >= 95 and non-fragment

That mapping function is the repo's own (`scripts/build_motif_registry.py`), so "requested"
and "detected" are always expressed in the same vocabulary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_motif_registry import (  # noqa: E402
    MIN_PERCMATCH,
    feature_to_category_token,
)

__all__ = [
    "MIN_PERCMATCH",
    "TOKEN_RE",
    "CATEGORIES",
    "category_of",
    "detect",
    "is_viable",
    "load_annotations",
    "wilson",
    "mcnemar_exact",
    "paired_bootstrap_ci",
]

TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")
CATEGORIES = ["ORI", "AMR", "PROM", "REPORTER", "ELEM", "TAG"]

_ORIGIN_TYPES = {"rep_origin"}
_PROMOTER_TYPES = {"promoter"}
_CDS_TYPES = {"CDS"}
_MARKER_PREFIX = "<AMR_"


def category_of(token: str) -> str:
    return token.strip("<>").split("_")[0]


def detect(features: list[dict], quality_filter: bool = True) -> set[str]:
    """Hard tokens pLannotate found in one sequence.

    The quality filter mirrors the registry build. Without it a 28%-coverage partial CDS
    counts as a present component, which inflates both recall and unrequested-component
    counts; `quality_filter=False` exists only for robustness checks.
    """
    out: set[str] = set()
    for f in features:
        if quality_filter:
            if (f.get("percmatch") or 0) < MIN_PERCMATCH or f.get("fragment"):
                continue
        m = feature_to_category_token(str(f.get("Feature", "")))
        if m:
            out.add(m[0])
    return out


def is_viable(features: list[dict], quality_filter: bool = True) -> bool:
    """The paper's viability rule: origin AND selection marker AND (promoter OR CDS).

    Deliberately unchanged from the paper even though it is permissive — Dok5 W3 is right
    that this means "annotatable", not "would propagate". Changing the rule mid-rebuttal
    would make our numbers incomparable to the submitted ones; the stricter functional
    check belongs in a separate, clearly-labelled metric.
    """
    types = set()
    for f in features:
        if quality_filter:
            if (f.get("percmatch") or 0) < MIN_PERCMATCH or f.get("fragment"):
                continue
        types.add(f.get("Type"))
    toks = detect(features, quality_filter)
    return (
        bool(types & _ORIGIN_TYPES)
        and any(t.startswith(_MARKER_PREFIX) for t in toks)
        and bool(types & (_PROMOTER_TYPES | _CDS_TYPES))
    )


def load_annotations(path: str | Path) -> dict[str, list[dict]]:
    """seq_id -> features, from an annotate.py plannotate_results.json."""
    rows = json.loads(Path(path).read_text())
    annotations = {str(row["seq_id"]): row.get("features", []) for row in rows}
    if len(annotations) != len(rows):
        raise ValueError(f"Duplicate seq_id in {path}")
    return annotations


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, as percentages. Matches the paper's convention."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes -> (a_only, b_only, p)."""
    from math import comb

    if len(a) != len(b):
        raise ValueError(f"Paired outcomes differ in length: {len(a)} != {len(b)}")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return (0, 0, 1.0)
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1))
    return (a_only, b_only, min(1.0, 2 * tail / 2**n))


def paired_bootstrap_ci(
    a: list[bool], b: list[bool], iters: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """95% CI on (mean(b) - mean(a)) in percentage points, resampling prompts together.

    Prompts are the unit of resampling: outcomes for the two models on the same prompt are
    correlated, so resampling them independently would understate the interval.
    """
    import random

    if len(a) != len(b):
        raise ValueError(f"Paired outcomes differ in length: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("Cannot bootstrap an empty panel")
    if iters <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append((sum(b[i] for i in idx) - sum(a[i] for i in idx)) / n * 100)
    deltas.sort()
    return (deltas[int(iters * 0.025)], deltas[int(iters * 0.975)])
