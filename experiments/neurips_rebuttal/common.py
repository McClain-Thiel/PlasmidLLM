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

import pandas as pd

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
    "paired_cluster_bootstrap_ci",
    "summarize_paired_training_seeds",
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
    failed = [str(row["seq_id"]) for row in rows if "error" in row]
    if failed:
        raise ValueError(f"Annotation failures cannot be scored as absent features: {failed[:10]}")
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


def paired_cluster_bootstrap_ci(
    baseline: list[list[float]], treatment: list[list[float]], cluster_ids: list[str],
    iters: int = 10000, seed: int = 42,
) -> tuple[float, float]:
    """95% CI in original metric units, resampling training seeds and source clusters.

    Inputs are aligned [training_seed, prompt] panels, averaged over the same fixed
    decoding seeds first. A single training seed estimates only prompt uncertainty.
    Preserve the pairing and all prompts within a sampled cluster. The estimand is
    prompt-weighted, not an unweighted average over unequal-sized clusters.
    """
    import numpy as np

    a, b = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1 or a.shape[1] != len(cluster_ids):
        raise ValueError("Expected matched nonempty [training_seed, prompt] arrays and cluster IDs")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or iters < 1:
        raise ValueError("Metrics must be finite and bootstrap iterations positive")
    if any(not isinstance(c, str) or not c for c in cluster_ids):
        raise ValueError("Every prompt must have a nonempty cluster ID")
    clusters = sorted(set(cluster_ids))
    if len(clusters) < 2:
        raise ValueError("At least two source clusters are required")
    masks = [np.array([c == cluster for c in cluster_ids]) for cluster in clusters]
    sums = np.stack([(b - a)[:, mask].sum(axis=1) for mask in masks], axis=1)
    counts = np.array([mask.sum() for mask in masks])
    rng = np.random.default_rng(seed)
    samples = np.empty(iters)
    for i in range(iters):
        seed_indices = rng.integers(a.shape[0], size=a.shape[0])
        cluster_indices = rng.integers(len(clusters), size=len(clusters))
        samples[i] = sums[np.ix_(seed_indices, cluster_indices)].sum() / (
            len(seed_indices) * counts[cluster_indices].sum())
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_paired_training_seeds(
    baseline: pd.DataFrame, treatment: pd.DataFrame, panel: pd.DataFrame,
    metric: str, decode_seeds: list[int],
    fixed_baseline: bool = False, iters: int = 10000,
) -> dict:
    """Validate the full paired grid before averaging decodes and bootstrapping.

    Frames contain sample_id, cluster_id, prompt, training_seed, seed (decoding),
    and the metric. The supplied panel is the preselected endpoint population.
    A fixed baseline has one pretrained model and does not gain seed uncertainty
    merely because it is compared against several RL runs.
    """
    import numpy as np

    identity = ["sample_id", "cluster_id", "prompt"]
    if panel.empty or panel.sample_id.duplicated().any() or panel[identity].isna().any().any():
        raise ValueError("Expected a nonempty panel with unique IDs and complete metadata")
    if not decode_seeds or len(decode_seeds) != len(set(decode_seeds)):
        raise ValueError("Expected unique, explicitly specified decoding seeds")
    reference = panel[identity].set_index("sample_id").sort_index()
    matrices = []
    training_seeds = []
    for label, frame in (("baseline", baseline), ("treatment", treatment)):
        required = set(identity + ["training_seed", "seed", metric])
        if not required <= set(frame.columns):
            raise ValueError(f"Missing {label} metric/identity columns")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing {label} metric/identity values")
        if frame.duplicated(["sample_id", "training_seed", "seed"]).any():
            raise ValueError(f"Duplicate {label} paired observations")
        if set(frame.seed) != set(decode_seeds):
            raise ValueError(f"Incomplete {label} decoding seed set")
        seeds = sorted(frame.training_seed.unique().tolist())
        if not seeds:
            raise ValueError(f"Empty {label} observations")
        training_seeds.append(seeds)
        values = []
        for training_seed in seeds:
            decodes = []
            for decode_seed in decode_seeds:
                rows = frame[(frame.training_seed == training_seed) & (frame.seed == decode_seed)]
                rows = rows.set_index("sample_id").sort_index()
                if not rows[["cluster_id", "prompt"]].equals(reference):
                    raise ValueError(f"{label} observations do not match the frozen panel")
                decodes.append(rows[metric].to_numpy(dtype=float))
            values.append(np.mean(decodes, axis=0))
        matrices.append(np.asarray(values))
    if fixed_baseline:
        if len(training_seeds[0]) != 1:
            raise ValueError("A fixed baseline must contain exactly one training seed")
        matrices[0] = np.repeat(matrices[0], len(training_seeds[1]), axis=0)
    elif training_seeds[0] != training_seeds[1]:
        raise ValueError("Baseline and treatment training seeds differ")
    low, high = paired_cluster_bootstrap_ci(
        matrices[0], matrices[1], reference.cluster_id.tolist(), iters=iters,
    )
    differences = (matrices[1] - matrices[0]).mean(axis=1)
    return dict(
        metric=metric, n_prompts=len(reference), n_clusters=reference.cluster_id.nunique(),
        baseline_training_seeds=training_seeds[0], treatment_training_seeds=training_seeds[1],
        decoding_seeds=decode_seeds, baseline_fixed=fixed_baseline,
        baseline_mean=float(matrices[0].mean()), treatment_mean=float(matrices[1].mean()),
        delta=float(differences.mean()), paired_cluster_training_seed_ci95=[low, high],
        per_training_seed_delta={str(seed): float(value)
                                 for seed, value in zip(training_seeds[1], differences)},
        aggregation="Average fixed decoding seeds, then resample paired training seeds and source clusters",
    )
