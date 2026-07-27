#!/usr/bin/env python3
"""Compare matched decoding seeds on standard generation guardrails."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_motif_registry import feature_to_category_token  # noqa: E402


TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG)_[A-Z0-9_]+>")
MIN_PERCMATCH = 95.0
ORIGIN_TYPES = {"rep_origin"}
PROMOTER_OR_CDS_TYPES = {"promoter", "CDS"}
EXTRA_CATEGORIES = ("ORI", "AMR", "PROM", "REPORTER", "ELEM", "TAG")


def parse_seed_path(value: str) -> tuple[int, Path]:
    raw_seed, separator, raw_path = value.partition("=")
    if not separator or not raw_seed or not raw_path:
        raise ValueError(f"Invalid seed path {value!r}; expected SEED=PATH")
    return int(raw_seed), Path(raw_path)


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    parsed = [parse_seed_path(value) for value in values]
    paths = dict(parsed)
    if len(paths) != len(parsed):
        raise ValueError("Each decoding seed may be supplied only once")
    return paths


def load_annotations(path: Path) -> dict[str, list[dict]]:
    rows = json.loads(path.read_text())
    annotations = {str(row["seq_id"]): row.get("features", []) for row in rows}
    if len(annotations) != len(rows):
        raise ValueError(f"duplicate seq_id in {path}")
    return annotations


def filtered_features(features: list[dict]) -> list[dict]:
    return [
        feature
        for feature in features
        if (feature.get("percmatch") or 0) >= MIN_PERCMATCH and not feature.get("fragment")
    ]


def detect_tokens(features: list[dict]) -> set[str]:
    tokens = set()
    for feature in filtered_features(features):
        mapped = feature_to_category_token(str(feature.get("Feature", "")))
        if mapped:
            tokens.add(mapped[0])
    return tokens


def score_annotations(path: Path, requested: list[set[str]]) -> dict[str, np.ndarray]:
    annotations = load_annotations(path)
    expected_ids = {f"gen_{i}" for i in range(len(requested))}
    missing = sorted(expected_ids - annotations.keys())
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} expected sequences; first={missing[:5]}"
        )

    faithfulness = []
    precision = []
    extras_count = []
    any_extra = []
    viability = []
    useful = []
    extra_categories = {category: [] for category in EXTRA_CATEGORIES}
    for i, wanted in enumerate(requested):
        features = annotations[f"gen_{i}"]
        filtered = filtered_features(features)
        found = detect_tokens(features)
        extra = found - wanted
        prompt_faithfulness = len(wanted & found) / len(wanted)
        prompt_precision = len(wanted & found) / len(found) if found else np.nan
        feature_types = {feature.get("Type") for feature in filtered}
        is_viable = (
            bool(feature_types & ORIGIN_TYPES)
            and any(token.startswith("<AMR_") for token in found)
            and bool(feature_types & PROMOTER_OR_CDS_TYPES)
        )

        faithfulness.append(prompt_faithfulness)
        precision.append(prompt_precision)
        extras_count.append(len(extra))
        any_extra.append(bool(extra))
        viability.append(is_viable)
        useful.append(is_viable and prompt_faithfulness >= 0.5)
        for category in EXTRA_CATEGORIES:
            extra_categories[category].append(
                any(token.startswith(f"<{category}_") for token in extra)
            )

    scores = {
        "faithfulness": np.asarray(faithfulness, dtype=float),
        "precision": np.asarray(precision, dtype=float),
        "extras_per_sequence": np.asarray(extras_count, dtype=float),
        "any_extra": np.asarray(any_extra, dtype=float),
        "viability": np.asarray(viability, dtype=float),
        "useful": np.asarray(useful, dtype=float),
    }
    scores.update(
        {
            f"extra_{category}": np.asarray(values, dtype=float)
            for category, values in extra_categories.items()
        }
    )
    return scores


def bootstrap_difference_ci(
    baseline: np.ndarray,
    treatment: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if baseline.shape != treatment.shape:
        raise ValueError(f"Paired arrays differ in shape: {baseline.shape} != {treatment.shape}")
    if not len(baseline):
        raise ValueError("Cannot bootstrap an empty panel")
    if n_bootstrap <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    means = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 1000):
        stop = min(start + 1000, n_bootstrap)
        indices = rng.integers(0, len(baseline), size=(stop - start, len(baseline)))
        means[start:stop] = np.nanmean(treatment[indices], axis=1) - np.nanmean(
            baseline[indices], axis=1
        )
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def mean_across_seeds(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values)
    counts = np.sum(~np.isnan(stacked), axis=0)
    result = np.full(stacked.shape[1], np.nan)
    valid = counts > 0
    result[valid] = np.nansum(stacked[:, valid], axis=0) / counts[valid]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--treatment", action="append", required=True)
    parser.add_argument("--baseline-label", default="base")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_paths = parse_seed_paths(args.baseline)
    treatment_paths = parse_seed_paths(args.treatment)
    if baseline_paths.keys() != treatment_paths.keys():
        raise ValueError(
            "baseline and treatment must contain the same decoding seeds; "
            f"baseline={sorted(baseline_paths)}, treatment={sorted(treatment_paths)}"
        )
    seeds = sorted(baseline_paths)
    if len(seeds) < 2:
        raise ValueError("Multiseed analysis requires at least two decoding seeds")

    benchmark = pd.read_parquet(args.benchmark)
    requested = [set(TOKEN_RE.findall(prompt)) for prompt in benchmark["prompt"]]
    if any(not tokens for tokens in requested):
        raise ValueError("every benchmark prompt must request at least one mapped token")

    per_seed = {}
    baseline_by_metric = {}
    treatment_by_metric = {}
    rng = np.random.default_rng(args.seed)
    for seed in seeds:
        baseline = score_annotations(baseline_paths[seed], requested)
        treatment = score_annotations(treatment_paths[seed], requested)
        seed_result = {}
        for metric in baseline:
            baseline_mean = float(np.nanmean(baseline[metric]))
            treatment_mean = float(np.nanmean(treatment[metric]))
            low, high = bootstrap_difference_ci(
                baseline[metric], treatment[metric], rng, args.bootstrap
            )
            seed_result[metric] = {
                args.baseline_label: baseline_mean,
                args.treatment_label: treatment_mean,
                "delta": treatment_mean - baseline_mean,
                "paired_bootstrap_95ci": [low, high],
            }
            baseline_by_metric.setdefault(metric, []).append(baseline[metric])
            treatment_by_metric.setdefault(metric, []).append(treatment[metric])
        per_seed[str(seed)] = seed_result

    aggregate = {}
    for metric in baseline_by_metric:
        baseline = mean_across_seeds(baseline_by_metric[metric])
        treatment = mean_across_seeds(treatment_by_metric[metric])
        baseline_mean = float(np.nanmean(baseline))
        treatment_mean = float(np.nanmean(treatment))
        low, high = bootstrap_difference_ci(baseline, treatment, rng, args.bootstrap)
        aggregate[metric] = {
            args.baseline_label: baseline_mean,
            args.treatment_label: treatment_mean,
            "delta": treatment_mean - baseline_mean,
            "paired_bootstrap_95ci": [low, high],
        }

    result = {
        "endpoint": {
            "match": f"pLAnnotate mapped token, percmatch >= {MIN_PERCMATCH}, non-fragment",
            "aggregation": "mean across fixed decoding seeds, then paired bootstrap prompts",
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
        },
        "n_prompts": len(benchmark),
        "decoding_seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
