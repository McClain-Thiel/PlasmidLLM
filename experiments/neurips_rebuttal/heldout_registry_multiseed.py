#!/usr/bin/env python3
"""Compare two fixed-seed model sets on held-out registry recovery."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"<(?:AMR|PROM|ORI|ELEM|REPORTER|TAG|REP)_[A-Z0-9_]+>")
MIN_PERCMATCH = 95.0


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


def detected_sseqids(features: list[dict]) -> set[str]:
    return {
        str(feature["sseqid"]).lower()
        for feature in features
        if (feature.get("percmatch") or 0) >= MIN_PERCMATCH
        and not feature.get("fragment")
        and feature.get("sseqid") is not None
    }


def score_annotations(
    path: Path,
    requested: list[list[str]],
    heldout_by_token: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    annotations = load_annotations(path)
    expected_ids = {f"gen_{i}" for i in range(len(requested))}
    missing = sorted(expected_ids - annotations.keys())
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} expected sequences; first={missing[:5]}"
        )

    prompt_scores = []
    request_hits = []
    for i, tokens in enumerate(requested):
        if not tokens:
            continue
        found = detected_sseqids(annotations[f"gen_{i}"])
        hits = [bool(found & heldout_by_token[token]) for token in tokens]
        prompt_scores.append(np.mean(hits))
        request_hits.extend(hits)
    return (
        np.asarray(prompt_scores, dtype=float),
        np.asarray(request_hits, dtype=float),
    )


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    if not len(values):
        raise ValueError("Cannot bootstrap an empty panel")
    if n_bootstrap <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    means = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 1000):
        stop = min(start + 1000, n_bootstrap)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(scores: np.ndarray, request_hits: np.ndarray) -> dict[str, float]:
    return {
        "prompt_macro": float(scores.mean()),
        "request_micro": float(request_hits.mean()),
        "any_prompt": float((scores > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--heldout-registry", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument(
        "--treatment",
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
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
        raise ValueError("multiseed analysis requires at least two decoding seeds")

    benchmark = pd.read_parquet(args.benchmark)
    heldout = pd.read_parquet(args.heldout_registry)
    heldout_by_token = {
        token: {str(sseqid).lower() for sseqid in rows["sseqid"]}
        for token, rows in heldout.groupby("token")
    }
    requested = [
        [token for token in TOKEN_RE.findall(prompt) if token in heldout_by_token]
        for prompt in benchmark["prompt"]
    ]
    eligible = np.asarray([bool(tokens) for tokens in requested])
    if not eligible.any():
        raise ValueError("No benchmark prompts request a held-out registry token")

    baseline_scores = []
    treatment_scores = []
    baseline_requests = []
    treatment_requests = []
    per_seed = {}
    rng = np.random.default_rng(args.seed)

    for seed in seeds:
        baseline_prompt, baseline_request = score_annotations(
            baseline_paths[seed], requested, heldout_by_token
        )
        treatment_prompt, treatment_request = score_annotations(
            treatment_paths[seed], requested, heldout_by_token
        )
        difference = treatment_prompt - baseline_prompt
        low, high = bootstrap_ci(difference, rng, args.bootstrap)
        per_seed[str(seed)] = {
            args.baseline_label: summarize(baseline_prompt, baseline_request),
            args.treatment_label: summarize(treatment_prompt, treatment_request),
            "delta_prompt_macro": float(difference.mean()),
            "paired_bootstrap_95ci": [low, high],
        }
        baseline_scores.append(baseline_prompt)
        treatment_scores.append(treatment_prompt)
        baseline_requests.append(baseline_request)
        treatment_requests.append(treatment_request)

    baseline_scores = np.stack(baseline_scores)
    treatment_scores = np.stack(treatment_scores)
    baseline_requests = np.stack(baseline_requests)
    treatment_requests = np.stack(treatment_requests)
    baseline_mean = baseline_scores.mean(axis=0)
    treatment_mean = treatment_scores.mean(axis=0)
    difference = treatment_mean - baseline_mean
    low, high = bootstrap_ci(difference, rng, args.bootstrap)
    baseline_summary = summarize(baseline_mean, baseline_requests.mean(axis=0))
    treatment_summary = summarize(treatment_mean, treatment_requests.mean(axis=0))
    baseline_summary["any_prompt"] = float((baseline_scores > 0).mean())
    treatment_summary["any_prompt"] = float((treatment_scores > 0).mean())
    baseline_summary["any_prompt_at_least_one_seed"] = float(
        (baseline_scores > 0).any(axis=0).mean()
    )
    treatment_summary["any_prompt_at_least_one_seed"] = float(
        (treatment_scores > 0).any(axis=0).mean()
    )

    result = {
        "endpoint": {
            "unit": "prompt",
            "score": "mean fraction of requested held-out registry tokens recovered",
            "match": f"pLAnnotate exact sseqid, percmatch >= {MIN_PERCMATCH}, non-fragment",
            "aggregation": "mean across fixed decoding seeds, then paired bootstrap prompts",
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
        },
        "n_benchmark_prompts": len(benchmark),
        "n_eligible_prompts": int(eligible.sum()),
        "n_heldout_request_units": int(sum(map(len, requested))),
        "decoding_seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            args.baseline_label: baseline_summary,
            args.treatment_label: treatment_summary,
            "delta_prompt_macro": float(difference.mean()),
            "paired_bootstrap_95ci": [low, high],
            "positive_and_significant": bool(low > 0),
        },
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
