#!/usr/bin/env python3
"""Summarize exhaustive BLAST output at pre-specified identity thresholds."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb


IDENTITY_THRESHOLDS = (0.0, 90.0, 95.0, 99.0)


def read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths = {}
    name = None
    length = 0
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = length
                name = line[1:].split()[0]
                if name in lengths:
                    raise ValueError(f"Duplicate FASTA identifier {name!r} in {path}")
                length = 0
            else:
                if name is None:
                    raise ValueError(f"Sequence before first header in {path}")
                length += len(line)
    if name is not None:
        lengths[name] = length
    if not lengths:
        raise ValueError(f"No sequences found in {path}")
    return lengths


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    return total + current_end - current_start + 1


def summarize_query(
    query_id: str,
    query_length: int,
    hsps: list[tuple[str, float, int, int]],
) -> dict:
    row = {"query_id": query_id, "length": query_length, "hsp_count": len(hsps)}
    for identity_threshold in IDENTITY_THRESHOLDS:
        suffix = f"pident_ge_{identity_threshold:g}"
        if query_length == 0:
            row[f"union_{suffix}"] = 0.0
            row[f"best_subject_{suffix}"] = 0.0
            row[f"subjects_{suffix}"] = 0
            continue
        all_intervals = []
        intervals_by_subject = defaultdict(list)
        for subject_id, identity, start, end in hsps:
            if identity < identity_threshold:
                continue
            interval = (start, end)
            all_intervals.append(interval)
            intervals_by_subject[subject_id].append(interval)
        row[f"union_{suffix}"] = union_length(all_intervals) / query_length
        row[f"best_subject_{suffix}"] = (
            max(
                union_length(subject_intervals)
                for subject_intervals in intervals_by_subject.values()
            )
            / query_length
            if intervals_by_subject
            else 0.0
        )
        row[f"subjects_{suffix}"] = len(intervals_by_subject)
    return row


def parse_panel(
    panel: str,
    fasta_path: Path,
    blast_path: Path,
) -> list[dict]:
    lengths = read_fasta_lengths(fasta_path)
    rows = []
    current_query = None
    current_hsps = []
    completed = set()

    def finish_query() -> None:
        if current_query is None:
            return
        rows.append(
            {
                "panel": panel,
                **summarize_query(
                    current_query,
                    lengths[current_query],
                    current_hsps,
                ),
            }
        )
        completed.add(current_query)

    with blast_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                raise ValueError(
                    f"{blast_path}:{line_number}: expected 10 fields, got {len(fields)}"
                )
            query_id, subject_id = fields[0], fields[1]
            if query_id not in lengths:
                raise KeyError(f"Unexpected query ID {query_id!r} in {blast_path}")
            if current_query is not None and query_id != current_query:
                finish_query()
                current_hsps = []
            if query_id in completed:
                raise ValueError(f"BLAST output is not grouped: {query_id} reappeared")
            current_query = query_id
            query_length = int(fields[4])
            if query_length != lengths[query_id]:
                raise ValueError(
                    f"Length mismatch for {query_id}: FASTA={lengths[query_id]}, "
                    f"BLAST={query_length}"
                )
            start, end = int(fields[6]), int(fields[7])
            current_hsps.append(
                (
                    subject_id,
                    float(fields[2]),
                    min(start, end),
                    max(start, end),
                )
            )
    finish_query()

    seen = {row["query_id"] for row in rows}
    for query_id, query_length in lengths.items():
        if query_id not in seen:
            rows.append(
                {
                    "panel": panel,
                    **summarize_query(query_id, query_length, []),
                }
            )
    rows.sort(key=lambda row: row["query_id"])
    return rows


def bootstrap_median_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        first_sample = rng.choice(first, len(first), replace=True)
        second_sample = rng.choice(second, len(second), replace=True)
        differences[index] = np.median(first_sample) - np.median(second_sample)
    return [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]


def summarize_values(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "fraction_at_least_50pct": sum(value >= 0.5 for value in values) / len(values),
        "fraction_at_least_80pct": sum(value >= 0.8 for value in values) / len(values),
        "fraction_at_least_90pct": sum(value >= 0.9 for value in values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--query-lengths-json",
        type=Path,
        help=(
            "Optional panel-to-query-length mapping written by "
            "exhaustive_training_mosaic.py; defaults to INPUT_DIR/query_lengths.json"
        ),
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["base", "grpo", "real_control"],
    )
    parser.add_argument(
        "--empty-query",
        action="append",
        default=[],
        help="Known empty query omitted from FASTA, in PANEL=QUERY_ID form",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--wandb-project", default="PlasmidLLM")
    parser.add_argument(
        "--wandb-run-name",
        default="neurips25202_exhaustive_training_mosaic_thresholds",
    )
    args = parser.parse_args()

    query_lengths_path = args.query_lengths_json or args.input_dir / "query_lengths.json"
    query_lengths = (
        json.loads(query_lengths_path.read_text()) if query_lengths_path.is_file() else {}
    )

    empty_queries_by_panel = defaultdict(list)
    for value in args.empty_query:
        panel, separator, query_id = value.partition("=")
        if not separator or not panel or not query_id:
            raise ValueError(f"Invalid --empty-query value: {value!r}")
        if panel not in args.panels:
            raise ValueError(f"Empty-query panel {panel!r} is not in --panels {args.panels}")
        empty_queries_by_panel[panel].append(query_id)

    all_rows = []
    for panel in args.panels:
        fasta_path = args.input_dir / f"{panel}.fasta"
        blast_path = args.input_dir / f"{panel}_vs_training.tsv"
        if not fasta_path.is_file():
            raise FileNotFoundError(fasta_path)
        if not blast_path.is_file():
            raise FileNotFoundError(blast_path)
        panel_rows = parse_panel(panel, fasta_path, blast_path)
        seen_query_ids = {row["query_id"] for row in panel_rows}
        for query_id, query_length in query_lengths.get(panel, {}).items():
            if query_id in seen_query_ids:
                continue
            if query_length != 0:
                raise ValueError(
                    f"{query_id!r} is absent from {fasta_path} but has nonzero "
                    f"length {query_length}"
                )
            panel_rows.append(
                {
                    "panel": panel,
                    **summarize_query(query_id, 0, []),
                }
            )
            seen_query_ids.add(query_id)
        for query_id in empty_queries_by_panel[panel]:
            if query_id in seen_query_ids:
                raise ValueError(f"Known-empty query {query_id!r} is already present in {panel}")
            panel_rows.append(
                {
                    "panel": panel,
                    **summarize_query(query_id, 0, []),
                }
            )
        panel_rows.sort(key=lambda row: row["query_id"])
        all_rows.extend(panel_rows)

    fieldnames = list(all_rows[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    results = {
        "method": {
            "primary_identity_threshold_percent": 90.0,
            "sensitivity_identity_thresholds_percent": [0.0, 95.0, 99.0],
            "coverage_definitions": {
                "union": "union of query intervals across every reported training subject",
                "best_subject": "union of query intervals for the single best-covered training subject",
            },
            "blast_evalue": 1e-10,
            "minimum_hsp_length": None,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "manually_included_empty_queries": dict(empty_queries_by_panel),
        },
        "panels": {},
        "comparisons_to_real_control": {},
    }
    rows_by_panel = {
        panel: [row for row in all_rows if row["panel"] == panel] for panel in args.panels
    }
    for panel, rows in rows_by_panel.items():
        results["panels"][panel] = {}
        for identity_threshold in IDENTITY_THRESHOLDS:
            suffix = f"pident_ge_{identity_threshold:g}"
            results["panels"][panel][suffix] = {
                "union": summarize_values([float(row[f"union_{suffix}"]) for row in rows]),
                "best_subject": summarize_values(
                    [float(row[f"best_subject_{suffix}"]) for row in rows]
                ),
            }

    if "real_control" in rows_by_panel:
        control_rows = rows_by_panel["real_control"]
        for panel in args.panels:
            if panel == "real_control":
                continue
            results["comparisons_to_real_control"][panel] = {}
            for identity_threshold in IDENTITY_THRESHOLDS:
                suffix = f"pident_ge_{identity_threshold:g}"
                results["comparisons_to_real_control"][panel][suffix] = {}
                for coverage_type in ["union", "best_subject"]:
                    column = f"{coverage_type}_{suffix}"
                    panel_values = np.asarray([float(row[column]) for row in rows_by_panel[panel]])
                    control_values = np.asarray([float(row[column]) for row in control_rows])
                    difference = float(np.median(panel_values) - np.median(control_values))
                    results["comparisons_to_real_control"][panel][suffix][coverage_type] = {
                        "median_difference": difference,
                        "independent_bootstrap_95ci": bootstrap_median_difference(
                            panel_values,
                            control_values,
                            resamples=args.bootstrap_resamples,
                            seed=args.bootstrap_seed,
                        ),
                    }

    args.output_json.write_text(json.dumps(results, indent=2) + "\n")

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=[
            "neurips25202",
            "rebuttal",
            "evaluation",
            "memorization",
            "blast",
            "sensitivity-analysis",
        ],
        config=results["method"],
    )
    try:
        for panel in args.panels:
            for identity_threshold in IDENTITY_THRESHOLDS:
                suffix = f"pident_ge_{identity_threshold:g}"
                for coverage_type in ["union", "best_subject"]:
                    summary = results["panels"][panel][suffix][coverage_type]
                    run.log(
                        {
                            "panel": panel,
                            "identity_threshold": identity_threshold,
                            "coverage_type": coverage_type,
                            "median_coverage": summary["median"],
                            "mean_coverage": summary["mean"],
                            "fraction_at_least_90pct": summary["fraction_at_least_90pct"],
                        }
                    )
        artifact = wandb.Artifact(
            "neurips25202-exhaustive-training-mosaic-thresholds",
            type="evaluation",
            metadata=results["method"],
        )
        artifact.add_file(str(args.output_json))
        artifact.add_file(str(args.output_csv))
        run.log_artifact(artifact)
    finally:
        run.finish()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
