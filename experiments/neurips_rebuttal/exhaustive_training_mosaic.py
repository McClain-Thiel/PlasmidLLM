#!/usr/bin/env python3
"""Exhaustive BLAST mosaic coverage against every pretraining plasmid.

This corrects the earlier rebuttal analysis that used ``-max_target_seqs 5`` while
interpreting the result as a union over the full training corpus.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import duckdb
import numpy as np


def write_training_fasta(train_pairs: Path, output: Path) -> tuple[int, set[str]]:
    rows = (
        duckdb.connect()
        .execute(
            """
        SELECT CAST(plasmid_id AS VARCHAR), ANY_VALUE(sequence)
        FROM read_parquet(?)
        GROUP BY 1
        ORDER BY 1
        """,
            [str(train_pairs)],
        )
        .fetchall()
    )
    train_ids = set()
    with output.open("w") as handle:
        for index, (plasmid_id, sequence) in enumerate(rows):
            sequence = (sequence or "").strip()
            if not sequence:
                raise ValueError(f"Empty training sequence for plasmid_id={plasmid_id}")
            train_ids.add(plasmid_id)
            handle.write(f">train_{index}\n{sequence}\n")
    return len(rows), train_ids


def write_parquet_fasta(parquet: Path, panel: str, output: Path) -> dict[str, int]:
    rows = (
        duckdb.connect()
        .execute(
            "SELECT sequence FROM read_parquet(?)",
            [str(parquet)],
        )
        .fetchall()
    )
    lengths = {}
    with output.open("w") as handle:
        for index, (sequence,) in enumerate(rows):
            sequence = (sequence or "").strip()
            query_id = f"{panel}_{index}"
            lengths[query_id] = len(sequence)
            if sequence:
                handle.write(f">{query_id}\n{sequence}\n")
    return lengths


def read_fasta(path: Path):
    name = None
    chunks = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                if name is None:
                    raise ValueError(f"Sequence before first FASTA header in {path}")
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def write_control_fasta(
    source: Path,
    train_ids: set[str],
    output: Path,
) -> dict[str, int]:
    lengths = {}
    with output.open("w") as handle:
        for index, (source_name, sequence) in enumerate(read_fasta(source)):
            plasmid_id = source_name.split("|")[0]
            if plasmid_id in train_ids:
                continue
            if not sequence:
                raise ValueError(f"Empty control sequence for {source_name}")
            query_id = f"control_{index}"
            lengths[query_id] = len(sequence)
            handle.write(f">{query_id}\n{sequence}\n")
    return lengths


def run_blast(
    query: Path,
    database: Path,
    output: Path,
    threads: int,
    max_target_seqs: int,
) -> float:
    started = time.monotonic()
    subprocess.run(
        [
            "blastn",
            "-query",
            str(query),
            "-db",
            str(database),
            "-out",
            str(output),
            "-outfmt",
            ("6 qseqid sseqid pident length qlen slen qstart qend bitscore evalue"),
            "-max_target_seqs",
            str(max_target_seqs),
            "-evalue",
            "1e-10",
            "-num_threads",
            str(threads),
        ],
        check=True,
    )
    return time.monotonic() - started


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


def parse_mosaic(
    blast_tsv: Path,
    query_lengths: dict[str, int],
) -> tuple[dict[str, float], dict]:
    coverage = {query_id: 0.0 for query_id in query_lengths}
    current_query = None
    intervals = []
    subjects = set()
    hsp_count = 0
    subject_counts = {}
    hsp_counts = {}
    completed_queries = set()

    def finish_query() -> None:
        if current_query is None:
            return
        coverage[current_query] = union_length(intervals) / query_lengths[current_query]
        subject_counts[current_query] = len(subjects)
        hsp_counts[current_query] = len(intervals)
        completed_queries.add(current_query)

    with blast_tsv.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10:
                raise ValueError(
                    f"{blast_tsv}:{line_number}: expected 10 columns, got {len(fields)}"
                )
            query_id, subject_id = fields[0], fields[1]
            if query_id not in query_lengths:
                raise KeyError(f"Unexpected query ID in BLAST output: {query_id}")
            if current_query is not None and query_id != current_query:
                finish_query()
                intervals = []
                subjects = set()
            if query_id in completed_queries:
                raise ValueError(f"BLAST output is not grouped by query; {query_id} reappeared")
            current_query = query_id
            start, end = int(fields[6]), int(fields[7])
            intervals.append((min(start, end), max(start, end)))
            subjects.add(subject_id)
            hsp_count += 1
    finish_query()

    subject_count_values = list(subject_counts.values())
    diagnostics = {
        "total_hsps": hsp_count,
        "queries_with_hits": len(subject_counts),
        "max_subjects_per_query": max(subject_counts.values(), default=0),
        "median_subjects_per_query": statistics.median(
            subject_count_values if subject_count_values else [0]
        ),
        "max_hsps_per_query": max(hsp_counts.values(), default=0),
    }
    return coverage, diagnostics


def bootstrap_median_difference(
    first: list[float],
    second: list[float],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    differences = np.empty(resamples)
    for index in range(resamples):
        first_sample = rng.choice(first_array, len(first_array), replace=True)
        second_sample = rng.choice(second_array, len(second_array), replace=True)
        differences[index] = np.median(first_sample) - np.median(second_sample)
    return [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]


def summarise(
    coverage: dict[str, float],
    lengths: dict[str, int],
    diagnostics: dict,
) -> dict:
    values = list(coverage.values())
    bins = {}
    for lower, upper in [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20)]:
        bin_values = [
            value
            for query_id, value in coverage.items()
            if lower * 1000 <= lengths[query_id] < upper * 1000
        ]
        bins[f"{lower}-{upper}kb"] = {
            "n": len(bin_values),
            "median": (float(statistics.median(bin_values)) if bin_values else None),
            "mean": float(statistics.mean(bin_values)) if bin_values else None,
        }
    return {
        "n": len(values),
        "no_hit": sum(value == 0.0 for value in values),
        "median_mosaic_coverage": float(statistics.median(values)),
        "mean_mosaic_coverage": float(statistics.mean(values)),
        "fraction_at_least_90pct": sum(value >= 0.9 for value in values) / len(values),
        "length_bins": bins,
        "blast_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        help="Named generation panel in NAME=/path/to/generations.parquet form",
    )
    parser.add_argument("--control-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-target-seqs", type=int, default=25000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--wandb-project", default="PlasmidLLM")
    parser.add_argument(
        "--wandb-run-name",
        default="neurips25202_exhaustive_training_mosaic",
    )
    args = parser.parse_args()

    panels = {}
    for item in args.panel:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Invalid --panel value: {item!r}")
        if name in panels:
            raise ValueError(f"Duplicate panel name: {name}")
        panels[name] = Path(raw_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    training_fasta = output_dir / "training_plasmids.fasta"
    database = output_dir / "training_db"

    n_train, train_ids = write_training_fasta(args.train_pairs, training_fasta)
    if args.max_target_seqs < n_train:
        raise ValueError(f"max_target_seqs={args.max_target_seqs} is below n_train={n_train}")
    subprocess.run(
        [
            "makeblastdb",
            "-in",
            str(training_fasta),
            "-dbtype",
            "nucl",
            "-out",
            str(database),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    panel_lengths = {}
    for name, parquet in panels.items():
        panel_lengths[name] = write_parquet_fasta(
            parquet,
            name,
            output_dir / f"{name}.fasta",
        )
    panel_lengths["real_control"] = write_control_fasta(
        args.control_fasta,
        train_ids,
        output_dir / "real_control.fasta",
    )
    query_lengths_json = output_dir / "query_lengths.json"
    query_lengths_json.write_text(json.dumps(panel_lengths, indent=2) + "\n")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required for this evaluation") from error

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["neurips25202", "rebuttal", "evaluation", "memorization", "blast"],
        config={
            "train_pairs": str(args.train_pairs),
            "panels": {name: str(path) for name, path in panels.items()},
            "control_fasta": str(args.control_fasta),
            "n_training_plasmids": n_train,
            "threads": args.threads,
            "max_target_seqs": args.max_target_seqs,
            "evalue": 1e-10,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )

    results = {
        "method": {
            "n_training_plasmids": n_train,
            "max_target_seqs": args.max_target_seqs,
            "evalue": 1e-10,
            "coverage": "union of all query intervals across all reported training subjects",
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "panels": {},
        "comparisons_to_real_control": {},
    }

    all_coverage = {}
    try:
        for name, lengths in panel_lengths.items():
            blast_tsv = output_dir / f"{name}_vs_training.tsv"
            runtime_seconds = run_blast(
                output_dir / f"{name}.fasta",
                database,
                blast_tsv,
                args.threads,
                args.max_target_seqs,
            )
            coverage, diagnostics = parse_mosaic(blast_tsv, lengths)
            all_coverage[name] = coverage
            summary = summarise(coverage, lengths, diagnostics)
            summary["runtime_seconds"] = runtime_seconds
            summary["blast_tsv_bytes"] = blast_tsv.stat().st_size
            results["panels"][name] = summary
            run.log(
                {
                    f"{name}/median_mosaic_coverage": summary["median_mosaic_coverage"],
                    f"{name}/mean_mosaic_coverage": summary["mean_mosaic_coverage"],
                    f"{name}/fraction_at_least_90pct": summary["fraction_at_least_90pct"],
                    f"{name}/runtime_seconds": runtime_seconds,
                    f"{name}/blast_tsv_bytes": summary["blast_tsv_bytes"],
                }
            )

        control_values = list(all_coverage["real_control"].values())
        for name in panels:
            panel_values = list(all_coverage[name].values())
            difference = statistics.median(panel_values) - statistics.median(control_values)
            interval = bootstrap_median_difference(
                panel_values,
                control_values,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )
            results["comparisons_to_real_control"][name] = {
                "median_difference": float(difference),
                "independent_bootstrap_95ci": interval,
            }
            run.log(
                {
                    f"{name}/median_difference_vs_real_control": difference,
                    f"{name}/median_difference_vs_real_control_ci_low": interval[0],
                    f"{name}/median_difference_vs_real_control_ci_high": interval[1],
                }
            )

        output_json = output_dir / "exhaustive_training_mosaic.json"
        output_json.write_text(json.dumps(results, indent=2) + "\n")
        artifact = wandb.Artifact(
            "neurips25202-exhaustive-training-mosaic",
            type="evaluation",
            metadata=results["method"],
        )
        artifact.add_file(str(output_json))
        artifact.add_file(str(query_lengths_json))
        run.log_artifact(artifact)
        print(json.dumps(results, indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
