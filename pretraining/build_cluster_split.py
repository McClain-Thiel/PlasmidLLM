"""Build immutable, source- and sequence-cluster-disjoint pretraining partitions.

MMseqs2's heuristic graph is not an exhaustive leakage certificate. Keep its raw
outputs and independently audit the held-out sequences before reporting results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq

MINORITY_TOKENS = ("<ORI_P15A>", "<ORI_PSC101>", "<AMR_CHLORAMPHENICOL>", "<COPY_LOW>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_circle(sequence: str) -> str:
    """Least rotation across both strands, in linear time (Booth's algorithm)."""
    sequence = sequence.upper()
    if not sequence or set(sequence) - set("ACGTN"):
        raise ValueError("Expected nonempty DNA containing only ACGTN")

    def least_rotation(s: str) -> str:
        doubled = s + s
        i, j, offset = 0, 1, 0
        while i < len(s) and j < len(s) and offset < len(s):
            a, b = doubled[i + offset], doubled[j + offset]
            if a == b:
                offset += 1
                continue
            if a > b:
                i += offset + 1
                if i <= j:
                    i = j + 1
            else:
                j += offset + 1
                if j <= i:
                    j = i + 1
            offset = 0
        start = min(i, j)
        return doubled[start:start + len(s)]

    return min(least_rotation(sequence), least_rotation(str(Seq(sequence).reverse_complement())))


def assign_partitions(
    rows: pd.DataFrame, clusters: pd.DataFrame, seed: int = 42,
    val_fraction: float = 0.05, test_fraction: float = 0.05,
) -> pd.DataFrame:
    """Merge MMseqs clusters connected by source-ID versions before splitting."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if clusters.member.duplicated().any():
        raise ValueError("Duplicate sequence memberships in MMseqs output")
    known = set(rows.sequence_id)
    if set(clusters.member) != known or not set(clusters.representative) <= known:
        raise ValueError("MMseqs memberships do not match the complete input sequence set")
    parent = {s: s for s in known}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for rep, member in clusters.itertuples(index=False, name=None):
        union(rep, member)
    for _, group in rows.groupby("plasmid_id", sort=False):
        ids = group.sequence_id.unique()
        for s in ids[1:]:
            union(ids[0], s)
    result = rows.copy()
    result["cluster_id"] = result.sequence_id.map(find)

    def split(cluster: str) -> str:
        value = int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest(), 16) / 2**256
        return "test" if value < test_fraction else (
            "validation" if value < test_fraction + val_fraction else "train")

    result["partition"] = result.cluster_id.map(split)
    if set(result.partition) != {"train", "validation", "test"}:
        raise ValueError("At least one partition is empty; too few independent clusters")
    return result


def build_split(
    source: Path, output: Path, mmseqs: str = "mmseqs", seed: int = 42,
    identity: float = 0.9, coverage: float = 0.9, threads: int = 4,
    val_fraction: float = 0.05, test_fraction: float = 0.05,
) -> dict:
    if not 0 < identity <= 1 or not 0 < coverage <= 1 or threads < 1:
        raise ValueError("Invalid clustering thresholds/thread count")
    # Existing partial and complete attempts are evidence, never a resume target.
    output.mkdir(parents=True, exist_ok=False)
    data = pd.read_parquet(source)
    if data[["plasmid_id", "prompt", "sequence"]].isna().any().any():
        raise ValueError("Source rows contain null IDs, prompts, or sequences")
    data["source_row"] = range(len(data))
    invalid = ~data.sequence.str.fullmatch("[ACGTNacgtn]+")
    excluded_ids = set(data.loc[invalid, "plasmid_id"])
    excluded = data[data.plasmid_id.isin(excluded_ids)].copy()
    excluded["exclusion_reason"] = "source_has_non_ACGTN_sequence"
    excluded.to_parquet(output / "excluded_source_rows.parquet", index=False)
    data = data[~data.plasmid_id.isin(excluded_ids)].copy()
    print(f"Quarantined {len(excluded)} rows from {len(excluded_ids)} sources; "
          "original rows preserved in excluded_source_rows.parquet", flush=True)
    unique = data.sequence.drop_duplicates()
    circles = {s: canonical_circle(s) for s in unique}
    ids = {s: hashlib.sha256(c.encode()).hexdigest() for s, c in circles.items()}
    data["sequence_id"] = data.sequence.map(ids)
    sequences = {ids[s]: c for s, c in circles.items()}
    fasta = output / "sequences.fasta"
    with fasta.open("w") as f:
        for sid, seq in sorted(sequences.items()):
            f.write(f">{sid}\n{seq}\n")
    data[["source_row", "plasmid_id", "sequence_id"]].to_parquet(output / "source_rows.parquet")
    required_scratch = max(4 * 1024**3, len(sequences)**2 * 48 + 3 * 1024**3)
    version = subprocess.run([mmseqs, "version"], check=True, capture_output=True, text=True).stdout.strip()
    cmd = [mmseqs, "easy-cluster", str(fasta), str(output / "clusters"), str(output / "tmp"),
           "--dbtype", "2", "--min-seq-id", str(identity), "-c", str(coverage),
           "--cov-mode", "0", "--cluster-mode", "1", "--alignment-mode", "3",
           "--seq-id-mode", "0", "--wrapped-scoring", "1",
           "-k", "12", "--max-seq-len", str(max(map(len, sequences.values()))),
           "--max-seqs", str(len(sequences)), "--threads", str(threads),
           "--split-memory-limit", "4G", "--remove-tmp-files", "0"]
    manifest = dict(source=str(source.resolve()), source_sha256=sha256_file(source),
                    seed=seed, identity=identity, coverage=coverage, coverage_mode="bidirectional",
                    circular_canonicalization=True, mmseqs_version=version, command=cmd,
                    val_fraction=val_fraction, test_fraction=test_fraction,
                    independent_leakage_audit="pending", status="clustering")
    manifest["excluded_rows"] = len(excluded)
    manifest["excluded_sources"] = len(excluded_ids)
    manifest["exclusions_sha256"] = sha256_file(output / "excluded_source_rows.parquet")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if shutil.disk_usage(output).free < required_scratch:
        manifest["status"] = "insufficient_scratch"
        manifest["required_scratch_bytes"] = required_scratch
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise OSError(f"Full-target clustering requires at least {required_scratch / 1024**3:.1f} GiB "
                      f"free scratch space; prepared inputs retained at {output}")
    with (output / "mmseqs.log").open("w") as log:
        try:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            manifest.update(status="clustering_failed", exit_code=exc.returncode)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
    clusters = pd.read_csv(output / "clusters_cluster.tsv", sep="\t", header=None,
                           names=["representative", "member"])
    data = assign_partitions(data, clusters, seed, val_fraction, test_fraction)
    manifest["partitions"] = {}
    for partition, subset in data.groupby("partition"):
        path = output / f"{partition}.parquet"
        subset.to_parquet(path, index=False)
        manifest["partitions"][partition] = dict(
            rows=len(subset), sources=subset.plasmid_id.nunique(),
            clusters=subset.cluster_id.nunique(), sha256=sha256_file(path),
            minority_counts={t: int(subset.prompt.str.contains(t, regex=False).sum())
                             for t in MINORITY_TOKENS})
    data[["source_row", "plasmid_id", "sequence_id", "cluster_id", "partition"]].to_parquet(
        output / "assignments.parquet", index=False)
    manifest["status"] = "split_built_audit_pending"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity", type=float, default=0.9)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(build_split(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
