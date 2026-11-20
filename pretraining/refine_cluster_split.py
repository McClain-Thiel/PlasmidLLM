"""Fixed-point BLAST closure over an audited plasmid cluster split.

Merges transitive connected components across cross-partition multi-HSP
homology findings from ``verify_cluster_split`` and searches any newly held-out
sequences until reaching a verified fixed point under the exact megablast
union-coverage criterion (>=90% identity HSP unions covering >=90% of both
circular molecules).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from pretraining.build_cluster_split import MINORITY_TOKENS, canonical_circle, sha256_file
from pretraining.verify_cluster_split import _run_blast_shard


def _assign_partition(cluster_id: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{cluster_id}".encode()).hexdigest(), 16) / 2**256
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "validation"
    return "train"


def _make_blast_db(seq_ids: list[str], canon_map: dict[str, str], prefix: Path) -> Path:
    fasta = prefix.with_suffix(".fasta")
    with fasta.open("w") as handle:
        for sid in sorted(seq_ids):
            handle.write(f">{sid}\n{canon_map[sid]}\n")
    subprocess.run(
        ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl", "-out", str(prefix)],
        check=True,
        capture_output=True,
    )
    return prefix


def _blast_pairs(
    query_ids: list[str],
    db_prefix: Path,
    max_target_seqs: int,
    doubled_map: dict[str, str],
    identity: float,
    coverage: float,
    threads: int,
    work: Path,
    tag: str,
) -> list[dict]:
    if not query_ids or max_target_seqs <= 0:
        return []
    ordered = sorted(query_ids)
    n_shards = min(max(1, threads), len(ordered))
    shard_args = []
    shard_files = []
    for idx in range(n_shards):
        shard_fasta = work / f"{tag}.shard{idx}.fasta"
        err_log = work / f"{tag}.shard{idx}.stderr.log"
        with shard_fasta.open("w") as handle:
            for sid in ordered[idx::n_shards]:
                handle.write(f">{sid}\n{doubled_map[sid]}\n")
        shard_args.append((shard_fasta, db_prefix, err_log, identity, coverage, max_target_seqs))
        shard_files.extend((shard_fasta, err_log))

    violations: list[dict] = []
    try:
        if n_shards == 1:
            violations.extend(_run_blast_shard(shard_args[0]))
        else:
            with ProcessPoolExecutor(max_workers=n_shards) as pool:
                for batch in pool.map(_run_blast_shard, shard_args):
                    violations.extend(batch)
    finally:
        for path in shard_files:
            if path.exists():
                path.unlink()
    return violations


def refine_split(
    base_split: Path,
    output: Path,
    threads: int = 10,
    audit_report: Path | None = None,
) -> dict:
    if threads < 1:
        raise ValueError("Thread count must be positive")
    # Preserve prior attempts; refuse to overwrite an existing split directory.
    output.mkdir(parents=True, exist_ok=False)

    base_manifest_path = base_split / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    report_path = audit_report if audit_report is not None else (base_split / "audit" / "report.json")
    report = json.loads(report_path.read_text())

    frames = []
    for part in ("train", "validation", "test"):
        path = base_split / f"{part}.parquet"
        if sha256_file(path) != base_manifest["partitions"][part]["sha256"]:
            raise ValueError(f"Base partition checksum mismatch: {part}")
        frame = pd.read_parquet(path)
        frame["partition"] = part
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    unique_seqs = data[["sequence_id", "sequence"]].drop_duplicates("sequence_id")
    canon_map = {row.sequence_id: canonical_circle(row.sequence) for row in unique_seqs.itertuples()}
    doubled_map = {sid: seq + seq for sid, seq in canon_map.items()}
    known = sorted(canon_map)
    known_set = set(known)

    parent = {sid: sid for sid in known}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[max(ra, rb)] = min(ra, rb)
        return True

    for _, group in data.groupby("cluster_id", sort=False):
        sids = group["sequence_id"].unique()
        for sid in sids[1:]:
            union(sids[0], sid)
    for _, group in data.groupby("plasmid_id", sort=False):
        sids = group["sequence_id"].unique()
        for sid in sids[1:]:
            union(sids[0], sid)

    def union_edges(edges: list[dict]) -> int:
        merges = 0
        for edge in edges:
            q, s = edge["query"], edge["subject"]
            if q in known_set and s in known_set and q != s:
                if union(q, s):
                    merges += 1
        return merges

    initial_report_edges = 0
    initial_merges = 0
    for edges in report["findings"].values():
        initial_report_edges += len(edges)
        initial_merges += union_edges(edges)

    seed = int(base_manifest.get("seed", 42))
    identity = float(base_manifest["identity"])
    coverage = float(base_manifest["coverage"])
    val_fraction = float(base_manifest.get("val_fraction", 0.05))
    test_fraction = float(base_manifest.get("test_fraction", 0.05))

    val0_sids = sorted(data.loc[data["partition"] == "validation", "sequence_id"].unique())
    test0_sids = sorted(data.loc[data["partition"] == "test", "sequence_id"].unique())

    iterations = 0
    refinement_edges_added = 0
    refinement_merges = 0

    with tempfile.TemporaryDirectory(prefix="cluster_refine_") as tmp:
        work = Path(tmp)
        all_db = _make_blast_db(known, canon_map, work / "all_db")
        val0_db = _make_blast_db(val0_sids, canon_map, work / "val0_db")
        test0_db = _make_blast_db(test0_sids, canon_map, work / "test0_db")

        intra_edges = (
            _blast_pairs(val0_sids, val0_db, len(val0_sids), doubled_map, identity, coverage, threads, work, "val0_vs_val0")
            + _blast_pairs(test0_sids, test0_db, len(test0_sids), doubled_map, identity, coverage, threads, work, "test0_vs_test0")
            + _blast_pairs(val0_sids, test0_db, len(test0_sids), doubled_map, identity, coverage, threads, work, "val0_vs_test0")
        )
        for p in work.glob("val0_db*"):
            p.unlink()
        for p in work.glob("test0_db*"):
            p.unlink()
        for p in (work / "val0_db.fasta", work / "test0_db.fasta"):
            if p.exists():
                p.unlink()

        refinement_edges_added += len(intra_edges)
        refinement_merges += union_edges(intra_edges)
        queried_against_all = set(val0_sids) | set(test0_sids)

        print(
            f"Iter 0: merged {initial_merges} components from {initial_report_edges} audit edges "
            f"+ {refinement_merges} from {len(intra_edges)} intra-heldout edges",
            flush=True,
        )

        while True:
            sid_to_cluster = {sid: find(sid) for sid in known}
            sid_to_part = {
                sid: _assign_partition(cid, seed, val_fraction, test_fraction)
                for sid, cid in sid_to_cluster.items()
            }
            if set(sid_to_part.values()) != {"train", "validation", "test"}:
                raise ValueError("Refinement collapsed at least one partition")
            heldout = sorted(sid for sid, part in sid_to_part.items() if part in ("validation", "test"))
            unqueried = sorted(set(heldout) - queried_against_all)
            n_clusters = len(set(sid_to_cluster.values()))
            print(
                f"Closure check: clusters={n_clusters}, heldout={len(heldout)}, unqueried={len(unqueried)}",
                flush=True,
            )
            if not unqueried:
                break

            iterations += 1
            edges_a = _blast_pairs(
                unqueried, all_db, len(known), doubled_map, identity, coverage, threads, work, f"iter{iterations}_unq_vs_all"
            )
            unq_prefix = work / f"iter{iterations}_unq_db"
            unq_db = _make_blast_db(unqueried, canon_map, unq_prefix)
            edges_b = _blast_pairs(
                heldout, unq_db, len(unqueried), doubled_map, identity, coverage, threads, work, f"iter{iterations}_held_vs_unq"
            )
            for p in work.glob(f"iter{iterations}_unq_db*"):
                p.unlink()

            batch = edges_a + edges_b
            refinement_edges_added += len(batch)
            merges = union_edges(batch)
            refinement_merges += merges
            queried_against_all.update(unqueried)
            print(
                f"Iter {iterations}: +{len(batch)} edges -> {merges} component merges",
                flush=True,
            )

    result = data.copy()
    result["cluster_id"] = result["sequence_id"].map(sid_to_cluster)
    result["partition"] = result["cluster_id"].map(
        lambda cid: _assign_partition(cid, seed, val_fraction, test_fraction)
    )

    part_sets = {p: result[result["partition"] == p] for p in ("train", "validation", "test")}
    if any(df.empty for df in part_sets.values()):
        raise ValueError("At least one refined partition is empty")
    for col in ("plasmid_id", "sequence_id", "cluster_id"):
        seen: set[str] = set()
        for p, df in part_sets.items():
            vals = set(df[col])
            overlap = seen & vals
            if overlap:
                raise ValueError(f"Refined split has {col} overlap in {p}: {len(overlap)}")
            seen.update(vals)

    has_prompt = "prompt" in result.columns
    if has_prompt:
        train_df = part_sets["train"]
        for tok in MINORITY_TOKENS:
            if result["prompt"].str.contains(tok, regex=False).any():
                if not train_df["prompt"].str.contains(tok, regex=False).any():
                    raise ValueError(f"Refined train partition has zero examples of {tok}")

    excluded_src = base_split / "excluded_source_rows.parquet"
    if excluded_src.exists():
        shutil.copy2(excluded_src, output / "excluded_source_rows.parquet")

    partitions_manifest = {}
    for part, subset in part_sets.items():
        path = output / f"{part}.parquet"
        subset.to_parquet(path, index=False)
        minority_counts = (
            {tok: int(subset["prompt"].str.contains(tok, regex=False).sum()) for tok in MINORITY_TOKENS}
            if has_prompt
            else {tok: 0 for tok in MINORITY_TOKENS}
        )
        partitions_manifest[part] = dict(
            rows=len(subset),
            sources=int(subset["plasmid_id"].nunique()),
            clusters=int(subset["cluster_id"].nunique()),
            sha256=sha256_file(path),
            minority_counts=minority_counts,
        )

    assign_cols = [c for c in ("source_row", "plasmid_id", "sequence_id", "cluster_id", "partition") if c in result.columns]
    result[assign_cols].to_parquet(output / "assignments.parquet", index=False)

    blast_version = subprocess.run(["blastn", "-version"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        key: base_manifest[key]
        for key in (
            "source",
            "source_sha256",
            "seed",
            "identity",
            "coverage",
            "coverage_mode",
            "circular_canonicalization",
            "val_fraction",
            "test_fraction",
            "excluded_rows",
            "excluded_sources",
            "exclusions_sha256",
        )
        if key in base_manifest
    }
    manifest.update(
        refined_from=str(base_split.resolve()),
        base_manifest_sha256=sha256_file(base_manifest_path),
        base_audit_sha256=sha256_file(report_path),
        blast_version=blast_version,
        refinement_iterations=iterations,
        initial_audit_edges=initial_report_edges,
        initial_audit_merges=initial_merges,
        refinement_edges_added=refinement_edges_added,
        refinement_merges=refinement_merges,
        independent_leakage_audit="pending",
        status="split_built_audit_pending",
        partitions=partitions_manifest,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--audit-report", type=Path, default=None)
    args = parser.parse_args()
    refine_split(args.base_split, args.output, args.threads, args.audit_report)


if __name__ == "__main__":
    main()
