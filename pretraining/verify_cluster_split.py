"""Independent exhaustive-target BLAST audit of all three partition pairs.

Report any pair with >=90% identity HSPs whose unions cover >=90% of both
molecules. Doubled queries allow circular boundary matches; modulo intervals
prevent double counting. This is a conservative mosaic screen, not proof of
absence of homology below the declared thresholds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import pandas as pd

from pretraining.build_cluster_split import canonical_circle, sha256_file


def interval_union(intervals: list[tuple[int, int]]) -> int:
    end, total = 0, 0
    for left, right in sorted(intervals):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def circular_intervals(start: int, end: int, length: int) -> list[tuple[int, int]]:
    left, right = min(start, end) - 1, max(start, end)
    span = right - left
    if span >= length:
        return [(0, length)]
    left %= length
    right = left + span
    return [(left, min(right, length))] + ([(0, right - length)] if right > length else [])


def violations_from_blast_lines(
    lines: Iterable[str], identity: float, coverage: float,
) -> list[dict]:
    violations = []
    # BLAST tabular output groups HSPs by query and subject.
    current = None
    qi, si = [], []
    qlen = slen = 0

    def finish() -> None:
        if current is None:
            return
        qc, sc = interval_union(qi) / qlen, interval_union(si) / slen
        if min(qc, sc) >= coverage:
            violations.append(dict(query=current[0], subject=current[1],
                                   query_coverage=qc, subject_coverage=sc))

    for line in lines:
        q, s, ident, qs, qe, ss, se, ql, sl = line.rstrip().split("\t")
        key = (q, s)
        if key != current:
            finish()
            current, qi, si = key, [], []
        qlen, slen = int(ql) // 2, int(sl)
        if float(ident) >= identity * 100:
            qi.extend(circular_intervals(int(qs), int(qe), qlen))
            si.extend(circular_intervals(int(ss), int(se), slen))
    finish()
    return violations


def violations_from_blast(path: Path, identity: float, coverage: float) -> list[dict]:
    with path.open() as f:
        return violations_from_blast_lines(f, identity, coverage)


def _run_blast_shard(args: tuple[Path, Path, Path, float, float, int]) -> list[dict]:
    shard_fasta, db_prefix, err_log_path, identity, coverage, max_target_seqs = args
    command = [
        "blastn", "-task", "megablast", "-query", str(shard_fasta),
        "-db", str(db_prefix),
        "-outfmt", "6 qseqid sseqid pident qstart qend sstart send qlen slen",
        "-perc_identity", str(100 * identity),
        "-max_target_seqs", str(max_target_seqs),
        "-dust", "no", "-evalue", "1e-10", "-num_threads", "1",
    ]
    with err_log_path.open("w") as err_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
            bufsize=1024 * 1024,
        )
        assert proc.stdout is not None
        violations = violations_from_blast_lines(proc.stdout, identity, coverage)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"blastn shard failed (rc={rc}): {err_log_path.read_text()[:500]}")
    return violations


def verify_split(split_dir: Path, output: Path, threads: int = 4) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames = {}
    target_fastas = {}
    doubled_records = {}
    for part in ("train", "validation", "test"):
        path = split_dir / f"{part}.parquet"
        if sha256_file(path) != manifest["partitions"][part]["sha256"]:
            raise ValueError(f"Partition checksum mismatch: {part}")
        frame = pd.read_parquet(path, columns=["sequence_id", "sequence", "cluster_id", "plasmid_id"])
        frames[part] = frame

    with tempfile.TemporaryDirectory(prefix="cluster_audit_") as tmp:
        work = Path(tmp)
        for part, frame in frames.items():
            fasta = work / f"{part}.fasta"
            records = []
            with fasta.open("w") as f:
                for row in frame.drop_duplicates("sequence_id").itertuples():
                    seq = canonical_circle(row.sequence)
                    f.write(f">{row.sequence_id}\n{seq}\n")
                    records.append((row.sequence_id, seq + seq))
            target_fastas[part] = fasta
            doubled_records[part] = records

        findings, commands = {}, []
        for target, query in combinations(frames, 2):
            for col in ("plasmid_id", "sequence_id", "cluster_id"):
                if set(frames[target][col]) & set(frames[query][col]):
                    raise ValueError(f"{col} overlap: {target}/{query}")
            database = work / f"{target}_db"
            build = ["makeblastdb", "-in", str(target_fastas[target]), "-dbtype", "nucl", "-out", str(database)]
            subprocess.run(build, check=True, capture_output=True)

            query_records = doubled_records[query]
            n_shards = min(max(1, threads), max(1, len(query_records)))
            max_target_seqs = frames[target].sequence_id.nunique()
            logical_cmd = [
                "blastn", "-task", "megablast", "-query", f"<{query}_doubled:{n_shards}_shards>",
                "-db", str(database),
                "-outfmt", "6 qseqid sseqid pident qstart qend sstart send qlen slen",
                "-perc_identity", str(100 * manifest["identity"]),
                "-max_target_seqs", str(max_target_seqs),
                "-dust", "no", "-evalue", "1e-10", "-num_threads", "1",
            ]
            commands.append(logical_cmd)

            shard_args = []
            for idx in range(n_shards):
                shard_rows = query_records[idx::n_shards]
                shard_fasta = work / f"{query}_vs_{target}.shard{idx}.fasta"
                err_log = output / f"{query}_vs_{target}.shard{idx}.stderr.log"
                with shard_fasta.open("w") as sf:
                    for sid, seq in shard_rows:
                        sf.write(f">{sid}\n{seq}\n")
                shard_args.append((
                    shard_fasta,
                    database,
                    err_log,
                    manifest["identity"],
                    manifest["coverage"],
                    max_target_seqs,
                ))

            pair_violations = []
            if n_shards == 1:
                pair_violations.extend(_run_blast_shard(shard_args[0]))
            else:
                with ProcessPoolExecutor(max_workers=n_shards) as pool:
                    for result in pool.map(_run_blast_shard, shard_args):
                        pair_violations.extend(result)
            # Clean empty shard stderr logs; preserve non-empty logs if any.
            for _, _, err_log, _, _, _ in shard_args:
                if err_log.exists() and err_log.stat().st_size == 0:
                    err_log.unlink()
            findings[f"{query}_vs_{target}"] = pair_violations
            print(
                f"Completed {query}_vs_{target} ({n_shards} shards): {len(pair_violations)} violations",
                flush=True,
            )

    passed = not any(findings.values())
    report = dict(passed=passed, findings=findings, commands=commands,
                  blast_version=subprocess.run(["blastn", "-version"], check=True,
                                               capture_output=True, text=True).stdout,
                  partition_sha256={p: info["sha256"] for p, info in manifest["partitions"].items()})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise ValueError(f"Cross-partition homology detected; inspect {output / 'report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    print(json.dumps(verify_split(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
