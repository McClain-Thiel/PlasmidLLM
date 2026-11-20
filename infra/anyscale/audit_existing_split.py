"""Run independent exhaustive-target BLAST audit on an existing MMseqs2 cluster split."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import boto3
import wandb

from plasmid_llm.provenance import file_sha256, s3_readback_sha256
from pretraining.verify_cluster_split import verify_split


def main() -> None:
    split_prefix = os.environ["S3_SPLIT_PREFIX"].rstrip("/")
    uri = urlparse(split_prefix)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("S3_SPLIT_PREFIX must be a nonempty S3 run prefix")
    bucket = uri.netloc
    key_prefix = uri.path.strip("/")
    audit_key_prefix = f"{key_prefix}/audit"

    client = boto3.client("s3")
    if client.list_objects_v2(
        Bucket=bucket, Prefix=audit_key_prefix + "/", MaxKeys=1,
    ).get("KeyCount", 0):
        raise FileExistsError(f"Audit destination already exists: {split_prefix}/audit")

    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "ncbi-blast+"], check=True)

    split_dir = Path("data/cluster_v2").resolve()
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "train.parquet", "validation.parquet", "test.parquet"):
        target = split_dir / name
        if not target.exists():
            subprocess.run(
                ["aws", "s3", "cp", f"{split_prefix}/{name}", str(target), "--no-progress"],
                check=True,
            )

    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_source_sha = os.environ.get("PLASMID_TRAIN_SOURCE_SHA256")
    if expected_source_sha and manifest.get("source_sha256") != expected_source_sha:
        raise ValueError("Split manifest source_sha256 differs from approved dataset")
    for part in ("train", "validation", "test"):
        actual = file_sha256(split_dir / f"{part}.parquet")
        expected = manifest["partitions"][part]["sha256"]
        if actual != expected:
            raise ValueError(f"Downloaded {part}.parquet sha256 mismatch: {actual} != {expected}")

    audit_dir = split_dir / "audit"
    threads = int(os.environ.get("PLASMID_CPU_THREADS", "32"))
    with wandb.init(
        project="PlasmidLLM",
        name=f"{os.environ['PLASMID_RUN_ID']}_cluster_audit",
        tags=["plasmidrl", "data-preparation", "cluster-disjoint-v2", "blast-audit"],
        config=dict(
            split_prefix=split_prefix,
            source_sha256=manifest.get("source_sha256"),
            partition_sha256={p: manifest["partitions"][p]["sha256"] for p in ("train", "validation", "test")},
            threads=threads,
        ),
    ) as run:
        try:
            report = verify_split(split_dir, audit_dir, threads=threads)
            report_path = audit_dir / "report.json"
            digest = file_sha256(report_path)
            report_key = f"{audit_key_prefix}/report.json"
            with report_path.open("rb") as src:
                client.put_object(
                    Bucket=bucket,
                    Key=report_key,
                    Body=src,
                    IfNoneMatch="*",
                    ContentType="application/json",
                )
            if s3_readback_sha256(client, bucket, report_key) != digest:
                raise ValueError(f"S3 readback checksum mismatch for {report_key}")
            run.log(
                {
                    "audit_passed": int(report["passed"]),
                    **{f"{p}/rows": info["rows"] for p, info in manifest["partitions"].items()},
                }
            )
            artifact = wandb.Artifact(f"{run.id}-split-audit", type="dataset")
            artifact.add_file(str(manifest_path))
            artifact.add_file(str(report_path))
            run.log_artifact(artifact).wait()
            print(f"Split audit passed and archived to {split_prefix}/audit/report.json", flush=True)
        finally:
            if audit_dir.exists():
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "sync",
                        str(audit_dir),
                        f"{split_prefix}/audit",
                        "--no-progress",
                        "--no-follow-symlinks",
                        "--exclude",
                        "report.json",
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
