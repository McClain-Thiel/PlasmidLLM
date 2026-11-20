"""CPU-only cluster preparation; archive complete and failed attempts to a fresh S3 prefix."""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import boto3
import wandb

from pretraining.build_cluster_split import build_split, sha256_file
from pretraining.verify_cluster_split import verify_split

MMSEQS_URL = "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-avx2.tar.gz"
MMSEQS_SHA256 = "bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce"


def install_mmseqs() -> str:
    """Pin the circular-scoring implementation and exercise the failing code path."""
    directory = Path(".tools/mmseqs18").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "release.tar.gz"
    with urllib.request.urlopen(MMSEQS_URL, timeout=120) as source, archive.open("xb") as target:
        shutil.copyfileobj(source, target)
    if sha256_file(archive) != MMSEQS_SHA256:
        raise ValueError("MMseqs2 archive checksum differs from the pinned official release")
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(directory)], check=True)
    binary = str(directory / "mmseqs/bin/mmseqs")
    fasta = directory / "preflight.fasta"
    fasta.write_text(">a\n" + "ACGTGCAATGCT" * 20 + "\n>b\n" + "GCAATGCTACGT" * 20 + "\n")
    database, hits, result = [str(directory / name) for name in ("db", "hits", "result")]
    commands = [
        [binary, "createdb", str(fasta), database, "--dbtype", "2"],
        [binary, "prefilter", database, database, hits, "-k", "8", "--threads", "1"],
        [binary, "rescorediagonal", database, database, hits, result,
         "--rescore-mode", "2", "--wrapped-scoring", "1", "--threads", "1"],
    ]
    with (directory / "preflight.log").open("x") as log:
        for command in commands:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    print("Pinned MMseqs2 circular-scoring preflight passed", flush=True)
    return binary


def main() -> None:
    source_prefix = os.environ["S3_DATA_PREFIX"].rstrip("/")
    destination = os.environ["S3_SPLIT_PREFIX"].rstrip("/")
    uri = urlparse(destination)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("S3_SPLIT_PREFIX must be a nonempty S3 run prefix")
    if boto3.client("s3").list_objects_v2(
        Bucket=uri.netloc, Prefix=uri.path.lstrip("/") + "/", MaxKeys=1,
    ).get("KeyCount", 0):
        raise FileExistsError(f"Split destination already exists: {destination}")
    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "ncbi-blast+"], check=True)
    mmseqs = install_mmseqs()
    source = Path("data/training_pairs_v4.parquet")
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(["aws", "s3", "cp", f"{source_prefix}/data/{source.name}", str(source)], check=True)
    if sha256_file(source) != os.environ["PLASMID_TRAIN_SOURCE_SHA256"]:
        raise ValueError("Training source differs from the approved input checksum")
    output = Path("data/cluster_v2")
    if output.exists():
        raise FileExistsError(output)
    threads = int(os.environ.get("PLASMID_CPU_THREADS", "8"))
    with wandb.init(project="PlasmidLLM", name=f"{os.environ['PLASMID_RUN_ID']}_cluster_data",
                    tags=["plasmidrl", "data-preparation", "cluster-disjoint-v2"],
                    config=dict(source_sha256=sha256_file(source), s3_output=destination,
                                mmseqs_archive=MMSEQS_URL, mmseqs_sha256=MMSEQS_SHA256,
                                identity=0.9, bidirectional_coverage=0.9, threads=threads)) as run:
        try:
            manifest = build_split(source, output, mmseqs=mmseqs, threads=threads)
            subprocess.run(["aws", "s3", "sync", str(output), destination, "--no-follow-symlinks"], check=True)
            audit = verify_split(output, output / "audit", threads=threads)
            run.log({"audit_passed": int(audit["passed"]),
                     **{f"{p}/rows": info["rows"] for p, info in manifest["partitions"].items()}})
            artifact = wandb.Artifact(f"{run.id}-split-manifests", type="dataset")
            artifact.add_file(str(output / "manifest.json"))
            artifact.add_file(str(output / "audit/report.json"))
            run.log_artifact(artifact).wait()
        finally:
            # A failed audit is useful evidence; never discard the raw search results.
            if output.exists():
                subprocess.run(["aws", "s3", "sync", str(output), destination, "--no-follow-symlinks"], check=True)


if __name__ == "__main__":
    main()
