#!/usr/bin/env python3
"""Run one CPU orthogonal-annotation shard (plus optional final summary aggregation).

Pipelined design:
1. Restores the archived ``annotation_tools_v5`` micromamba environment, pLannotate
   databases, and NCBI AMRFinderPlus database under a short ``/tmp/ann_*`` prefix
   and verifies every extracted database file against ``validation.json`` SHA256s.
2. Optionally annotates ``reference_control`` (the frozen source sequences) using the
   exact same ``annotate.py --circular-prodigal`` + ``amrfinder --plus`` path.
3. For each assigned ``(model, decode_seed)``, polls S3 for ``evaluation_manifest.json``
   while aborting immediately on upstream GPU stage failures, runs annotation and
   per-sequence scoring, and publishes readback-verified artifacts + ``annotation_manifest.json``.
4. When ``run_final_aggregation`` is set, waits for all 45 panels (failing fast if
   any peer CPU shard failed) and publishes the paired statistical summary to S3 & W&B.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import pandas as pd
import wandb
from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.neurips_rebuttal.common import load_annotations  # noqa: E402
from experiments.neurips_rebuttal.summarize_rigor_evaluation import (  # noqa: E402
    detect_prodigal_peptide_tags,
    parse_dustmasker_intervals,
    score_single_panel_dataframe,
    summarize_rigor_evaluation,
)
from plasmid_llm.provenance import file_sha256, s3_readback_sha256  # noqa: E402

AMRFINDER_HEADER = (
    "Protein id\tElement symbol\tElement name\tScope\tType\tSubtype\tClass\t"
    "Subclass\tMethod\tTarget length\tReference sequence length\t"
    "% Coverage of reference\t% Identity to reference\tAlignment length\t"
    "Closest reference accession\tClosest reference name\tHMM accession\tHMM description\n"
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.rstrip("/"))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a nonempty S3 URI, got: {uri}")
    return parsed.netloc, parsed.path.strip("/")


def _is_not_found(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def clean_conda_env(prefix: Path) -> dict[str, str]:
    """Strip Ray/Anaconda env overrides and configure pinned conda runtime variables."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"}
    }
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(prefix)
    env["PYTHONNOUSERSITE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    return env


def safe_extract_tar(
    tar_path: Path,
    dest_dir: Path,
    *,
    strip_prefix: str = "",
    only_subpath: str = "",
) -> list[Path]:
    """Safely extract regular files/dirs from a tar.gz without symlinks or path escape."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not (member.isdir() or member.isreg()):
                raise ValueError(f"Unsupported tar entry type for {member.name}")
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe tar entry path: {member.name}")
            clean_name = member_path.as_posix()
            while clean_name.startswith("./"):
                clean_name = clean_name.removeprefix("./")
            if only_subpath and not (clean_name == only_subpath.rstrip("/") or clean_name.startswith(only_subpath)):
                continue
            if strip_prefix:
                if clean_name == strip_prefix.rstrip("/"):
                    continue
                if not clean_name.startswith(strip_prefix):
                    raise ValueError(f"Unexpected tar entry prefix: {member.name}")
                rel = clean_name[len(strip_prefix):]
            else:
                rel = clean_name
            if not rel:
                continue
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"Unsafe tar entry path: {member.name}")
            target = (dest_resolved / rel_path).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"Tar path escapes destination directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    raise ValueError(f"Unable to extract tar member: {member.name}")
                with src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(target, member.mode & 0o777 or 0o644)
                extracted.append(target)
    return extracted


def verify_directory_hashes(root: Path, expected_files: dict[str, dict[str, Any]], *, prefix_filter: str = "") -> None:
    """Verify SHA256 and byte counts against ``validation.json`` metadata."""
    for rel_name, info in expected_files.items():
        if prefix_filter and not rel_name.startswith(prefix_filter):
            continue
        path = (root / rel_name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing required archived file: {rel_name}")
        if path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch for restored file: {rel_name}")


def _download_verified(client, bucket: str, key: str, dest: Path, expected_sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    if expected_sha256 and file_sha256(dest) != expected_sha256:
        raise ValueError(f"Downloaded object checksum mismatch: s3://{bucket}/{key}")
    return dest


def restore_annotation_tools(
    client,
    tools_s3_prefix: str,
    manifest_sha256: str,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Restore micromamba env, pLannotate DBs, and AMRFinderPlus DB under /tmp/ann_*."""
    bucket, prefix = parse_s3_uri(tools_s3_prefix)
    scratch = Path(tempfile.mkdtemp(prefix="ann_", dir="/tmp")).resolve()
    env_prefix = scratch / "env"
    if len(str(env_prefix)) >= 80:
        raise ValueError(f"Short /tmp prefix expected, got {len(str(env_prefix))} chars: {env_prefix}")

    downloads = scratch / "downloads"
    manifest_path = _download_verified(
        client, bucket, f"{prefix}/archive_manifest.json",
        downloads / "archive_manifest.json", expected_sha256=manifest_sha256,
    )
    archive_manifest = json.loads(manifest_path.read_text())["files"]

    needed = (
        "validation.json",
        "micromamba_release.json",
        "linux-64.sha256.explicit.txt",
        "plannotate_databases.tar.gz",
        "amrfinder_database.tar.gz",
    )
    for fname in needed:
        if fname not in archive_manifest:
            raise ValueError(f"{fname} missing from {tools_s3_prefix}/archive_manifest.json")
        _download_verified(
            client, bucket, f"{prefix}/{fname}",
            downloads / fname, expected_sha256=archive_manifest[fname]["sha256"],
        )

    validation = json.loads((downloads / "validation.json").read_text())
    if validation.get("passed") is not True:
        raise ValueError("Archived annotation toolchain failed validation")

    release = json.loads((downloads / "micromamba_release.json").read_text())
    selected = next(asset for asset in release["assets"] if asset["name"] == "micromamba-linux-64")
    tool = scratch / "micromamba"
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(selected["url"], timeout=120) as resp, tool.open("wb") as out:
                shutil.copyfileobj(resp, out)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    if file_sha256(tool) != selected["digest"].removeprefix("sha256:"):
        raise ValueError("Official micromamba binary checksum mismatch")
    tool.chmod(0o700)

    install_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"}
    }
    install_env["MAMBA_ROOT_PREFIX"] = str(scratch / "mamba")
    install_env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [
            str(tool), "create", "--no-rc", "--yes", "--prefix", str(env_prefix),
            "--file", str(downloads / "linux-64.sha256.explicit.txt"),
        ],
        env=install_env, check=True, timeout=600,
    )

    conda_env = clean_conda_env(env_prefix)
    pkg_probe = subprocess.run(
        [str(env_prefix / "bin/python"), "-c", "import pathlib, plannotate; print(pathlib.Path(plannotate.__file__).resolve().parent)"],
        env=conda_env, check=True, capture_output=True, text=True, timeout=60,
    )
    plannotate_pkg = Path(pkg_probe.stdout.strip().splitlines()[-1]).resolve(strict=True)
    safe_extract_tar(
        downloads / "plannotate_databases.tar.gz",
        plannotate_pkg,
        strip_prefix="plannotate/",
        only_subpath="plannotate/data/BLAST_dbs/",
    )
    verify_directory_hashes(plannotate_pkg, validation["plannotate_db_files"], prefix_filter="data/BLAST_dbs/")

    amr_root = scratch / "amrfinder_db"
    safe_extract_tar(downloads / "amrfinder_database.tar.gz", amr_root)
    amr_db_dir = (amr_root / "database").resolve(strict=True)
    verify_directory_hashes(amr_db_dir, validation["database_files"])

    subprocess.run(
        [
            str(env_prefix / "bin/python"), "-X", "faulthandler", "-c",
            "from plannotate import resources; from plannotate.annotate import annotate; "
            "assert resources.databases_exist(), 'pLannotate DBs missing'; "
            "annotate('ACGT' * 100, is_detailed=True, linear=False)",
        ],
        env=conda_env, check=True, timeout=180,
    )
    subprocess.run(["amrfinder", "--version"], env=conda_env, check=True, timeout=30)
    smoke_faa = scratch / "smoke_proteins.faa"
    smoke_faa.write_text(">smoke\nM" + "ACDEFGHIKLMNPQRSTVWY" * 5 + "\n")
    subprocess.run(
        [
            "amrfinder", "--database", str(amr_db_dir),
            "--protein", str(smoke_faa), "--plus", "--threads", "2",
            "--output", str(scratch / "smoke_amr.tsv"),
        ],
        env=conda_env, check=True, timeout=120,
    )
    return scratch, env_prefix, amr_db_dir, conda_env


def wait_for_s3_manifest(
    client,
    manifest_s3_uri: str,
    workflow_events_s3_uri: str | None = None,
    stage_index: int | None = None,
    required_files: tuple[str, ...] = ("generations.parquet", "generations.fasta", "generation_summary.json", "config.json"),
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 10800.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll S3 until target manifest JSON exists, aborting early if upstream GPU stage failed."""
    bucket, key = parse_s3_uri(manifest_s3_uri)
    events_loc = parse_s3_uri(workflow_events_s3_uri) if workflow_events_s3_uri else None

    started = monotonic_fn()
    while True:
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            try:
                manifest = json.loads(body.read().decode("utf-8"))
            finally:
                body.close()
            files = manifest.get("files", {})
            missing = [fname for fname in required_files if fname not in files]
            if missing:
                raise ValueError(f"Manifest {manifest_s3_uri} lacks required files: {missing}")
            return manifest
        except ClientError as exc:
            if not _is_not_found(exc):
                raise

        if events_loc is not None and stage_index is not None:
            ev_bucket, ev_prefix = events_loc
            for idx in range(stage_index + 1):
                fail_key = f"{ev_prefix}/{idx:02d}_failed.json"
                try:
                    client.head_object(Bucket=ev_bucket, Key=fail_key)
                    raise RuntimeError(
                        f"Upstream GPU workflow stage {idx:02d} failed ({ev_prefix}); aborting CPU wait"
                    )
                except ClientError as exc:
                    if not _is_not_found(exc):
                        raise

        elapsed = monotonic_fn() - started
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"Timed out after {elapsed:.1f}s waiting for {manifest_s3_uri}")
        sleep_fn(min(poll_seconds, max_wait_seconds - elapsed))


def wait_for_peer_annotations(
    client,
    annotation_s3_root: str,
    required_manifest_relpaths: list[str],
    peer_shard_names: list[str],
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 10800.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, dict[str, Any]]:
    """Wait until all required panel annotation manifests are present, checking peer shards for failure."""
    bucket, prefix = parse_s3_uri(annotation_s3_root)
    manifests: dict[str, dict[str, Any]] = {}
    pending = list(required_manifest_relpaths)
    started = monotonic_fn()

    while pending:
        for peer in peer_shard_names:
            shard_key = f"{prefix}/_shards/{peer}.json"
            try:
                body = client.get_object(Bucket=bucket, Key=shard_key)["Body"]
                try:
                    status = json.loads(body.read().decode("utf-8"))
                finally:
                    body.close()
                if status.get("state") == "failed":
                    raise RuntimeError(f"Peer CPU shard {peer} failed: {status.get('error')}")
            except ClientError as exc:
                if not _is_not_found(exc):
                    raise

        still_pending: list[str] = []
        for relpath in pending:
            obj_key = f"{prefix}/{relpath.lstrip('/')}"
            try:
                body = client.get_object(Bucket=bucket, Key=obj_key)["Body"]
                try:
                    manifests[relpath] = json.loads(body.read().decode("utf-8"))
                finally:
                    body.close()
            except ClientError as exc:
                if not _is_not_found(exc):
                    raise
                still_pending.append(relpath)
        pending = still_pending
        if not pending:
            return manifests

        elapsed = monotonic_fn() - started
        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"Timed out after {elapsed:.1f}s waiting for {len(pending)} annotation manifests; first={pending[:3]}"
            )
        sleep_fn(min(poll_seconds, max_wait_seconds - elapsed))
    return manifests


def archive_verified_directory(client, source_dir: Path, destination_s3: str, manifest_name: str) -> dict[str, Any]:
    """Upload all files under ``source_dir`` immutably with SHA256 readback verification."""
    bucket, prefix = parse_s3_uri(destination_s3)
    if client.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(f"Destination S3 prefix already populated: {destination_s3}")
    files_meta: dict[str, dict[str, Any]] = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(source_dir))
        digest = file_sha256(path)
        key = f"{prefix}/{rel}"
        with path.open("rb") as fh:
            client.put_object(
                Bucket=bucket, Key=key, Body=fh, IfNoneMatch="*",
                Metadata={"sha256": digest},
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            )
        if s3_readback_sha256(client, bucket, key) != digest:
            raise ValueError(f"S3 readback checksum mismatch: s3://{bucket}/{key}")
        files_meta[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    if not files_meta:
        raise ValueError(f"Refusing to publish empty directory to {destination_s3}")
    manifest = {"files": files_meta, "published_at": datetime.now(timezone.utc).isoformat()}
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{manifest_name}",
        Body=(json.dumps(manifest, indent=2) + "\n").encode(),
        IfNoneMatch="*",
        ContentType="application/json",
    )
    return manifest


def annotate_and_score_panel(
    *,
    panel_df: pd.DataFrame,
    gen_parquet: Path,
    fasta_path: Path,
    ann_dir: Path,
    env_prefix: Path,
    amr_db_dir: Path,
    conda_env: dict[str, str],
    model_name: str,
    model_family: str,
    training_seed: int,
    decode_seed: int,
    plannotate_workers: int,
    amr_threads: int,
) -> Path:
    """Run pLannotate + circular Prodigal + dustmasker + AMRFinderPlus + orthogonal scoring."""
    ann_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(env_prefix / "bin/python"),
            str(REPO / "eval/scripts/annotate.py"),
            "--input", str(fasta_path),
            "--output-dir", str(ann_dir),
            "--workers", str(plannotate_workers),
            "--circular-prodigal",
        ],
        env=conda_env,
        cwd=str(REPO),
        check=True,
    )

    proteins_faa = ann_dir / "orfs/prodigal_proteins.faa"
    amr_tsv = ann_dir / "amrfinder.tsv"
    if not proteins_faa.exists():
        raise FileNotFoundError(f"Missing Prodigal output: {proteins_faa}")
    if proteins_faa.stat().st_size == 0:
        amr_tsv.write_text(AMRFINDER_HEADER)
    else:
        subprocess.run(
            [
                "amrfinder",
                "--database", str(amr_db_dir),
                "--protein", str(proteins_faa),
                "--plus",
                "--threads", str(amr_threads),
                "--output", str(amr_tsv),
            ],
            env=conda_env,
            check=True,
        )

    orth_all_path = ann_dir / "orthogonal_all.parquet"
    orth_intact_path = ann_dir / "orthogonal_intact.parquet"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "experiments/neurips_rebuttal/orthogonal.py"),
            "--amr", str(amr_tsv),
            "--generations", str(gen_parquet),
            "--label", f"{model_name}_s{decode_seed}_all",
            "--output", str(orth_all_path),
        ],
        cwd=str(REPO),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO / "experiments/neurips_rebuttal/orthogonal.py"),
            "--amr", str(amr_tsv),
            "--generations", str(gen_parquet),
            "--label", f"{model_name}_s{decode_seed}_intact",
            "--intact-only",
            "--proteins", str(proteins_faa),
            "--output", str(orth_intact_path),
        ],
        cwd=str(REPO),
        check=True,
    )

    gen_df = pd.read_parquet(gen_parquet)
    plannotate_ann = load_annotations(ann_dir / "plannotate_results.json")
    orth_all_df = pd.read_parquet(orth_all_path)
    orth_intact_df = pd.read_parquet(orth_intact_path)
    peptide_tags = detect_prodigal_peptide_tags(proteins_faa)
    lengths = {
        f"gen_{i}": int(row["length"]) if "length" in gen_df.columns else len(str(row["sequence"]))
        for i, row in gen_df.reset_index(drop=True).iterrows()
    }
    low_complexity = parse_dustmasker_intervals(ann_dir / "dustmasker/dustmasker.intervals", lengths)

    metrics_df = score_single_panel_dataframe(
        panel_df=panel_df,
        generations_df=gen_df,
        plannotate_annotations=plannotate_ann,
        orthogonal_all_df=orth_all_df,
        orthogonal_intact_df=orth_intact_df,
        peptide_tags=peptide_tags,
        low_complexity=low_complexity,
        model_name=model_name,
        model_family=model_family,
        training_seed=training_seed,
        decode_seed=decode_seed,
    )
    metrics_path = ann_dir / "per_sequence_metrics.parquet"
    with metrics_path.open("xb") as fh:
        metrics_df.to_parquet(fh, index=False)
    return metrics_path


def _publish_shard_status(client, annotation_s3_root: str, shard_name: str, payload: dict[str, Any]) -> None:
    bucket, prefix = parse_s3_uri(annotation_s3_root)
    key = f"{prefix}/_shards/{shard_name}.json"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, indent=2) + "\n").encode(),
        IfNoneMatch="*",
        ContentType="application/json",
    )


def _check_empty_prefix(client, s3_uri: str) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    if client.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(f"Output S3 prefix already exists: {s3_uri}")


def run_shard(plan: dict[str, Any]) -> None:
    client = boto3.client("s3")
    shard_name = plan["shard_name"]
    ann_root = plan["annotation_s3_root"].rstrip("/")
    ann_bucket, ann_prefix = parse_s3_uri(ann_root)

    # Pre-flight immutability checks before spending compute.
    shard_key = f"{ann_prefix}/_shards/{shard_name}.json"
    try:
        client.head_object(Bucket=ann_bucket, Key=shard_key)
        raise FileExistsError(f"Shard status already exists on S3: s3://{ann_bucket}/{shard_key}")
    except ClientError as exc:
        if not _is_not_found(exc):
            raise
    if plan.get("annotate_reference_control"):
        _check_empty_prefix(client, f"{ann_root}/reference_control")
    for task in plan["tasks"]:
        _check_empty_prefix(client, f"{ann_root}/{task['relpath']}")
    if plan.get("run_final_aggregation"):
        _check_empty_prefix(client, plan["results_s3_destination"])

    manifest_path = REPO / plan["panel_manifest_local_path"]
    if file_sha256(manifest_path) != plan["panel_manifest_sha256"]:
        raise ValueError(f"Panel manifest checksum mismatch: {manifest_path}")
    panel_manifest = json.loads(manifest_path.read_text())

    work_root = Path("output") / shard_name
    work_root.mkdir(parents=True, exist_ok=False)
    panel_bucket, panel_key = parse_s3_uri(plan["panel_s3"])
    local_panel = work_root / "frozen_panel_v3.parquet"
    _download_verified(client, panel_bucket, panel_key, local_panel, expected_sha256=plan["panel_sha256"])
    panel_df = pd.read_parquet(local_panel)

    scratch: Path | None = None
    with wandb.init(
        project=plan.get("wandb_project", "PlasmidLLM"),
        name=f"rigor_20261116_{shard_name}",
        tags=["plasmidrl", "evaluation", "cpu-annotation", shard_name],
        config={"shard_name": shard_name, "n_tasks": len(plan["tasks"]),
                "annotate_reference_control": bool(plan.get("annotate_reference_control")),
                "run_final_aggregation": bool(plan.get("run_final_aggregation"))},
    ) as tracking:
        try:
            scratch, env_prefix, amr_db_dir, conda_env = restore_annotation_tools(
                client=client,
                tools_s3_prefix=plan["annotation_tools_s3"],
                manifest_sha256=plan["annotation_tools_manifest_sha256"],
            )

            workers = int(plan.get("plannotate_workers", 16))
            amr_threads = int(plan.get("amr_threads", 8))

            if plan.get("annotate_reference_control"):
                ref_dir = work_root / "reference_control"
                ref_dir.mkdir(parents=True, exist_ok=False)
                ref_gen_df = pd.DataFrame({
                    "prompt": panel_df["prompt"].astype(str),
                    "sample_id": panel_df["sample_id"].astype(str),
                    "cluster_id": panel_df["cluster_id"].astype(str),
                    "sequence": panel_df["source_sequence"].astype(str),
                    "length": panel_df["source_sequence"].astype(str).str.len(),
                    "gc": panel_df["source_sequence"].astype(str).map(
                        lambda s: (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1)
                    ),
                    "has_eos": True,
                    "truncated": False,
                })
                ref_gen_parquet = ref_dir / "generations.parquet"
                ref_gen_df.to_parquet(ref_gen_parquet, index=False)
                ref_fasta = ref_dir / "generations.fasta"
                with ref_fasta.open("w") as fh:
                    for i, seq in enumerate(ref_gen_df["sequence"]):
                        fh.write(f">gen_{i}\n{seq}\n")
                annotate_and_score_panel(
                    panel_df=panel_df,
                    gen_parquet=ref_gen_parquet,
                    fasta_path=ref_fasta,
                    ann_dir=ref_dir / "annotation",
                    env_prefix=env_prefix,
                    amr_db_dir=amr_db_dir,
                    conda_env=conda_env,
                    model_name="reference_control",
                    model_family="reference_control",
                    training_seed=0,
                    decode_seed=0,
                    plannotate_workers=workers,
                    amr_threads=amr_threads,
                )
                archive_verified_directory(
                    client, ref_dir / "annotation", f"{ann_root}/reference_control", "annotation_manifest.json"
                )
                tracking.log({"reference_control_done": 1})

            for idx, task in enumerate(plan["tasks"], start=1):
                gen_s3_prefix = task["generation_s3_prefix"].rstrip("/")
                eval_manifest = wait_for_s3_manifest(
                    client,
                    manifest_s3_uri=f"{gen_s3_prefix}/evaluation_manifest.json",
                    workflow_events_s3_uri=task["workflow_events_s3"],
                    stage_index=int(task["stage_index"]),
                    poll_seconds=float(plan.get("poll_seconds", 30.0)),
                    max_wait_seconds=float(plan.get("max_wait_seconds", 10800.0)),
                )
                gen_bucket, gen_prefix = parse_s3_uri(gen_s3_prefix)
                task_dir = work_root / task["relpath"]
                input_dir = task_dir / "inputs"
                input_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("generations.parquet", "generations.fasta", "generation_summary.json", "config.json"):
                    _download_verified(
                        client,
                        gen_bucket,
                        f"{gen_prefix}/{fname}",
                        input_dir / fname,
                        expected_sha256=eval_manifest["files"][fname]["sha256"],
                    )
                annotate_and_score_panel(
                    panel_df=panel_df,
                    gen_parquet=input_dir / "generations.parquet",
                    fasta_path=input_dir / "generations.fasta",
                    ann_dir=task_dir / "annotation",
                    env_prefix=env_prefix,
                    amr_db_dir=amr_db_dir,
                    conda_env=conda_env,
                    model_name=task["model_name"],
                    model_family=task["model_family"],
                    training_seed=int(task["training_seed"]),
                    decode_seed=int(task["decode_seed"]),
                    plannotate_workers=workers,
                    amr_threads=amr_threads,
                )
                archive_verified_directory(
                    client,
                    task_dir / "annotation",
                    f"{ann_root}/{task['relpath']}",
                    "annotation_manifest.json",
                )
                tracking.log({"completed_tasks": idx})
                print(f"[{shard_name}] Annotated {idx}/{len(plan['tasks'])}: {task['relpath']}", flush=True)

            _publish_shard_status(
                client,
                ann_root,
                shard_name,
                {"shard_name": shard_name, "state": "succeeded", "completed_at": datetime.now(timezone.utc).isoformat()},
            )
        except Exception as exc:
            try:
                _publish_shard_status(
                    client,
                    ann_root,
                    shard_name,
                    {
                        "shard_name": shard_name,
                        "state": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                pass
            raise
        finally:
            if scratch is not None and scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)

    if plan.get("run_final_aggregation"):
        manifests = wait_for_peer_annotations(
            client,
            annotation_s3_root=ann_root,
            required_manifest_relpaths=plan["all_required_annotation_manifests"],
            peer_shard_names=plan.get("peer_shard_names", []),
            poll_seconds=float(plan.get("poll_seconds", 30.0)),
            max_wait_seconds=float(plan.get("max_wait_seconds", 10800.0)),
        )
        agg_dir = work_root / "aggregation"
        dl_dir = agg_dir / "metrics"
        dl_dir.mkdir(parents=True, exist_ok=True)
        ref_frame: pd.DataFrame | None = None
        model_frames: list[pd.DataFrame] = []
        for rel_manifest, manifest in manifests.items():
            rel_parent = rel_manifest.rsplit("/", 1)[0]
            metrics_key = f"{ann_prefix}/{rel_parent}/per_sequence_metrics.parquet"
            local_metrics = dl_dir / f"{rel_parent.replace('/', '_')}.parquet"
            expected_sha = manifest["files"]["per_sequence_metrics.parquet"]["sha256"]
            _download_verified(client, ann_bucket, metrics_key, local_metrics, expected_sha256=expected_sha)
            frame = pd.read_parquet(local_metrics)
            if rel_parent == "reference_control":
                ref_frame = frame
            else:
                model_frames.append(frame)
        if ref_frame is None:
            raise ValueError("reference_control metrics frame missing")

        results_dir = agg_dir / "results"
        summary = summarize_rigor_evaluation(
            panel_df=panel_df,
            panel_manifest=panel_manifest,
            ref_df=ref_frame,
            model_frames=model_frames,
            decode_seeds=plan.get("decode_seeds", [42, 43, 44, 45]),
            iters=int(plan.get("bootstrap_iters", 10000)),
            output_dir=results_dir,
        )
        archive_verified_directory(
            client,
            results_dir,
            plan["results_s3_destination"],
            "results_manifest.json",
        )
        with wandb.init(
            project=plan.get("wandb_project", "PlasmidLLM"),
            name="rigor_20261116_final_eval_summary_v1",
            tags=["plasmidrl", "evaluation", "final-summary"],
            config={"n_panels": len(manifests), "results_s3_destination": plan["results_s3_destination"]},
        ) as summary_run:
            log_metrics: dict[str, float] = {}
            for key_prefix, label in (
                ("exp1_v2_vs_exp1_v1:main", "exp1_v2_vs_v1"),
                ("exp2_v2_vs_cluster_base_s42:main", "exp2_v2_vs_base"),
            ):
                bucket_metrics = summary["paired_comparisons"].get(key_prefix, {})
                for m in ("useful_05", "viable", "faithfulness", "extra_ori_amr", "amr_orthogonal_intact_recall"):
                    if m in bucket_metrics:
                        log_metrics[f"{label}/{m}_delta"] = bucket_metrics[m]["delta"]
                        if bucket_metrics[m].get("holm_p_value") is not None:
                            log_metrics[f"{label}/{m}_holm_p"] = bucket_metrics[m]["holm_p_value"]
            if log_metrics:
                summary_run.log(log_metrics)
            artifact = wandb.Artifact(f"{summary_run.id}-rigor-results", type="evaluation-summary")
            for path in sorted(results_dir.rglob("*")):
                if path.is_file() and path.suffix != ".parquet":
                    artifact.add_file(str(path), name=str(path.relative_to(results_dir)))
            summary_run.log_artifact(artifact).wait()
        print(f"[{shard_name}] Final summary archived: {plan['results_s3_destination']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    run_shard(json.loads(args.plan.read_text()))


if __name__ == "__main__":
    main()
