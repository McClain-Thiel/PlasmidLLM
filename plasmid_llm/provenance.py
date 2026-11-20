"""Capture exact inputs and working source for a training attempt."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from dataclasses import asdict
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s3_readback_sha256(client, bucket: str, key: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 by iterating S3 StreamingBody directly rather than via context manager."""
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    checked = hashlib.sha256()
    try:
        for chunk in body.iter_chunks(chunk_size=chunk_size):
            checked.update(chunk)
    finally:
        body.close()
    return checked.hexdigest()


def write_run_provenance(config, output: Path, inputs: list[Path]) -> tuple[Path, Path]:
    """Record dirty working source too; a commit alone does not identify a run."""
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    source_path = output / "run_source.tar.gz"
    if manifest_path.exists() or source_path.exists():
        raise FileExistsError(f"Training attempt already exists: {output}")
    files = []
    for directory in ("plasmid_llm", "pretraining", "post_training", "infra/anyscale"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".sh"})
    files.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    snapshot_path = root / "source_snapshot_manifest.json"
    if snapshot_path.exists():
        files.append(snapshot_path)
    manifest = dict(
        config=asdict(config),
        inputs={str(p.resolve()): file_sha256(p) for p in sorted(set(inputs))},
        source={str(p.relative_to(root)): file_sha256(p) for p in sorted(files)},
    )
    if (root / ".git").exists():
        manifest["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        manifest["git_status"] = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    elif snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        manifest["git_commit"] = snapshot["git_commit"]
        manifest["git_status"] = snapshot["git_status"]
    with source_path.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)))
    manifest["source_archive_sha256"] = file_sha256(source_path)
    with manifest_path.open("x") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
    return manifest_path, source_path
