"""Download a verified partition, then run pretraining on one CUDA device."""

from __future__ import annotations

import os
import base64
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3

from plasmid_llm.provenance import file_sha256, s3_readback_sha256
from plasmid_llm.utils import load_config


def export_final_model(client, final: Path, destination: str) -> dict:
    """Publish the completion manifest only after every object passes readback."""
    uri = urlparse(destination)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("Expected a nonempty S3 export prefix")
    prefix = uri.path.strip("/")
    required = {"model.safetensors", "config.json", "vocab.json",
                "tokenizer_config.json", "training_report.json"}
    files = {str(p.relative_to(final)): p for p in final.rglob("*") if p.is_file()}
    if not required <= files.keys() or any(p.stat().st_size == 0 for p in files.values()):
        raise ValueError("Final checkpoint is missing required files or contains empty files")
    report = json.loads((final / "training_report.json").read_text())
    if report["selection"] != "last_step" or report["global_step"] != report["requested_steps"]:
        raise ValueError("Final checkpoint does not match the fixed-step training plan")
    manifest = dict(destination=destination, training=report, files={})
    for name, path in sorted(files.items()):
        digest = file_sha256(path)
        key = f"{prefix}/{name}"
        with path.open("rb") as source:
            client.put_object(Bucket=uri.netloc, Key=key, Body=source, IfNoneMatch="*",
                              ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
                              Metadata={"sha256": digest})
        if s3_readback_sha256(client, uri.netloc, key) != digest:
            raise ValueError(f"S3 readback checksum mismatch: {key}")
        manifest["files"][name] = dict(sha256=digest, bytes=path.stat().st_size)
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    client.put_object(Bucket=uri.netloc, Key=f"{prefix}/export_manifest.json",
                      Body=payload, IfNoneMatch="*", ContentType="application/json")
    with (final.parent / "export_manifest.json").open("xb") as local:
        local.write(payload)
    return manifest


def main() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Cluster pretraining requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cluster pretraining requires CUDA bfloat16 support")
    prefix = os.environ["S3_SPLIT_PREFIX"].rstrip("/")
    destination = os.environ["S3_PRETRAIN_EXPORT_PREFIX"].rstrip("/")
    uri = urlparse(destination)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("S3_PRETRAIN_EXPORT_PREFIX must be a nonempty S3 run prefix")
    client = boto3.client("s3")
    if client.list_objects_v2(Bucket=uri.netloc, Prefix=uri.path.lstrip("/") + "/",
                             MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(f"Pretraining export already exists: {destination}")
    split = Path("data/cluster_v2").resolve()
    for name in ("train.parquet", "validation.parquet", "test.parquet", "manifest.json", "audit/report.json"):
        path = split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(["aws", "s3", "cp", f"{prefix}/{name}", str(path)], check=True)
    os.environ["PLASMID_SPLIT_DIR"] = str(split)
    os.environ["PLASMID_SPLIT_AUDIT"] = str(split / "audit/report.json")
    config_path = Path("pretraining/configs/cluster_disjoint_v2.py")
    config = load_config(config_path)
    subprocess.run([sys.executable, "-m", "pretraining.train",
                    str(config_path)], check=True)
    export_final_model(client, config.output_dir / "final", destination)
    print(f"Final model exported and readback verified: {destination}", flush=True)


if __name__ == "__main__":
    if os.environ.get("PLASMID_PRETRAIN_ON_RAY") == "1":
        import ray

        ray.init(address="auto")
        try:
            # A stage's environment differs from the enclosing Anyscale job's.
            worker_env = {key: value for key, value in os.environ.items()
                          if key.startswith("PLASMID_") or key in {
                              "S3_SPLIT_PREFIX", "S3_PRETRAIN_EXPORT_PREFIX"}}
            task = ray.remote(num_gpus=1, runtime_env={"env_vars": worker_env})(main)
            ray.get(task.remote())
        finally:
            ray.shutdown()
    else:
        main()
