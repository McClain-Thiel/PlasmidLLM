"""Evaluate one fixed checkpoint on a hashed panel, archiving every decoding seed."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3

from plasmid_llm.provenance import file_sha256, s3_readback_sha256


def archive_results(client, output: Path, destination: str) -> dict:
    uri = urlparse(destination)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("Evaluation output requires a nonempty S3 prefix")
    prefix = uri.path.strip("/")
    if client.list_objects_v2(Bucket=uri.netloc, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(destination)
    manifest = {"files": {}}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(output))
        digest = file_sha256(path)
        key = f"{prefix}/{name}"
        with path.open("rb") as source:
            client.put_object(Bucket=uri.netloc, Key=key, Body=source, IfNoneMatch="*",
                              Metadata={"sha256": digest},
                              ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode())
        if s3_readback_sha256(client, uri.netloc, key) != digest:
            raise ValueError(f"Evaluation archive readback failed: {key}")
        manifest["files"][name] = dict(sha256=digest, bytes=path.stat().st_size)
    if not manifest["files"]:
        raise ValueError("Cannot publish an empty evaluation archive")
    client.put_object(Bucket=uri.netloc, Key=f"{prefix}/evaluation_manifest.json",
                      Body=(json.dumps(manifest, indent=2) + "\n").encode(),
                      IfNoneMatch="*", ContentType="application/json")
    return manifest


def verify_checkpoint(model: Path, expected_steps: int, require_export: bool) -> None:
    import torch

    required = {"config.json", "model.safetensors", "tokenizer_config.json", "vocab.json"}
    for name in required:
        if not (model / name).is_file() or not (model / name).stat().st_size:
            raise ValueError(f"Checkpoint lacks {name}")
    if require_export:
        manifest = json.loads((model / "export_manifest.json").read_text())
        report = manifest["training"]
        if (report["global_step"] != expected_steps or report["requested_steps"] != expected_steps
                or report["selection"] != "last_step"):
            raise ValueError("Pretrained checkpoint is not the requested final step")
        if not required <= manifest["files"].keys():
            raise ValueError("Export manifest omits required checkpoint files")
        for name, info in manifest["files"].items():
            path = (model / name).resolve()
            if not path.is_relative_to(model.resolve()):
                raise ValueError(f"Invalid checkpoint manifest path: {name}")
            if file_sha256(path) != info["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
    else:
        state = torch.load(model / "training_state.pt", map_location="cpu", weights_only=True)
        if state["step"] != expected_steps:
            raise ValueError(f"GRPO checkpoint step {state['step']} differs from {expected_steps}")


def evaluate(plan: dict) -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires exactly one visible CUDA GPU")
    work = Path("output") / plan["name"]
    work.mkdir(parents=True, exist_ok=False)
    (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    panel = work / "panel.parquet"
    subprocess.run(["aws", "s3", "cp", plan["panel"], str(panel)], check=True)
    if file_sha256(panel) != plan["panel_sha256"]:
        raise ValueError("Frozen evaluation panel checksum mismatch")
    model = plan["model"]
    revision = []
    if model.startswith("s3://"):
        local_model = work / "model"
        local_model.mkdir()
        subprocess.run(["aws", "s3", "sync", model, str(local_model)], check=True)
        verify_checkpoint(local_model, plan["expected_steps"], plan.get("require_export", False))
        model = str(local_model)
    else:
        revision = ["--revision", plan["revision"]]
    client = boto3.client("s3")
    for seed in plan["decode_seeds"]:
        output = work / f"decode_s{seed}"
        command = [sys.executable, "eval/scripts/generate.py", "--model", model, *revision,
                   "--prompts", str(panel), "--prompts-verbatim", "--batch-size", str(plan["batch_size"]),
                   "--max-new-tokens", str(plan["max_new_tokens"]),
                   "--temperature", str(plan["temperature"]), "--top-p", str(plan["top_p"]),
                   "--top-k", str(plan["top_k"]), "--seed", str(seed), "--device", "cuda",
                   "--run-name", f"{plan['name']}_decode_s{seed}", "--output-dir", str(output)]
        subprocess.run(command, check=True)
        (output / "evaluation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        archive_results(client, output, f"{plan['destination'].rstrip('/')}/decode_s{seed}")
        print(f"Evaluation seed {seed} archived and readback verified", flush=True)


def main() -> None:
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    destination = urlparse(plan["destination"])
    if destination.scheme != "s3" or not destination.netloc or not destination.path.strip("/"):
        raise ValueError("Evaluation output requires a nonempty S3 prefix")
    if boto3.client("s3").list_objects_v2(
            Bucket=destination.netloc, Prefix=destination.path.strip("/") + "/", MaxKeys=1,
    ).get("KeyCount", 0):
        raise FileExistsError(f"Evaluation attempt already exists: {plan['destination']}")
    ray.init(address="auto")
    try:
        ray.get(ray.remote(num_gpus=1)(evaluate).remote(plan))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
