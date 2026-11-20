"""Wait for verified cluster-disjoint base export before starting dependent Exp2 GRPO."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

REQUIRED_EXPORT_FILES = {
    "model.safetensors",
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "training_report.json",
}


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.rstrip("/"))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a nonempty S3 URI, got: {uri}")
    return parsed.netloc, parsed.path.strip("/")


def _is_not_found(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def validate_export_manifest(manifest: dict, expected_steps: int) -> dict:
    training = manifest.get("training", {})
    if training.get("selection") != "last_step":
        raise ValueError(f"Unexpected pretraining selection: {training.get('selection')!r}")
    if training.get("global_step") != expected_steps:
        raise ValueError(
            f"Export global_step {training.get('global_step')!r} != expected {expected_steps}"
        )
    files = manifest.get("files", {})
    missing = sorted(REQUIRED_EXPORT_FILES - set(files))
    if missing:
        raise ValueError(f"Export manifest missing required files: {missing}")
    for name, info in files.items():
        if not info.get("sha256") or not isinstance(info.get("bytes"), int) or info["bytes"] <= 0:
            raise ValueError(f"Invalid file metadata in export_manifest.json for {name}")
    return manifest


def wait_for_export(
    client,
    export_prefix: str,
    pretrain_events_prefix: str,
    expected_steps: int = 100000,
    poll_seconds: float = 60.0,
    max_wait_seconds: float = 28800.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict:
    """Poll S3 until verified export_manifest.json exists or stage 00 reports failure."""
    export_bucket, export_key_prefix = _parse_s3_uri(export_prefix)
    events_bucket, events_key_prefix = _parse_s3_uri(pretrain_events_prefix)
    manifest_key = f"{export_key_prefix}/export_manifest.json"
    failed_key = f"{events_key_prefix}/00_failed.json"

    start = monotonic_fn()
    while True:
        try:
            body = client.get_object(Bucket=export_bucket, Key=manifest_key)["Body"]
            try:
                manifest = json.loads(body.read().decode("utf-8"))
            finally:
                body.close()
            validated = validate_export_manifest(manifest, expected_steps=expected_steps)
            print(
                f"Verified cluster base export at s3://{export_bucket}/{export_key_prefix} "
                f"(step={expected_steps}, files={len(validated['files'])})",
                flush=True,
            )
            return validated
        except ClientError as exc:
            if not _is_not_found(exc):
                raise

        try:
            client.head_object(Bucket=events_bucket, Key=failed_key)
            raise RuntimeError(
                "Cluster pretraining stage 00 failed; aborting dependent Exp2 stage without waiting"
            )
        except ClientError as exc:
            if not _is_not_found(exc):
                raise

        elapsed = monotonic_fn() - start
        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"Timed out after {elapsed:.1f}s waiting for s3://{export_bucket}/{manifest_key}"
            )
        remaining = max_wait_seconds - elapsed
        sleep_fn(min(poll_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-prefix", required=True)
    parser.add_argument("--pretrain-events-prefix", required=True)
    parser.add_argument("--expected-steps", type=int, default=100000)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=28800.0)
    args = parser.parse_args()
    client = boto3.client("s3")
    wait_for_export(
        client=client,
        export_prefix=args.export_prefix,
        pretrain_events_prefix=args.pretrain_events_prefix,
        expected_steps=args.expected_steps,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )


if __name__ == "__main__":
    main()
