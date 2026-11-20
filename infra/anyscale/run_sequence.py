"""Run a bounded sequence on one cluster, stopping immediately on a failed stage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3


def validate_plan(plan: dict) -> None:
    stages = plan["stages"]
    if not stages or len({stage["name"] for stage in stages}) != len(stages):
        raise ValueError("A sequence needs nonempty stages with unique names")
    for stage in stages:
        if (not isinstance(stage["command"], list) or not stage["command"]
                or not all(isinstance(arg, str) and arg for arg in stage["command"])):
            raise ValueError(f"Stage {stage['name']} requires an argument list")
        if type(stage["timeout_s"]) is not int or stage["timeout_s"] <= 0:
            raise ValueError(f"Stage {stage['name']} requires a positive integer timeout")
        if not all(isinstance(k, str) and isinstance(v, str)
                   for k, v in stage.get("env", {}).items()):
            raise ValueError("Stage environment keys and values must be strings")


def run_stage(stage: dict) -> None:
    env = {**os.environ, **stage.get("env", {})}
    process = subprocess.Popen(stage["command"], env=env, start_new_session=True)
    try:
        code = process.wait(timeout=stage["timeout_s"])
        if code:
            raise subprocess.CalledProcessError(code, stage["command"])
    finally:
        if process.poll() is None:
            print(f"Stopping unfinished stage {stage['name']}", flush=True)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Stage {stage['name']} did not stop after SIGTERM; sending SIGKILL", flush=True)
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_sequence(plan: dict, record: Callable[[str, dict], None]) -> None:
    validate_plan(plan)
    for index, stage in enumerate(plan["stages"]):
        event = dict(index=index, stage=stage["name"],
                     started_at=datetime.now(timezone.utc).isoformat())
        record(f"{index:02d}_started", event)
        print(f"Starting stage {index + 1}/{len(plan['stages'])}: {stage['name']}", flush=True)
        try:
            run_stage(stage)
        except (subprocess.SubprocessError, OSError, KeyboardInterrupt) as exc:
            event.update(state="failed", error=f"{type(exc).__name__}: {exc}",
                         finished_at=datetime.now(timezone.utc).isoformat())
            record(f"{index:02d}_failed", event)
            raise
        event.update(state="succeeded", finished_at=datetime.now(timezone.utc).isoformat())
        record(f"{index:02d}_succeeded", event)
        print(f"Completed stage: {stage['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    uri = urlparse(os.environ["S3_SEQUENCE_EVENTS"])
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        raise ValueError("S3_SEQUENCE_EVENTS must identify a nonempty S3 attempt prefix")
    prefix = uri.path.strip("/")
    client = boto3.client("s3")
    if client.list_objects_v2(Bucket=uri.netloc, Prefix=prefix + "/", MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(f"Sequence attempt already exists: {uri.geturl()}")
    output = Path(plan["event_directory"])
    output.mkdir(parents=True, exist_ok=False)

    def record(name: str, event: dict) -> None:
        payload = (json.dumps(event, indent=2) + "\n").encode()
        with (output / f"{name}.json").open("xb") as local:
            local.write(payload)
        client.put_object(Bucket=uri.netloc, Key=f"{prefix}/{name}.json", Body=payload,
                          IfNoneMatch="*", ContentType="application/json")

    def interrupted(signum: int, frame) -> None:
        raise KeyboardInterrupt(f"Sequence received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    record("plan", plan)
    run_sequence(plan, record)


if __name__ == "__main__":
    main()
