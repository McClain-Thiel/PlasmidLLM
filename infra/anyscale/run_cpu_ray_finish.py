#!/usr/bin/env python3
"""Parallel multi-node Ray runner for CPU orthogonal annotation and final aggregation.

Reuses any panel whose ``annotation_manifest.json`` and verified
``per_sequence_metrics.parquet`` already exist under configured read-only S3 roots
(e.g., ``annotations_v1``), dispatches remaining panels LPT-ordered across Ray
worker nodes (one actor per 32-vCPU node with 28 pLannotate workers) into an
immutable destination prefix (``annotations_v2``), and publishes the final paired
statistical summary to S3 and W&B.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import wandb
from botocore.exceptions import ClientError

try:
    import ray  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - local test venv without ray
    ray = None

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.neurips_rebuttal.summarize_rigor_evaluation import (  # noqa: E402
    summarize_rigor_evaluation,
)
from infra.anyscale.run_cpu_annotation_shard import (  # noqa: E402
    _check_empty_prefix,
    _download_verified,
    _is_not_found,
    _publish_shard_status,
    annotate_and_score_panel,
    archive_verified_directory,
    parse_s3_uri,
    restore_annotation_tools,
    wait_for_s3_manifest,
)
from plasmid_llm.provenance import file_sha256, s3_readback_sha256  # noqa: E402

FAMILY_LPT_PRIORITY = {
    "reference_control": 0,
    "exp2_v2": 1,
    "cluster_base_s42": 2,
    "exp1_v2": 3,
    "exp1_v1": 4,
    "hf_base": 5,
}


def discover_reusable_panels(
    client,
    reuse_s3_roots: list[str],
    tasks: list[dict[str, Any]],
    annotate_reference_control: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Identify completed panels with SHA256-verified metrics under ``reuse_s3_roots``.

    Panels without an ``annotation_manifest.json`` (including partial uploads from a
    terminated shard) are returned in ``pending_tasks``. Existing manifests whose
    ``per_sequence_metrics.parquet`` checksum does not match raise ``ValueError``.
    """
    candidates: list[dict[str, Any]] = []
    if annotate_reference_control:
        candidates.append({
            "relpath": "reference_control",
            "model_name": "reference_control",
            "model_family": "reference_control",
            "training_seed": 0,
            "decode_seed": 0,
        })
    candidates.extend(tasks)

    reused: dict[str, dict[str, Any]] = {}
    pending_tasks: list[dict[str, Any]] = []

    for task in candidates:
        relpath = str(task["relpath"]).strip("/")
        found_entry: dict[str, Any] | None = None
        for root in reuse_s3_roots:
            bucket, prefix = parse_s3_uri(root)
            manifest_key = f"{prefix}/{relpath}/annotation_manifest.json"
            try:
                body = client.get_object(Bucket=bucket, Key=manifest_key)["Body"]
                try:
                    manifest = json.loads(body.read().decode("utf-8"))
                finally:
                    body.close()
            except ClientError as exc:
                if _is_not_found(exc):
                    continue
                raise

            files = manifest.get("files", {})
            metrics_info = files.get("per_sequence_metrics.parquet")
            if not isinstance(metrics_info, dict) or not metrics_info.get("sha256"):
                raise ValueError(
                    f"Malformed annotation_manifest.json without per_sequence_metrics.parquet sha256: "
                    f"s3://{bucket}/{manifest_key}"
                )
            metrics_key = f"{prefix}/{relpath}/per_sequence_metrics.parquet"
            expected_sha = str(metrics_info["sha256"])
            try:
                actual_sha = s3_readback_sha256(client, bucket, metrics_key)
            except ClientError as exc:
                if _is_not_found(exc):
                    raise ValueError(
                        f"annotation_manifest.json exists but per_sequence_metrics.parquet is missing: "
                        f"s3://{bucket}/{metrics_key}"
                    ) from exc
                raise
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for reusable panel s3://{bucket}/{metrics_key}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            found_entry = {
                "s3_prefix": f"{root.rstrip('/')}/{relpath}",
                "manifest": manifest,
                "source": "reused",
            }
            break

        if found_entry is not None:
            reused[relpath] = found_entry
        else:
            pending_tasks.append(task)

    return reused, pending_tasks


def order_tasks_lpt(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order tasks Longest-Processing-Time-First to minimize tail stragglers."""
    return sorted(
        tasks,
        key=lambda t: (
            FAMILY_LPT_PRIORITY.get(str(t.get("model_family", "")), 9),
            int(t.get("training_seed", 0)),
            int(t.get("decode_seed", 0)),
            str(t.get("relpath", "")),
        ),
    )


def restore_annotation_tools_with_retry(
    client,
    tools_s3_prefix: str,
    manifest_sha256: str,
    *,
    stagger_seconds: float = 0.0,
    max_attempts: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Stagger and retry toolchain restoration while cleaning failed ``/tmp/ann_*`` dirs."""
    if stagger_seconds > 0:
        sleep_fn(stagger_seconds)
    tmp_root = Path("/tmp")
    for attempt in range(1, max_attempts + 1):
        before = set(tmp_root.glob("ann_*")) if tmp_root.exists() else set()
        try:
            return restore_annotation_tools(
                client=client,
                tools_s3_prefix=tools_s3_prefix,
                manifest_sha256=manifest_sha256,
            )
        except Exception:
            after = set(tmp_root.glob("ann_*")) if tmp_root.exists() else set()
            for leaked in after - before:
                shutil.rmtree(leaked, ignore_errors=True)
            if attempt >= max_attempts:
                raise
            sleep_fn(float(5 * attempt))
    raise RuntimeError("Unreachable retry loop termination")


class AnnotationWorkerImpl:
    """Node-pinned worker actor executing pLannotate + Prodigal + AMRFinderPlus panels."""

    def __init__(self, worker_index: int, plan: dict[str, Any]) -> None:
        self.worker_index = int(worker_index)
        self.plan = plan
        self.client: Any = None
        self.panel_dir: Path | None = None
        self.panel_df: pd.DataFrame | None = None
        self.scratch: Path | None = None
        self.env_prefix: Path | None = None
        self.amr_db_dir: Path | None = None
        self.conda_env: dict[str, str] | None = None

    def setup(self) -> dict[str, Any]:
        """Download frozen panel and restore pinned bioinformatics toolchain on this node."""
        t0 = time.monotonic()
        self.client = boto3.client("s3")
        self.panel_dir = Path(
            tempfile.mkdtemp(prefix=f"panel_w{self.worker_index}_", dir="/tmp")
        ).resolve()
        panel_bucket, panel_key = parse_s3_uri(self.plan["panel_s3"])
        local_panel = self.panel_dir / "frozen_panel_v3.parquet"
        _download_verified(
            self.client,
            panel_bucket,
            panel_key,
            local_panel,
            expected_sha256=self.plan["panel_sha256"],
        )
        self.panel_df = pd.read_parquet(local_panel)

        stagger_s = float(self.plan.get("setup_stagger_seconds", 3.0)) * self.worker_index
        self.scratch, self.env_prefix, self.amr_db_dir, self.conda_env = restore_annotation_tools_with_retry(
            client=self.client,
            tools_s3_prefix=self.plan["annotation_tools_s3"],
            manifest_sha256=self.plan["annotation_tools_manifest_sha256"],
            stagger_seconds=stagger_s,
        )
        return {
            "worker_index": self.worker_index,
            "host": socket.gethostname(),
            "setup_s": round(time.monotonic() - t0, 2),
        }

    def annotate_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Annotate, score, and immutably archive one panel under ``write_annotation_s3_root``."""
        if (
            self.client is None
            or self.panel_df is None
            or self.scratch is None
            or self.env_prefix is None
            or self.amr_db_dir is None
            or self.conda_env is None
        ):
            raise RuntimeError("Worker setup() must complete before annotate_task()")

        t0 = time.monotonic()
        relpath = str(task["relpath"]).strip("/")
        write_root = str(self.plan["write_annotation_s3_root"]).rstrip("/")
        task_dir = self.scratch / "work" / relpath.replace("/", "__")
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        task_dir.mkdir(parents=True, exist_ok=True)

        try:
            workers = int(self.plan.get("plannotate_workers", 28))
            amr_threads = int(self.plan.get("amr_threads", 16))
            input_dir = task_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)

            if relpath == "reference_control":
                ref_gen_df = pd.DataFrame({
                    "prompt": self.panel_df["prompt"].astype(str),
                    "sample_id": self.panel_df["sample_id"].astype(str),
                    "cluster_id": self.panel_df["cluster_id"].astype(str),
                    "sequence": self.panel_df["source_sequence"].astype(str),
                    "length": self.panel_df["source_sequence"].astype(str).str.len(),
                    "gc": self.panel_df["source_sequence"].astype(str).map(
                        lambda s: (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1)
                    ),
                    "has_eos": True,
                    "truncated": False,
                })
                gen_parquet = input_dir / "generations.parquet"
                ref_gen_df.to_parquet(gen_parquet, index=False)
                fasta_path = input_dir / "generations.fasta"
                with fasta_path.open("w") as fh:
                    for i, seq in enumerate(ref_gen_df["sequence"]):
                        fh.write(f">gen_{i}\n{seq}\n")
            else:
                gen_s3_prefix = str(task["generation_s3_prefix"]).rstrip("/")
                eval_manifest = wait_for_s3_manifest(
                    self.client,
                    manifest_s3_uri=f"{gen_s3_prefix}/evaluation_manifest.json",
                    workflow_events_s3_uri=task["workflow_events_s3"],
                    stage_index=int(task["stage_index"]),
                    poll_seconds=float(self.plan.get("poll_seconds", 20.0)),
                    max_wait_seconds=float(self.plan.get("max_wait_seconds", 3600.0)),
                )
                gen_bucket, gen_prefix = parse_s3_uri(gen_s3_prefix)
                for fname in ("generations.parquet", "generations.fasta", "generation_summary.json", "config.json"):
                    _download_verified(
                        self.client,
                        gen_bucket,
                        f"{gen_prefix}/{fname}",
                        input_dir / fname,
                        expected_sha256=eval_manifest["files"][fname]["sha256"],
                    )
                gen_parquet = input_dir / "generations.parquet"
                fasta_path = input_dir / "generations.fasta"

            annotate_and_score_panel(
                panel_df=self.panel_df,
                gen_parquet=gen_parquet,
                fasta_path=fasta_path,
                ann_dir=task_dir / "annotation",
                env_prefix=self.env_prefix,
                amr_db_dir=self.amr_db_dir,
                conda_env=self.conda_env,
                model_name=str(task["model_name"]),
                model_family=str(task["model_family"]),
                training_seed=int(task["training_seed"]),
                decode_seed=int(task["decode_seed"]),
                plannotate_workers=workers,
                amr_threads=amr_threads,
            )
            dest_s3 = f"{write_root}/{relpath}"
            manifest = archive_verified_directory(
                self.client,
                task_dir / "annotation",
                dest_s3,
                "annotation_manifest.json",
            )
            return {
                "relpath": relpath,
                "s3_prefix": dest_s3,
                "manifest": manifest,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "worker_index": self.worker_index,
            }
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

    def cleanup(self) -> bool:
        """Best-effort cleanup of local worker scratch directories."""
        if self.scratch is not None and self.scratch.exists():
            shutil.rmtree(self.scratch, ignore_errors=True)
        if self.panel_dir is not None and self.panel_dir.exists():
            shutil.rmtree(self.panel_dir, ignore_errors=True)
        return True


if ray is not None:
    AnnotationWorker = ray.remote(max_restarts=0, max_task_retries=0)(AnnotationWorkerImpl)
else:  # pragma: no cover - local test venv without ray
    AnnotationWorker = None


def _execute_tasks_on_ray(
    *,
    plan: dict[str, Any],
    pending_tasks: list[dict[str, Any]],
    completed_manifests: dict[str, dict[str, Any]],
    total_required: int,
    tracking: Any,
) -> None:
    """Schedule ``pending_tasks`` dynamically across Ray worker actors."""
    if ray is None or AnnotationWorker is None:
        raise RuntimeError("Ray is required to execute CPU annotation tasks across nodes")

    shard_name = str(plan["shard_name"])
    ray.init(address="auto", ignore_reinit_error=True)

    n_actors = min(int(plan.get("num_ray_workers", 7)), len(pending_tasks))
    worker_cpus = int(plan.get("worker_num_cpus", 28))
    actors = [AnnotationWorker.options(num_cpus=worker_cpus).remote(i, plan) for i in range(n_actors)]

    setup_futures: dict[Any, Any] = {actor.setup.remote(): actor for actor in actors}
    task_futures: dict[Any, tuple[Any, dict[str, Any]]] = {}
    idle_actors: collections.deque[Any] = collections.deque()
    task_queue: collections.deque[dict[str, Any]] = collections.deque(pending_tasks)

    ready_workers_count = 0
    newly_done = 0
    boot_started = time.monotonic()
    boot_timeout_s = float(plan.get("worker_boot_timeout_s", 900.0))

    try:
        while task_queue or task_futures:
            while idle_actors and task_queue:
                actor = idle_actors.popleft()
                task = task_queue.popleft()
                fut = actor.annotate_task.remote(task)
                task_futures[fut] = (actor, task)

            if not task_futures and not setup_futures:
                break

            if ready_workers_count == 0:
                elapsed_boot = time.monotonic() - boot_started
                remaining = boot_timeout_s - elapsed_boot
                if remaining <= 0:
                    raise TimeoutError(f"No Ray annotation worker became ready within {boot_timeout_s}s")
                wait_timeout: float | None = min(30.0, remaining)
            else:
                wait_timeout = None

            active = list(setup_futures.keys()) + list(task_futures.keys())
            done, _ = ray.wait(active, num_returns=1, timeout=wait_timeout)
            if not done:
                if ready_workers_count == 0 and (time.monotonic() - boot_started) >= boot_timeout_s:
                    raise TimeoutError(f"No Ray annotation worker became ready within {boot_timeout_s}s")
                continue

            for fut in done:
                if fut in setup_futures:
                    actor = setup_futures.pop(fut)
                    info = ray.get(fut)
                    ready_workers_count += 1
                    print(
                        f"[{shard_name}] Worker {info['worker_index']} ready on {info['host']} "
                        f"in {info['setup_s']:.1f}s ({ready_workers_count}/{n_actors})",
                        flush=True,
                    )
                    idle_actors.append(actor)
                else:
                    actor, _task = task_futures.pop(fut)
                    res = ray.get(fut)
                    relpath = res["relpath"]
                    completed_manifests[relpath] = {
                        "s3_prefix": res["s3_prefix"],
                        "manifest": res["manifest"],
                        "source": "annotations_v2",
                    }
                    newly_done += 1
                    print(
                        f"[{shard_name}] Annotated {newly_done}/{len(pending_tasks)} "
                        f"({len(completed_manifests)}/{total_required}): {relpath} "
                        f"in {res['elapsed_s']:.1f}s (worker {res['worker_index']})",
                        flush=True,
                    )
                    tracking.log({
                        "completed_new_tasks": newly_done,
                        "total_completed_panels": len(completed_manifests),
                        "last_task_elapsed_s": res["elapsed_s"],
                    })
                    idle_actors.append(actor)

        if task_queue:
            raise RuntimeError(f"Unscheduled annotation tasks remain after Ray loop: {len(task_queue)}")
    finally:
        cleanup_refs = [actor.cleanup.remote() for actor in actors]
        ray.wait(cleanup_refs, num_returns=len(cleanup_refs), timeout=15.0)


def run_ray_finish(plan: dict[str, Any]) -> None:
    """Discover reusable panels, annotate remaining panels on Ray, and publish final summary."""
    client = boto3.client("s3")
    shard_name = str(plan["shard_name"])
    write_root = str(plan["write_annotation_s3_root"]).rstrip("/")
    write_bucket, write_prefix = parse_s3_uri(write_root)

    # Pre-flight immutability checks before spending cluster compute.
    shard_key = f"{write_prefix}/_shards/{shard_name}.json"
    try:
        client.head_object(Bucket=write_bucket, Key=shard_key)
        raise FileExistsError(f"Shard status already exists on S3: s3://{write_bucket}/{shard_key}")
    except ClientError as exc:
        if not _is_not_found(exc):
            raise
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

    reuse_roots = [str(r).rstrip("/") for r in plan.get("reuse_annotation_s3_roots", [])]
    completed_manifests, raw_pending = discover_reusable_panels(
        client=client,
        reuse_s3_roots=reuse_roots,
        tasks=plan["tasks"],
        annotate_reference_control=bool(plan.get("annotate_reference_control", True)),
    )
    pending_tasks = order_tasks_lpt(raw_pending)

    for task in pending_tasks:
        _check_empty_prefix(client, f"{write_root}/{str(task['relpath']).strip('/')}")

    required_relpaths = [m.rsplit("/", 1)[0] for m in plan["all_required_annotation_manifests"]]
    total_required = len(required_relpaths)
    print(
        f"[{shard_name}] Reusable panels verified: {len(completed_manifests)}/{total_required}; "
        f"pending panels: {len(pending_tasks)}",
        flush=True,
    )

    with wandb.init(
        project=plan.get("wandb_project", "PlasmidLLM"),
        name=f"rigor_20261116_{shard_name}",
        tags=["plasmidrl", "evaluation", "cpu-annotation", "ray-finish", shard_name],
        config={
            "shard_name": shard_name,
            "reused_panels": len(completed_manifests),
            "pending_panels": len(pending_tasks),
            "num_ray_workers": int(plan.get("num_ray_workers", 7)),
            "plannotate_workers": int(plan.get("plannotate_workers", 28)),
        },
    ) as tracking:
        try:
            tracking.log({
                "reused_panels": len(completed_manifests),
                "initial_pending_panels": len(pending_tasks),
            })
            if pending_tasks:
                _execute_tasks_on_ray(
                    plan=plan,
                    pending_tasks=pending_tasks,
                    completed_manifests=completed_manifests,
                    total_required=total_required,
                    tracking=tracking,
                )

            _publish_shard_status(
                client,
                write_root,
                shard_name,
                {
                    "shard_name": shard_name,
                    "state": "succeeded",
                    "reused_panels": sum(1 for v in completed_manifests.values() if v["source"] == "reused"),
                    "newly_annotated_panels": sum(
                        1 for v in completed_manifests.values() if v["source"] != "reused"
                    ),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            try:
                _publish_shard_status(
                    client,
                    write_root,
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

    missing = [rp for rp in required_relpaths if rp not in completed_manifests]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required annotated panels: {missing[:5]}")

    agg_dir = work_root / "aggregation"
    dl_dir = agg_dir / "metrics"
    dl_dir.mkdir(parents=True, exist_ok=True)
    ref_frame: pd.DataFrame | None = None
    model_frames: list[pd.DataFrame] = []
    provenance_panels: dict[str, dict[str, str]] = {}

    for relpath in required_relpaths:
        entry = completed_manifests[relpath]
        entry_bucket, entry_prefix = parse_s3_uri(entry["s3_prefix"])
        metrics_key = f"{entry_prefix}/per_sequence_metrics.parquet"
        expected_sha = str(entry["manifest"]["files"]["per_sequence_metrics.parquet"]["sha256"])
        local_metrics = dl_dir / f"{relpath.replace('/', '_')}.parquet"
        _download_verified(client, entry_bucket, metrics_key, local_metrics, expected_sha256=expected_sha)
        provenance_panels[relpath] = {
            "s3_prefix": entry["s3_prefix"],
            "source": entry["source"],
            "per_sequence_metrics_sha256": expected_sha,
        }
        frame = pd.read_parquet(local_metrics)
        if relpath == "reference_control":
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
    provenance_payload = {
        "reused_count": sum(1 for v in provenance_panels.values() if v["source"] == "reused"),
        "newly_annotated_count": sum(1 for v in provenance_panels.values() if v["source"] != "reused"),
        "panels": provenance_panels,
    }
    (results_dir / "panel_provenance.json").write_text(json.dumps(provenance_payload, indent=2) + "\n")

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
        config={
            "n_panels": len(required_relpaths),
            "reused_panels": provenance_payload["reused_count"],
            "newly_annotated_panels": provenance_payload["newly_annotated_count"],
            "results_s3_destination": plan["results_s3_destination"],
        },
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
    run_ray_finish(json.loads(args.plan.read_text()))


if __name__ == "__main__":
    main()
