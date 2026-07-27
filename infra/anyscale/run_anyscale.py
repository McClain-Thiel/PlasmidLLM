#!/usr/bin/env python
"""Anyscale entrypoint: download data from S3, run post-training."""

import logging
import os
import subprocess
import sys
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("anyscale_runner")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value.rstrip("/")


def s3_download(s3_path: str, local_path: Path) -> None:
    if local_path.exists():
        log.info("  [skip] %s already exists", local_path.name)
        return
    log.info("  Downloading %s → %s", s3_path, local_path)
    subprocess.check_call(["aws", "s3", "cp", s3_path, str(local_path)])


def install_blast():
    """Install NCBI BLAST+ if not already available."""
    try:
        subprocess.run(["blastn", "-version"], capture_output=True, check=True)
        log.info("BLAST+ already installed")
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.info("Installing BLAST+ via apt...")
        subprocess.check_call(["sudo", "apt-get", "update", "-qq"])
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "-qq", "ncbi-blast+"])
        log.info("BLAST+ installed")


def required_data_paths(cfg) -> list[Path]:
    """Return repository data files referenced by a post-training config."""
    configured = [cfg.prompts_parquet, cfg.eval_parquet]
    configured.extend(cfg.scorer_kwargs.values())

    paths = []
    for value in configured:
        if not value:
            continue
        path = Path(value).resolve()
        if path.parent != DATA_DIR.resolve():
            continue
        paths.append(path)
    return sorted(set(paths))


def main():
    config_name = os.environ.get("CONFIG", "grpo_plannotate_anyscale")
    s3_prefix = required_env("S3_DATA_PREFIX")
    checkpoint_prefix = os.environ.get(
        "S3_CHECKPOINT_PREFIX",
        f"{s3_prefix}/checkpoints",
    ).rstrip("/")

    from post_training.runners.run import load_config, run

    config_path = ROOT / "post_training" / "configs" / f"{config_name}.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Post-training config not found: {config_path}")
    log.info("Using config: %s", config_path)
    cfg = load_config(config_path)

    # ── Install BLAST+ if BLAST-based scorer ──────────────────────────────
    if "plannotate" in config_name:
        install_blast()

    # ── Download data ─────────────────────────────────────────────────────
    log.info("Downloading data from S3...")
    for local_path in required_data_paths(cfg):
        s3_download(f"{s3_prefix}/data/{local_path.name}", local_path)
    log.info("Data download complete.")

    # ── Resume from S3 checkpoint ────────────────────────────────────────
    # Pass S3 path directly — ModelActor downloads on the worker node.
    resume_from = os.environ.get("RESUME_FROM_S3")
    if resume_from:
        log.info("Will resume from S3 checkpoint: %s", resume_from)
        cfg.model = resume_from

    # ── Set S3 checkpoint sync (runs on worker via ModelActor) ───────────
    run_name = config_name.replace("_anyscale", "")
    cfg.s3_checkpoint_prefix = f"{checkpoint_prefix}/{run_name}"

    # ── Run post-training ─────────────────────────────────────────────────
    run(cfg)


if __name__ == "__main__":
    main()
