#!/usr/bin/env python
"""Anyscale entrypoint: download data from S3, run post-training."""

import logging
import os
import subprocess
import sys
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("anyscale_runner")

S3_PREFIX = os.environ.get(
    "S3_DATA_PREFIX",
    "s3://anyscale-production-data-vm-us-east-1-f7164253"
    "/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm",
)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

S3_CHECKPOINT_PREFIX = os.environ.get(
    "S3_CHECKPOINT_PREFIX",
    f"{S3_PREFIX}/checkpoints",
)


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
        subprocess.check_call(
            ["sudo", "apt-get", "install", "-y", "-qq", "ncbi-blast+"]
        )
        log.info("BLAST+ installed")


def main():
    config_name = os.environ.get("CONFIG", "grpo_dense_anyscale")

    # ── Install BLAST+ if plannotate scorer ────────────────────────────────
    if "plannotate" in config_name:
        install_blast()

    # ── Download data ─────────────────────────────────────────────────────
    log.info("Downloading data from S3...")
    s3_download(
        f"{S3_PREFIX}/data/motif_registry_combined.parquet",
        DATA_DIR / "motif_registry_combined.parquet",
    )
    s3_download(
        f"{S3_PREFIX}/data/training_pairs_v4.parquet",
        DATA_DIR / "training_pairs_v4.parquet",
    )
    s3_download(
        f"{S3_PREFIX}/data/plannotate_db.parquet",
        DATA_DIR / "plannotate_db.parquet",
    )
    log.info("Data download complete.")

    # ── Load config ───────────────────────────────────────────────────────
    from post_training.runners.run import load_config, run

    config_name = os.environ.get("CONFIG", "grpo_dense_anyscale")
    config_path = ROOT / "post_training" / "configs" / f"{config_name}.py"
    log.info("Using config: %s", config_path)
    cfg = load_config(config_path)

    # ── Resume from S3 checkpoint ────────────────────────────────────────
    # Pass S3 path directly — ModelActor downloads on the worker node.
    resume_from = os.environ.get("RESUME_FROM_S3")
    if resume_from:
        log.info("Will resume from S3 checkpoint: %s", resume_from)
        cfg.model = resume_from

    # ── Set S3 checkpoint sync (runs on worker via ModelActor) ───────────
    run_name = config_name.replace("_anyscale", "")
    cfg.s3_checkpoint_prefix = f"{S3_CHECKPOINT_PREFIX}/{run_name}"

    # ── Run post-training ─────────────────────────────────────────────────
    run(cfg)


if __name__ == "__main__":
    main()
