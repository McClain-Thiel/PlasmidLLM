"""Shared helpers for PlasmidLLM training scripts."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def load_config(config_path: Path):
    """Load a config dataclass from a Python file exporting ``config``."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise ValueError(f"Config file must define 'config' variable: {config_path}")
    return module.config


def load_env_file(project_root: Path | None = None) -> None:
    """Load .env file into environment (setdefault, won't overwrite)."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())
    log.info(f"Loaded env vars from {env_file}")


def setup_mlflow(tracking_uri: str | None, experiment_name: str) -> bool:
    """Configure MLflow tracking. Returns True if MLflow is active."""
    if not tracking_uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.set_experiment(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
            exp = mlflow.set_experiment(experiment_id=exp_id)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        exp_id = exp.experiment_id if exp else "unknown"
        log.info(f"MLflow: {tracking_uri} / {experiment_name} (id={exp_id})")
        return True
    except Exception as e:
        log.warning(f"MLflow setup failed: {e} — continuing without MLflow")
        return False


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file (first 16 chars for logging)."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_prompts_from_parquet(
    parquet_path: str | Path, filter_hard_tokens: bool = True
) -> List[str]:
    """Load prompt strings from training pairs parquet.

    Returns list of prompt strings with ``<SEP>`` appended.
    """
    parquet_path = str(parquet_path)
    meta = pq.ParquetFile(parquet_path).schema.names
    need_cols = []
    for c in ["prompt", "token_prompt", "reward_motifs", "has_hard_tokens"]:
        if c in meta:
            need_cols.append(c)
    if "prompt" not in need_cols and "token_prompt" not in need_cols:
        if "full_text" in meta:
            need_cols.append("full_text")

    log.info(f"Reading columns {need_cols} from {parquet_path}")
    table = pq.read_table(parquet_path, columns=need_cols)
    col_names = table.column_names

    # Filter to prompts with hard tokens
    if "has_hard_tokens" in col_names and filter_hard_tokens:
        has_hard = table.column("has_hard_tokens").to_pylist()
        indices = [i for i, h in enumerate(has_hard) if h]
        table = table.take(indices)
    elif "reward_motifs" in col_names and filter_hard_tokens:
        motifs = table.column("reward_motifs").to_pylist()
        indices = [i for i, m in enumerate(motifs) if m]
        table = table.take(indices)

    # Extract prompt column
    if "prompt" in col_names:
        prompts = table.column("prompt").to_pylist()
    elif "token_prompt" in col_names:
        prompts = table.column("token_prompt").to_pylist()
    elif "full_text" in col_names:
        full_texts = table.column("full_text").to_pylist()
        prompts = []
        for text in full_texts:
            match = re.search(r"(.*?)<SEP>", text)
            prompts.append(match.group(1) if match else text)
    else:
        raise ValueError(f"No valid prompt column found. Available: {col_names}")

    prompts = [p + "<SEP>" for p in prompts]
    log.info(f"Loaded {len(prompts)} prompts (filtered={filter_hard_tokens})")
    return prompts


def cycling_batch_iterator(
    items: List, batch_size: int, seed: int = 42
) -> Iterator[List]:
    """Yield batches from items, reshuffling each epoch.

    Cycles indefinitely through the data, yielding ``batch_size`` items at a
    time. Reshuffles the order at the start of each epoch.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(items))
    pos = 0
    rng.shuffle(indices)
    while True:
        if pos + batch_size > len(indices):
            # Wrap around: finish partial, reshuffle, continue
            batch_indices = list(indices[pos:])
            rng.shuffle(indices)
            remaining = batch_size - len(batch_indices)
            batch_indices.extend(indices[:remaining])
            pos = remaining
        else:
            batch_indices = indices[pos : pos + batch_size]
            pos += batch_size
        yield [items[i] for i in batch_indices]
