from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, List

import torch

log = logging.getLogger(__name__)

_wandb = None


def _get_wandb():
    """Lazy-import wandb. Returns the module if installed and a run is active, else None."""
    global _wandb
    if _wandb is False:
        return None
    if _wandb is None:
        try:
            import wandb
            _wandb = wandb
        except ImportError:
            _wandb = False
            return None
    if _wandb.run is None:
        return None
    return _wandb


def wandb_log(metrics: dict, step: int | None = None, prefix: str = "") -> None:
    """Log metrics to wandb if a run is active. No-op otherwise.

    Args:
        metrics: Flat dict of metric names to scalar values.
        step: Global step (optional, wandb auto-increments if omitted).
        prefix: Prepend to every key (e.g. "grpo/" → "grpo/loss").
    """
    wb = _get_wandb()
    if wb is None:
        return
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    wb.log(metrics, step=step)


@contextmanager
def timer(name: str | None = None):
    """Context manager that yields a callable returning elapsed seconds.

    Usage::

        with timer("generation") as t:
            do_stuff()
        elapsed = t()
    """
    t0 = time.monotonic()
    elapsed = [0.0]

    def get():
        return elapsed[0]

    try:
        yield get
    finally:
        elapsed[0] = time.monotonic() - t0
        if name:
            log.debug("%s took %.2fs", name, elapsed[0])


def average_gradients(all_grads: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average gradients from N actors."""
    avg = {}
    keys = all_grads[0].keys()
    for key in keys:
        stacked = torch.stack([g[key] for g in all_grads if key in g])
        avg[key] = stacked.mean(dim=0)
    return avg
