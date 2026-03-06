"""Config-driven post-training runner.

Loads a PostTrainingConfig from a Python config file and runs the
specified (algorithm, model, scorer) combination.

Usage:
    python -m post_training.runners.run post_training/configs/smoke_test.py
    python -m post_training.runners.run post_training/configs/grpo_motif.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import ray
import torch

from post_training.algorithms import build_algorithm
from post_training.common.model import ModelActor
from post_training.config import PostTrainingConfig
from post_training.scorers import build_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Config loading (same pattern as plasmid_llm.utils.load_config)
# ══════════════════════════════════════════════════════════════════════════


def load_config(config_path: str | Path) -> PostTrainingConfig:
    """Load a PostTrainingConfig from a Python file exporting ``config``."""
    config_path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise ValueError(f"Config file must define 'config' variable: {config_path}")
    cfg = module.config
    if not isinstance(cfg, PostTrainingConfig):
        raise TypeError(
            f"Expected PostTrainingConfig, got {type(cfg).__name__} in {config_path}"
        )
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# Scorer builders (maps scorer name → constructor)
# ══════════════════════════════════════════════════════════════════════════


class SubstringScorer:
    """Toy scorer for smoke tests."""

    def __init__(self, targets: list[str]):
        self.targets = targets

    def score(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        rewards = []
        for comp in completions:
            hits = sum(1 for t in self.targets if t in comp.lower())
            rewards.append(hits / max(len(self.targets), 1))
        return torch.tensor(rewards)


def _build_scorer(cfg: PostTrainingConfig):
    """Build a scorer from config. Falls back to registry for real scorers."""
    if cfg.scorer == "substring":
        return SubstringScorer(**cfg.scorer_kwargs)
    return build_scorer(cfg.scorer, **cfg.scorer_kwargs)


# ══════════════════════════════════════════════════════════════════════════
# Prompt loading
# ══════════════════════════════════════════════════════════════════════════


def _load_prompts(cfg: PostTrainingConfig) -> list[str]:
    """Load prompts from config — either inline list or parquet file."""
    if cfg.prompts_list:
        log.info("Using %d inline prompts", len(cfg.prompts_list))
        return cfg.prompts_list

    if cfg.prompts_parquet:
        from plasmid_llm.utils import load_prompts_from_parquet
        prompts = load_prompts_from_parquet(
            cfg.prompts_parquet,
            filter_hard_tokens=cfg.filter_hard_tokens,
        )
        log.info("Loaded %d prompts from %s", len(prompts), cfg.prompts_parquet)
        return prompts

    raise ValueError("Config must set either prompts_list or prompts_parquet")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def run(cfg: PostTrainingConfig) -> None:
    """Execute a full post-training run from a config."""

    # ── wandb ──────────────────────────────────────────────────────────────
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                entity=cfg.wandb_entity,
                config=cfg.to_wandb_config(),
            )
            log.info("wandb run: %s/%s", wandb.run.project, wandb.run.name)
        except ImportError:
            log.warning("wandb_project set but wandb is not installed; continuing without it")

    ray.init(ignore_reinit_error=True)

    # ── Actors ─────────────────────────────────────────────────────────────
    actors = [
        ModelActor.remote(cfg.model, **cfg.actor_kwargs())
        for _ in range(cfg.num_actors)
    ]
    if cfg.num_actors > 1:
        from post_training.algorithms.base import Algorithm
        Algorithm.sync_actors(actors)

    # ── Algorithm + scorer ─────────────────────────────────────────────────
    algo = build_algorithm(cfg.algorithm, **cfg.algorithm_kwargs())
    scorer = _build_scorer(cfg)
    all_prompts = _load_prompts(cfg)

    log.info(
        "Starting %s: model=%s, actors=%d, steps=%d, scorer=%s",
        cfg.algorithm.upper(), cfg.model, cfg.num_actors, cfg.steps, cfg.scorer,
    )

    # ── Prompt batching ────────────────────────────────────────────────────
    if len(all_prompts) <= cfg.prompt_batch_size:
        prompt_iter = iter(lambda: all_prompts, None)
    else:
        from plasmid_llm.utils import cycling_batch_iterator
        prompt_iter = cycling_batch_iterator(
            all_prompts, cfg.prompt_batch_size, seed=cfg.seed,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    for step in range(1, cfg.steps + 1):
        batch = next(prompt_iter)
        metrics = algo.step(actors, batch, scorer)

        log.info(
            "step=%d  loss=%.4f  reward=%.3f  kl=%.4f  "
            "grad_norm=%.4f  lr=%.2e  [total=%.1fs]",
            step,
            metrics["loss"],
            metrics["mean_reward"],
            metrics["kl"],
            metrics.get("grad_norm", 0),
            metrics.get("lr", 0),
            metrics.get("time_total", 0),
        )

        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            ckpt = f"{cfg.checkpoint_dir}/step_{step}"
            ray.get(actors[0].save_checkpoint.remote(ckpt))
            log.info("Checkpoint → %s", ckpt)

    # ── Final checkpoint ───────────────────────────────────────────────────
    final = f"{cfg.checkpoint_dir}/final"
    ray.get(actors[0].save_checkpoint.remote(final))
    log.info("Training complete. Final model → %s", final)

    # ── Cleanup ────────────────────────────────────────────────────────────
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    ray.shutdown()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m post_training.runners.run <config.py>", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    run(cfg)


if __name__ == "__main__":
    main()
