"""Example: GRPO training with ModelActor + GRPOAlgorithm.

Shows how the Algorithm drives the ModelActor through its API.
The actor has no idea it's doing GRPO — it just executes primitives.

Usage:
    python -m post_training.runners.run_grpo
    python -m post_training.runners.run_grpo --model gpt2 --steps 10
    python -m post_training.runners.run_grpo --model McClain/PlasmidLM-kmer6-MoE --steps 100
    python -m post_training.runners.run_grpo --wandb --wandb-project plasmid-rl
"""

from __future__ import annotations

import argparse
import logging
import sys

import ray
import torch

from post_training.algorithms import GRPOAlgorithm, Scorer
from post_training.common.model import ModelActor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Scorer — swap this for AlignmentScorer, MotifScorer, LLM judge, etc.
# ══════════════════════════════════════════════════════════════════════════


class SubstringScorer:
    """Toy scorer: reward = fraction of target substrings found.

    Satisfies the Scorer protocol expected by Algorithm.step().
    Replace with your real scorer for actual training.
    """

    def __init__(self, targets: list[str]):
        self.targets = targets

    def score(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        rewards = []
        for comp in completions:
            hits = sum(1 for t in self.targets if t in comp.lower())
            rewards.append(hits / max(len(self.targets), 1))
        return torch.tensor(rewards)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="GRPO training example")
    parser.add_argument("--model", default="gpt2", help="HF model id or local path")
    parser.add_argument("--num-actors", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints/grpo")

    # wandb args
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", default="post-training", help="wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="wandb run name (auto-generated if omitted)")
    parser.add_argument("--wandb-entity", default=None, help="wandb team/entity")
    args = parser.parse_args()

    # ── wandb init ────────────────────────────────────────────────────────
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={
                    "algorithm": "grpo",
                    "model": args.model,
                    "num_actors": args.num_actors,
                    "steps": args.steps,
                    "lr": args.lr,
                    "kl_coef": args.kl_coef,
                    "cliprange": args.cliprange,
                    "num_generations": args.num_generations,
                    "micro_batch_size": args.micro_batch_size,
                    "max_new_tokens": args.max_new_tokens,
                },
            )
            log.info("wandb run: %s/%s", wandb.run.project, wandb.run.name)
        except ImportError:
            log.warning("--wandb flag set but wandb is not installed; continuing without it")

    ray.init(ignore_reinit_error=True)

    # ── Actors: pure GPU workers ──────────────────────────────────────────
    actors = [
        ModelActor.remote(
            args.model,
            learning_rate=args.lr,
            max_steps=args.steps,
            bf16=torch.cuda.is_available(),
        )
        for _ in range(args.num_actors)
    ]

    if args.num_actors > 1:
        GRPOAlgorithm.sync_actors(actors)

    # ── Algorithm: drives the actors ──────────────────────────────────────
    algo = GRPOAlgorithm(
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        num_generations=args.num_generations,
        micro_batch_size=args.micro_batch_size,
    )

    scorer = SubstringScorer(targets=["the", "is", "and"])

    # ── Prompts ───────────────────────────────────────────────────────────
    prompts = [
        "Once upon a time",
        "The experiment showed",
        "In a world where",
        "The data revealed",
    ]

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(
        "Starting GRPO: model=%s, actors=%d, steps=%d, G=%d",
        args.model, args.num_actors, args.steps, args.num_generations,
    )

    for step in range(1, args.steps + 1):
        metrics = algo.step(actors, prompts, scorer)

        log.info(
            "step=%d  loss=%.4f  reward=%.3f  reward_best=%.3f  "
            "kl=%.4f  adv_std=%.3f  grad_norm=%.4f  lr=%.2e  "
            "[gen=%.1fs score=%.1fs train=%.1fs total=%.1fs]",
            step,
            metrics["loss"],
            metrics["mean_reward"],
            metrics.get("reward_best", 0),
            metrics["kl"],
            metrics.get("adv_std", 0),
            metrics.get("grad_norm", 0),
            metrics.get("lr", 0),
            metrics.get("time_generation", 0),
            metrics.get("time_scoring", 0),
            metrics.get("time_train", 0),
            metrics.get("time_total", 0),
        )

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            ckpt = f"{args.checkpoint_dir}/step_{step}"
            ray.get(actors[0].save_checkpoint.remote(ckpt))
            log.info("Checkpoint → %s", ckpt)

    # ── Final checkpoint ──────────────────────────────────────────────────
    final = f"{args.checkpoint_dir}/final"
    ray.get(actors[0].save_checkpoint.remote(final))
    log.info("Training complete. Final model → %s", final)

    # ── Cleanup ───────────────────────────────────────────────────────────
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
