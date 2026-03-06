"""GRPO training — CLI wrapper around the config-driven runner.

For quick experiments with CLI args. For reproducible runs, prefer a
config file with ``python -m post_training.runners.run <config.py>``.

Usage:
    python -m post_training.runners.run_grpo
    python -m post_training.runners.run_grpo --model gpt2 --steps 10
    python -m post_training.runners.run_grpo --wandb --wandb-project plasmid-rl
"""

from __future__ import annotations

import argparse
import logging
import sys

from post_training.config import PostTrainingConfig
from post_training.runners.run import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="GRPO training (CLI)")
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
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", default="post-training")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    import torch

    cfg = PostTrainingConfig(
        model=args.model,
        bf16=torch.cuda.is_available(),
        num_actors=args.num_actors,
        algorithm="grpo",
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        num_generations=args.num_generations,
        micro_batch_size=args.micro_batch_size,
        max_new_tokens=args.max_new_tokens,
        scorer="substring",
        scorer_kwargs={"targets": ["the", "is", "and"]},
        prompts_list=[
            "Once upon a time",
            "The experiment showed",
            "In a world where",
            "The data revealed",
        ],
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
    )

    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
