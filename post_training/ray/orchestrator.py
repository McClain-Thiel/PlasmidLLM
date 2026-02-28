"""Training loop orchestrator for Ray-based distributed post-training.

Coordinates the GPU policy actor, CPU reward scoring tasks, and algorithm
layer into a coherent training loop with curriculum learning, MLflow
logging, and checkpointing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import ray
import torch

from post_training.ray.config import RayPostTrainingConfig

log = logging.getLogger(__name__)


class Orchestrator:
    """Main training loop driver.

    Coordinates:
    1. Sampling prompt batches
    2. GPU generation (PolicyActor)
    3. Distributed CPU scoring (reward tasks)
    4. Advantage computation (algorithm layer)
    5. GPU training step (PolicyActor)
    6. Curriculum updates, logging, checkpointing
    """

    def __init__(self, config: RayPostTrainingConfig):
        self.config = config
        self._mlflow = None
        self._mlflow_active = False

    def train(self) -> None:
        """Run the full training loop."""
        import sys
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from plasmid_llm.utils import (
            _get_git_commit,
            cycling_batch_iterator,
            load_prompts_from_parquet,
            setup_mlflow,
        )
        from post_training.ray.algorithms import build_algorithm
        from post_training.ray.policy_actor import PolicyActor
        from post_training.ray.reward_tasks import score_batch_distributed
        from post_training.reward import build_category_index, load_motif_lookup

        config = self.config

        # ── Initialize Ray ────────────────────────────────────────────
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            log.info(f"Ray initialized: {ray.cluster_resources()}")

        # ── Load motif registry ───────────────────────────────────────
        log.info(f"Loading motif lookup from {config.motif_lookup}")
        lookup_df = load_motif_lookup(str(config.motif_lookup))
        category_index = build_category_index(lookup_df)
        log.info(
            f"Motif lookup: {len(lookup_df)} entries, "
            f"categories: {list(category_index.keys())}"
        )

        # Put scoring context into Ray object store (zero-copy to workers)
        alpha = config.curriculum_alpha_start
        scoring_context = {
            "lookup_df": lookup_df,
            "category_index": category_index,
            "alpha": alpha,
            "eos_bonus": config.eos_bonus,
            "length_penalty_threshold": config.length_penalty_threshold,
        }
        context_ref = ray.put(scoring_context)

        # ── Load prompts ──────────────────────────────────────────────
        log.info(f"Loading prompts from {config.training_pairs}")
        prompts = load_prompts_from_parquet(config.training_pairs)
        prompt_iter = cycling_batch_iterator(
            prompts, config.generation_batch_size, seed=config.seed
        )

        # ── Create policy actor ───────────────────────────────────────
        log.info("Creating PolicyActor on GPU")
        policy = PolicyActor.remote(
            model_checkpoint=str(config.model_checkpoint),
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            bf16=config.bf16,
            max_completion_length=config.max_completion_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            seed=config.seed,
        )

        # ── Build algorithm ───────────────────────────────────────────
        algo_kwargs = {"kl_coef": config.kl_coef}
        algorithm = build_algorithm(config.algorithm, **algo_kwargs)
        log.info(f"Algorithm: {config.algorithm} (kl_coef={config.kl_coef})")

        # ── MLflow setup ──────────────────────────────────────────────
        self._mlflow_active = setup_mlflow(
            config.mlflow_tracking_uri, config.mlflow_experiment
        )
        if self._mlflow_active:
            import mlflow

            self._mlflow = mlflow
            params = config.to_mlflow_params()
            params["git_commit"] = _get_git_commit()
            mlflow.log_params(params)

        # ── Output directory ──────────────────────────────────────────
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Training loop ─────────────────────────────────────────────
        log.info(f"Starting training: {config.max_steps} steps")
        for step in range(1, config.max_steps + 1):
            step_start = time.time()

            # 1. Sample batch of prompts
            batch_prompts = next(prompt_iter)

            # 2. Generate rollouts on GPU
            rollout = ray.get(policy.generate_rollouts.remote(batch_prompts))

            # 3. Score completions on CPU workers
            rewards_list = score_batch_distributed(
                rollout["prompts"],
                rollout["completion_texts"],
                context_ref,
                reward_fn_name=config.reward_fn_name,
                chunk_size=config.scoring_batch_size,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32)

            # 4. Compute advantages
            advantages = algorithm.compute_advantages(
                rewards, rollout["log_probs"], rollout["ref_log_probs"]
            )

            # 5. Train step on GPU
            train_batch = {
                "full_ids": rollout["full_ids"],
                "prompt_len": rollout["prompt_len"],
                "advantages": advantages,
                "ref_log_probs": rollout["ref_log_probs"],
                "old_log_probs": rollout["log_probs"],
                "algorithm_name": config.algorithm,
                "algorithm_kwargs": algo_kwargs,
            }
            metrics = ray.get(policy.train_step.remote(train_batch))

            # 6. Update curriculum alpha
            progress = min(step / max(config.curriculum_alpha_warmup_steps, 1), 1.0)
            new_alpha = (
                config.curriculum_alpha_start
                + progress * (config.curriculum_alpha_end - config.curriculum_alpha_start)
            )
            if new_alpha != alpha:
                alpha = new_alpha
                scoring_context = {
                    "lookup_df": lookup_df,
                    "category_index": category_index,
                    "alpha": alpha,
                    "eos_bonus": config.eos_bonus,
                    "length_penalty_threshold": config.length_penalty_threshold,
                }
                context_ref = ray.put(scoring_context)

            step_time = time.time() - step_start

            # 7. Logging
            if step % config.logging_steps == 0:
                reward_mean = rewards.mean().item()
                reward_std = rewards.std().item()
                log.info(
                    f"step={step} | alpha={alpha:.3f} | "
                    f"reward={reward_mean:.4f} | reward_std={reward_std:.4f} | "
                    f"loss={metrics['loss']:.6f} | kl={metrics['kl']:.5f} | "
                    f"grad_norm={metrics['grad_norm']:.4f} | "
                    f"lr={metrics['lr']:.2e} | time={step_time:.1f}s"
                )

                if self._mlflow_active:
                    self._mlflow.log_metrics(
                        {
                            "reward_mean": reward_mean,
                            "reward_std": reward_std,
                            "loss": metrics["loss"],
                            "kl": metrics["kl"],
                            "grad_norm": metrics["grad_norm"],
                            "lr": metrics["lr"],
                            "alpha": alpha,
                            "step_time": step_time,
                        },
                        step=step,
                    )

            # 8. Checkpointing
            if step % config.save_steps == 0:
                ckpt_path = str(config.output_dir / f"checkpoint-{step}")
                ray.get(policy.save_checkpoint.remote(ckpt_path))

                if self._mlflow_active:
                    self._mlflow.log_artifacts(
                        ckpt_path,
                        artifact_path=f"checkpoints/checkpoint-{step}",
                    )

        # ── Save final checkpoint ─────────────────────────────────────
        final_path = str(config.output_dir / "final")
        ray.get(policy.save_checkpoint.remote(final_path))
        log.info(f"Training complete. Final model saved to {final_path}")

        if self._mlflow_active:
            self._mlflow.log_artifacts(final_path, artifact_path="checkpoints/final")
