"""Training loop orchestrator for Ray-based distributed post-training.

Coordinates GPU policy actor(s), CPU reward scoring tasks, and algorithm
layer into a coherent training loop with W&B/MLflow logging and checkpointing.

Multi-GPU: N PolicyActors generate rollouts in parallel, ALL train on the
merged batch (identical update keeps weights in sync — no weight transfer).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import ray
import torch

from post_training.ray.config import RayPostTrainingConfig

log = logging.getLogger(__name__)


def _merge_rollouts(rollouts: list[dict]) -> dict:
    """Merge rollouts from multiple GPU actors into a single batch.

    Handles different prompt_len across actors by left-padding shorter batches
    so that all sequences share a uniform prompt_len = max across actors.
    """
    merged = {"prompts": [], "completion_texts": []}
    for r in rollouts:
        merged["prompts"].extend(r["prompts"])
        merged["completion_texts"].extend(r["completion_texts"])

    pad_id = rollouts[0]["pad_token_id"]
    max_prompt_len = max(r["prompt_len"] for r in rollouts)

    # ── Align full_ids: left-pad so all have prompt_len = max_prompt_len ──
    aligned = []
    for r in rollouts:
        ids = r["full_ids"]
        extra = max_prompt_len - r["prompt_len"]
        if extra > 0:
            pad = torch.full((ids.shape[0], extra), pad_id, dtype=ids.dtype)
            ids = torch.cat([pad, ids], dim=1)
        aligned.append(ids)

    # Right-pad to max total length
    max_total = max(a.shape[1] for a in aligned)
    padded = []
    for a in aligned:
        if a.shape[1] < max_total:
            pad = torch.full(
                (a.shape[0], max_total - a.shape[1]), pad_id, dtype=a.dtype
            )
            padded.append(torch.cat([a, pad], dim=1))
        else:
            padded.append(a)
    merged["full_ids"] = torch.cat(padded, dim=0)
    merged["prompt_len"] = max_prompt_len

    # ── completion_ids: right-pad and concat ──
    comps = [r["completion_ids"] for r in rollouts]
    max_comp = max(c.shape[1] for c in comps)
    padded_c = []
    for c in comps:
        if c.shape[1] < max_comp:
            pad = torch.full(
                (c.shape[0], max_comp - c.shape[1]), pad_id, dtype=c.dtype
            )
            padded_c.append(torch.cat([c, pad], dim=1))
        else:
            padded_c.append(c)
    merged["completion_ids"] = torch.cat(padded_c, dim=0)

    # ── Scalar-per-sequence tensors: just concat ──
    merged["log_probs"] = torch.cat([r["log_probs"] for r in rollouts])
    merged["ref_log_probs"] = torch.cat([r["ref_log_probs"] for r in rollouts])

    # ── Per-token tensors: right-pad to max completion length, then concat ──
    for key in ["per_token_log_probs", "ref_per_token_log_probs"]:
        tensors = [r[key] for r in rollouts]
        max_len = max(t.shape[1] for t in tensors)
        padded_t = []
        for t in tensors:
            if t.shape[1] < max_len:
                pad = torch.zeros(
                    (t.shape[0], max_len - t.shape[1]), dtype=t.dtype
                )
                padded_t.append(torch.cat([t, pad], dim=1))
            else:
                padded_t.append(t)
        merged[key] = torch.cat(padded_t, dim=0)

    # completion_mask: right-pad with False
    masks = [r["completion_mask"] for r in rollouts]
    max_mask_len = max(m.shape[1] for m in masks)
    padded_m = []
    for m in masks:
        if m.shape[1] < max_mask_len:
            pad = torch.zeros(
                (m.shape[0], max_mask_len - m.shape[1]), dtype=m.dtype
            )
            padded_m.append(torch.cat([m, pad], dim=1))
        else:
            padded_m.append(m)
    merged["completion_mask"] = torch.cat(padded_m, dim=0)

    return merged


class Orchestrator:
    """Main training loop driver.

    Coordinates:
    1. Sampling prompt batches
    2. Parallel GPU generation (PolicyActor(s))
    3. Distributed CPU scoring (reward tasks)
    4. Advantage computation (algorithm layer)
    5. Parallel GPU training (all actors train on same batch → stay in sync)
    6. Logging, checkpointing
    """

    def __init__(self, config: RayPostTrainingConfig):
        self.config = config
        self._mlflow = None
        self._mlflow_active = False
        self._wandb = None
        self._wandb_active = False

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
        from post_training.reward import load_motif_lookup

        config = self.config
        G = config.num_generations_per_prompt
        num_gpus = config.num_policy_gpus

        # ── Initialize Ray ────────────────────────────────────────────
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": project_root
                        + ":"
                        + str(Path(project_root) / "src")
                    }
                },
            )
            log.info(f"Ray initialized: {ray.cluster_resources()}")

        # ── Load motif registry ───────────────────────────────────────
        log.info(f"Loading motif lookup from {config.motif_lookup}")
        lookup_df = load_motif_lookup(str(config.motif_lookup))
        log.info(f"Motif lookup: {len(lookup_df)} entries")

        scoring_context = {
            "lookup_df": lookup_df,
            "eos_bonus": config.eos_bonus,
        }
        context_ref = ray.put(scoring_context)

        # ── Load prompts ──────────────────────────────────────────────
        log.info(f"Loading prompts from {config.training_pairs}")
        prompts = load_prompts_from_parquet(config.training_pairs)
        prompt_iter = cycling_batch_iterator(
            prompts, config.generation_batch_size, seed=config.seed
        )

        # ── Create policy actor(s) ───────────────────────────────────
        actor_kwargs = dict(
            model_checkpoint=str(config.model_checkpoint),
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_grad_norm=config.max_grad_norm,
            bf16=config.bf16,
            max_completion_length=config.max_completion_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            seed=config.seed,
        )

        log.info(f"Creating {num_gpus} PolicyActor(s) on GPU")
        actors = [PolicyActor.remote(**actor_kwargs) for _ in range(num_gpus)]
        primary = actors[0]

        # ── Build algorithm ───────────────────────────────────────────
        algo_kwargs = {"kl_coef": config.kl_coef}
        if config.algorithm == "grpo":
            algo_kwargs["cliprange"] = config.cliprange
            algo_kwargs["num_generations"] = G
        algorithm = build_algorithm(config.algorithm, **algo_kwargs)
        log.info(
            f"Algorithm: {config.algorithm} (kl_coef={config.kl_coef}, "
            f"G={G}, batch={config.generation_batch_size}, gpus={num_gpus})"
        )

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

        # ── W&B setup ────────────────────────────────────────────────
        if config.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_mlflow_params(),
                )
                wandb.define_metric("*", step_metric="train_step")
                self._wandb = wandb
                self._wandb_active = True
                log.info(
                    f"W&B initialized: project={config.wandb_project}, "
                    f"run={wandb.run.name}"
                )
            except Exception as e:
                log.warning(f"W&B init failed: {e}")
                self._wandb_active = False

        # ── Output directory ──────────────────────────────────────────
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Training loop ─────────────────────────────────────────────
        total_seqs_per_step = config.generation_batch_size * G
        log.info(
            f"Starting training: {config.max_steps} steps, "
            f"{total_seqs_per_step} sequences/step "
            f"({config.generation_batch_size} prompts x {G} generations, "
            f"{num_gpus} GPU(s), {total_seqs_per_step // num_gpus} seq/GPU)"
        )
        for step in range(1, config.max_steps + 1):
            step_start = time.time()

            # 1. Sample unique prompts, repeat each G times for GRPO
            unique_prompts = next(prompt_iter)
            if G > 1:
                batch_prompts = [p for p in unique_prompts for _ in range(G)]
            else:
                batch_prompts = unique_prompts

            # 2. Generate rollouts on GPU(s)
            if num_gpus == 1:
                rollout = ray.get(primary.generate_rollouts.remote(batch_prompts))
            else:
                # Split prompts across actors for parallel generation
                chunk = len(batch_prompts) // num_gpus
                futures = []
                for i, actor in enumerate(actors):
                    lo = i * chunk
                    hi = lo + chunk if i < num_gpus - 1 else len(batch_prompts)
                    futures.append(
                        actor.generate_rollouts.remote(batch_prompts[lo:hi])
                    )
                rollout = _merge_rollouts(ray.get(futures))

            gen_time = time.time() - step_start

            # 3. Score completions on CPU workers
            rewards_list = score_batch_distributed(
                rollout["prompts"],
                rollout["completion_texts"],
                context_ref,
                reward_fn_name=config.reward_fn_name,
                chunk_size=config.scoring_batch_size,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32)

            # 4. Compute advantages (group-relative for GRPO)
            advantages = algorithm.compute_advantages(
                rewards, rollout["log_probs"], rollout["ref_log_probs"]
            )

            # 5. Train step — all actors train on the SAME batch
            #    (identical data + identical starting weights → weights stay in sync)
            train_batch = {
                "full_ids": rollout["full_ids"],
                "prompt_len": rollout["prompt_len"],
                "advantages": advantages,
                "old_per_token_log_probs": rollout["per_token_log_probs"],
                "ref_per_token_log_probs": rollout["ref_per_token_log_probs"],
                "completion_mask": rollout["completion_mask"],
                "algorithm_name": config.algorithm,
                "algorithm_kwargs": algo_kwargs,
            }
            train_futures = [
                actor.train_step.remote(train_batch) for actor in actors
            ]
            all_metrics = ray.get(train_futures)
            metrics = all_metrics[0]

            step_time = time.time() - step_start

            # 6. Logging
            if step % config.logging_steps == 0:
                reward_mean = rewards.mean().item()
                reward_std = rewards.std().item()
                if G > 1:
                    reward_best = (
                        rewards.view(-1, G).max(dim=1).values.mean().item()
                    )
                else:
                    reward_best = reward_mean
                log.info(
                    f"step={step} | "
                    f"reward={reward_mean:.4f} | reward_best={reward_best:.4f} | "
                    f"reward_std={reward_std:.4f} | "
                    f"loss={metrics['loss']:.6f} | kl={metrics['kl']:.5f} | "
                    f"grad_norm={metrics['grad_norm']:.4f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"gen={gen_time:.1f}s | total={step_time:.1f}s"
                )

                log_dict = {
                    "train_step": step,
                    "reward_mean": reward_mean,
                    "reward_best": reward_best,
                    "reward_std": reward_std,
                    "loss": metrics["loss"],
                    "kl_divergence": metrics["kl"],
                    "grad_norm": metrics["grad_norm"],
                    "lr": metrics["lr"],
                    "step_time": step_time,
                    "gen_time": gen_time,
                }

                if self._mlflow_active:
                    self._mlflow.log_metrics(log_dict, step=step)

                if self._wandb_active:
                    self._wandb.log(log_dict)

            # 7. Checkpointing
            if step % config.save_steps == 0:
                ckpt_path = str(config.output_dir / f"checkpoint-{step}")
                ray.get(primary.save_checkpoint.remote(ckpt_path))

                if self._mlflow_active:
                    self._mlflow.log_artifacts(
                        ckpt_path,
                        artifact_path=f"checkpoints/checkpoint-{step}",
                    )

        # ── Save final checkpoint ─────────────────────────────────────
        final_path = str(config.output_dir / "final")
        ray.get(primary.save_checkpoint.remote(final_path))
        log.info(f"Training complete. Final model saved to {final_path}")

        if self._mlflow_active:
            self._mlflow.log_artifacts(final_path, artifact_path="checkpoints/final")

        if self._wandb_active:
            self._wandb.finish()
