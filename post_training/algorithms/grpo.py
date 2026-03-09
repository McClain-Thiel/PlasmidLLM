"""GRPO algorithm — Group Relative Policy Optimization (DeepSeek-style).

For each prompt, generates N completions and normalizes rewards within
the group to produce advantages (no value function needed).
"""

from __future__ import annotations

import logging

import ray
import torch

from post_training.algorithms.base import Algorithm, Scorer
from post_training.common.utils import timer, wandb_log

log = logging.getLogger(__name__)


class GRPOAlgorithm(Algorithm):
    """GRPO with multi-actor parallel generation and gradient averaging.

    Usage::

        algo = GRPOAlgorithm(kl_coef=0.3, cliprange=0.2, num_generations=8)
        metrics = algo.step(actors, prompts, scorer)
    """

    def __init__(
        self,
        kl_coef: float = 0.1,
        cliprange: float = 0.2,
        num_generations: int = 4,
        micro_batch_size: int = 64,
        gen_kwargs: dict | None = None,
    ):
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        self.num_generations = num_generations
        self.micro_batch_size = micro_batch_size
        self.gen_kwargs = gen_kwargs or {}
        self._global_step = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        *,
        group_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Group-normalize rewards: (B*G,) → (B*G,) advantages.

        Rewards for the same prompt are contiguous — each group of G
        rewards is independently z-scored.
        """
        G = group_size or self.num_generations
        grouped = rewards.view(-1, G)

        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)

        return ((grouped - means) / stds).view(-1)

    def step(self, actors, prompts, scorer) -> dict[str, float]:
        """Full GRPO step with group sampling and optional multi-actor sharding."""
        self._global_step += 1
        G = self.num_generations
        n_actors = len(actors)

        expanded = [p for p in prompts for _ in range(G)]
        shards = self._shard(expanded, n_actors)

        # Reset peak memory tracking so each step is isolated
        ray.get([a.reset_peak_memory.remote() for a in actors])

        # ── Parallel generation across actors ─────────────────────────────
        log.info(
            "Step %d: generating %d completions (%d prompts × G=%d) across %d actor(s)",
            self._global_step, len(expanded), len(prompts), G, n_actors,
        )
        with timer("generation") as t_gen:
            gen_futures = [
                a.generate.remote(s, **self.gen_kwargs)
                for a, s in zip(actors, shards)
            ]
            generations = ray.get(gen_futures)

        all_prompts = [p for g in generations for p in g.prompts]
        all_completions = [t for g in generations for t in g.completion_texts]

        # Per-actor generation times (reported by the workers themselves)
        actor_gen_times = [g.elapsed_s for g in generations]

        total_comp_tokens = sum(g.completion_ids.numel() for g in generations)
        comp_lengths = [len(t) for t in all_completions]

        log.info(
            "Generation done in %.1fs (actor times: %s) — %d completions, "
            "mean_len=%.0f chars, %d total tokens",
            t_gen(),
            ", ".join(f"{t:.1f}s" for t in actor_gen_times),
            len(all_completions),
            sum(comp_lengths) / max(len(comp_lengths), 1),
            total_comp_tokens,
        )

        # ── Score all completions ─────────────────────────────────────────
        with timer("scoring") as t_score:
            rewards = scorer.score(all_prompts, all_completions)
        advantages = self.compute_advantages(rewards, group_size=G)
        log.info(
            "Scoring done in %.1fs — mean_reward=%.3f std=%.3f min=%.3f max=%.3f",
            t_score(), rewards.mean().item(), rewards.std().item(),
            rewards.min().item(), rewards.max().item(),
        )

        # ── Forward-backward on each actor with its shard ─────────────────
        loss_fn_kwargs = dict(cliprange=self.cliprange, kl_coef=self.kl_coef)

        idx = 0
        total_loss = 0.0
        t_log_probs_total = 0.0
        t_fwd_bwd_total = 0.0

        with timer("train") as t_train:
            for actor, gen in zip(actors, generations):
                n = len(gen.prompts)
                adv_slice = advantages[idx : idx + n]

                with timer() as t_lp:
                    old_lp = ray.get(
                        actor.get_log_probs.remote(gen.full_ids, gen.prompt_len),
                    )
                    ref_lp = ray.get(
                        actor.get_log_probs.remote(
                            gen.full_ids, gen.prompt_len, use_ref=True,
                        ),
                    )
                t_log_probs_total += t_lp()

                with timer() as t_fb:
                    loss = self.micro_batch_forward_backward(
                        actor, gen.full_ids, gen.prompt_len,
                        adv_slice, old_lp.per_token, ref_lp.per_token,
                        old_lp.mask, "grpo", loss_fn_kwargs, self.micro_batch_size,
                    )
                t_fwd_bwd_total += t_fb()

                total_loss += loss
                idx += n

            # ── Sync gradients across actors → step ──────────────────────
            with timer() as t_grad_sync:
                if n_actors > 1:
                    results = self.average_and_apply_gradients(actors)
                else:
                    results = [ray.get(actors[0].clip_and_step.remote())]

        backward_result = results[0]
        log.info(
            "Train done in %.1fs (log_probs=%.1fs fwd_bwd=%.1fs grad_sync=%.1fs) "
            "— loss=%.4f grad_norm=%.4f lr=%.2e",
            t_train(), t_log_probs_total, t_fwd_bwd_total, t_grad_sync(),
            total_loss / n_actors,
            backward_result.grad_norm, backward_result.lr,
        )

        # ── GPU hardware stats ────────────────────────────────────────────
        gpu_stats = ray.get([a.get_gpu_stats.remote() for a in actors])
        gpu_metrics = self._aggregate_gpu_stats(gpu_stats)

        # ── Metrics ───────────────────────────────────────────────────────
        kl = ray.get(actors[0].get_kl.remote(
            generations[0].full_ids, generations[0].prompt_len,
        ))

        reward_groups = rewards.view(-1, G)
        gen_time = max(t_gen(), 1e-6)
        score_time = max(t_score(), 1e-6)

        metrics = {
            "loss": total_loss / n_actors,
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "reward_best": reward_groups.max(dim=1).values.mean().item(),
            "reward_worst": reward_groups.min(dim=1).values.mean().item(),
            "kl": kl["kl_per_seq"].mean().item(),
            "adv_mean": advantages.mean().item(),
            "adv_std": advantages.std().item(),
            "grad_norm": backward_result.grad_norm,
            "lr": backward_result.lr,
            "mean_completion_len": sum(comp_lengths) / max(len(comp_lengths), 1),
            # ── Wall-clock timing breakdown ───────────────────────────
            "time/generation": t_gen(),
            "time/scoring": t_score(),
            "time/train": t_train(),
            "time/train_log_probs": t_log_probs_total,
            "time/train_fwd_bwd": t_fwd_bwd_total,
            "time/train_grad_sync": t_grad_sync(),
            "time/total": t_gen() + t_score() + t_train(),
            "time/generation_pct": t_gen() / (t_gen() + t_score() + t_train()) * 100,
            "time/scoring_pct": t_score() / (t_gen() + t_score() + t_train()) * 100,
            "time/train_pct": t_train() / (t_gen() + t_score() + t_train()) * 100,
            # Per-actor generation times (identifies straggler GPUs)
            "time/gen_actor_mean": sum(actor_gen_times) / max(len(actor_gen_times), 1),
            "time/gen_actor_max": max(actor_gen_times),
            "time/gen_actor_min": min(actor_gen_times),
            # ── Throughput ────────────────────────────────────────────
            "throughput/gen_tokens_per_sec": total_comp_tokens / gen_time,
            "throughput/gen_seqs_per_sec": len(all_completions) / gen_time,
            "throughput/scoring_seqs_per_sec": len(all_completions) / score_time,
        }

        # Merge GPU hardware metrics
        metrics.update(gpu_metrics)

        wandb_log(metrics, step=self._global_step, prefix="grpo/")

        return metrics

    @staticmethod
    def _aggregate_gpu_stats(
        all_stats: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate GPU stats across actors into wandb-friendly metrics.

        Takes the mean across actors for most metrics, max for peak memory.
        """
        if not all_stats or not all_stats[0]:
            return {}

        keys = all_stats[0].keys()
        metrics: dict[str, float] = {}
        for key in keys:
            vals = [s[key] for s in all_stats if key in s]
            if not vals:
                continue
            if "peak" in key:
                metrics[f"gpu/{key}"] = max(vals)
            else:
                metrics[f"gpu/{key}"] = sum(vals) / len(vals)

        return metrics
