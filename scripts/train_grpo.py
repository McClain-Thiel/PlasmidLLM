"""GRPO training script for PlasmidLM post-training with sequence alignment rewards.

Usage:
    python scripts/train_grpo.py configs/grpo_g6big.py
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import datasets
import pyarrow.parquet as pq
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM

from src.plasmid_llm.config import PostTrainingConfig
from src.plasmid_llm.models.hf_plasmid_lm import (
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMModel,
    PlasmidLMTokenizer,
)
from post_training.reward import build_category_index, load_motif_lookup, plasmid_reward_fn

# Register custom model so AutoConfig/AutoModelForCausalLM can find it
# (required by TRL's GRPOTrainer for creating the reference model)
AutoConfig.register("plasmid_lm", PlasmidLMConfig)
AutoModelForCausalLM.register(PlasmidLMConfig, PlasmidLMForCausalLM)

# TRL's create_model_from_path does getattr(transformers, "PlasmidLMForCausalLM")
# so the class must be visible on the transformers module
import transformers
transformers.PlasmidLMForCausalLM = PlasmidLMForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_prompt_dataset(parquet_path: str, filter_hard_tokens: bool = True) -> datasets.Dataset:
    """Build a HF Dataset with a 'prompt' string column for GRPOTrainer.

    GRPOTrainer handles tokenization internally — we just need to provide
    prompt strings. Each prompt gets <SEP> appended for generation.
    """
    # Read only lightweight columns — avoid pulling huge sequence/full_text over S3
    meta = pq.ParquetFile(parquet_path).schema.names
    need_cols = []
    for c in ["prompt", "token_prompt", "reward_motifs", "has_hard_tokens"]:
        if c in meta:
            need_cols.append(c)
    # Fallback: if no prompt column, we need full_text
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
        # reward_motifs is non-empty for prompts with scorable motifs
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
            if match:
                prompts.append(match.group(1))
            else:
                prompts.append(text)
    else:
        raise ValueError(f"No valid prompt column found. Available: {col_names}")

    # Append <SEP> for generation
    prompts = [p + "<SEP>" for p in prompts]
    log.info(f"Built {len(prompts)} prompts (filtered={filter_hard_tokens})")

    return datasets.Dataset.from_dict({"prompt": prompts})


class GRPOMetricsCallback(TrainerCallback):
    """Print key GRPO metrics to stdout on each log step."""

    def __init__(self, curriculum_state: dict | None = None):
        self.curriculum_state = curriculum_state

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        step = state.global_step
        reward = logs.get("reward", logs.get("rewards/reward_fn/mean"))
        reward_std = logs.get("reward_std", logs.get("rewards/reward_fn/std"))
        kl = logs.get("kl")
        loss = logs.get("loss")
        clip = logs.get("completions/clipped_ratio")
        mean_len = logs.get("completions/mean_length")
        step_time = logs.get("step_time")
        parts = [f"step={step}"]
        if self.curriculum_state is not None:
            parts.append(f"alpha={self.curriculum_state['alpha']:.3f}")
        if reward is not None:
            parts.append(f"reward={reward:.4f}")
        if reward_std is not None:
            parts.append(f"reward_std={reward_std:.4f}")
        if kl is not None:
            parts.append(f"kl={kl:.5f}")
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if clip is not None:
            parts.append(f"clip={clip:.2f}")
        if mean_len is not None:
            parts.append(f"len={mean_len:.0f}")
        if step_time is not None:
            parts.append(f"time={step_time:.1f}s")
        log.info(" | ".join(parts))


class CurriculumCallback(TrainerCallback):
    """Linearly ramp curriculum alpha from start to end over warmup steps."""

    def __init__(self, curriculum_state: dict, config: PostTrainingConfig):
        self.curriculum_state = curriculum_state
        self.alpha_start = config.curriculum_alpha_start
        self.alpha_end = config.curriculum_alpha_end
        self.warmup_steps = max(config.curriculum_alpha_warmup_steps, 1)

    def on_step_begin(self, args, state, control, **kwargs):
        progress = min(state.global_step / self.warmup_steps, 1.0)
        self.curriculum_state["alpha"] = (
            self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        )


class GRPOLineageCallback(TrainerCallback):
    """Log GRPO config and data lineage to MLflow."""

    def __init__(self, config: PostTrainingConfig):
        self.config = config
        self.logged_params = False
        self._mlflow = None

    def _get_mlflow(self):
        if self._mlflow is None:
            import mlflow
            self._mlflow = mlflow
        return self._mlflow

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or self.logged_params:
            return

        try:
            mlflow = self._get_mlflow()

            params = self.config.to_mlflow_params()
            params["git_commit"] = _get_git_commit()
            params["architecture"] = "plasmid_lm_grpo"

            if model:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params["total_params"] = n_params

            mlflow.log_params(params)
            log.info(f"Logged {len(params)} config params to MLflow")
            self.logged_params = True

        except Exception as e:
            log.warning(f"Config lineage logging failed: {e}")

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Upload self-contained checkpoint (weights + config + vocab) to MLflow."""
        if not state.is_world_process_zero:
            return
        try:
            mlflow = self._get_mlflow()
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint_dir.exists():
                return

            # Save model config alongside weights
            if model is not None and hasattr(model, "config"):
                model.config.save_pretrained(str(checkpoint_dir))

            # Copy vocab.json into checkpoint for self-containment
            for parent in [Path(args.output_dir), self.config.model_checkpoint]:
                vocab_src = parent / "vocab.json"
                if vocab_src.exists():
                    shutil.copy2(vocab_src, checkpoint_dir / "vocab.json")
                    break

            # Copy special_tokens.txt
            st_src = Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt"
            if st_src.exists():
                shutil.copy2(st_src, checkpoint_dir / "special_tokens.txt")

            mlflow.log_artifacts(
                str(checkpoint_dir),
                artifact_path=f"checkpoints/checkpoint-{state.global_step}",
            )
            log.info(f"Logged checkpoint-{state.global_step} to MLflow artifacts")
        except Exception as e:
            log.warning(f"Checkpoint artifact logging failed: {e}")


def load_config(config_path: Path) -> PostTrainingConfig:
    """Load config from Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "config"):
        raise ValueError(f"Config file must define 'config' variable: {config_path}")

    return module.config


def main():
    parser = argparse.ArgumentParser(description="Train PlasmidLM with GRPO")
    parser.add_argument("config", type=Path, help="Path to Python config file")
    args = parser.parse_args()

    # Load .env for Databricks credentials
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
        log.info(f"Loaded env vars from {env_file}")

    # Load config
    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    log.info(f"Config loaded: {config.model_checkpoint.name}")

    # Load model and tokenizer
    log.info(f"Loading model from {config.model_checkpoint}")
    model = PlasmidLMForCausalLM.from_pretrained(str(config.model_checkpoint))
    tokenizer = PlasmidLMTokenizer.from_pretrained(str(config.model_checkpoint))

    # GRPOTrainer requires left-padding for generation
    tokenizer.padding_side = "left"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model loaded: {n_params:,} trainable params")

    # Load motif lookup for reward function
    log.info(f"Loading motif lookup from {config.motif_lookup}")
    lookup_df = load_motif_lookup(str(config.motif_lookup))
    log.info(f"Motif lookup loaded: {len(lookup_df)} entries, {len(lookup_df.index.unique())} unique tokens")

    # Build category index for curriculum presence scoring
    category_index = build_category_index(lookup_df)
    log.info(f"Category index built: {list(category_index.keys())} "
             f"({sum(len(v) for v in category_index.values())} representative entries)")

    # Curriculum alpha state — mutable dict so the reward_fn closure sees updates
    curriculum_state = {"alpha": config.curriculum_alpha_start}
    log.info(f"Curriculum: alpha {config.curriculum_alpha_start} -> {config.curriculum_alpha_end} "
             f"over {config.curriculum_alpha_warmup_steps} steps")

    # Build HF Dataset with "prompt" column (GRPOTrainer tokenizes internally)
    log.info(f"Loading prompts from {config.training_pairs}")
    dataset = build_prompt_dataset(str(config.training_pairs), filter_hard_tokens=True)
    log.info(f"Dataset loaded: {len(dataset)} prompts with hard tokens")

    # Setup MLflow
    report_to = "none"
    if config.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            exp = mlflow.set_experiment(config.mlflow_experiment)
            if exp is None:
                # Databricks mlflow 3.9+ may return None for new experiments — create explicitly
                log.info("set_experiment returned None, creating experiment explicitly...")
                exp_id = mlflow.create_experiment(config.mlflow_experiment)
                exp = mlflow.set_experiment(experiment_id=exp_id)
            os.environ["MLFLOW_EXPERIMENT_NAME"] = config.mlflow_experiment
            os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri
            report_to = "mlflow"
            exp_id = exp.experiment_id if exp else "unknown"
            log.info(f"MLflow: {config.mlflow_tracking_uri} / {config.mlflow_experiment} (id={exp_id})")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} — continuing without MLflow")

    # Define reward function wrapper — TRL 0.16+ signature: (prompts, completions, **kwargs)
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Reward function for GRPO. Scores generated DNA against motif references."""
        # completions are list of dicts with "content" key, or plain strings
        texts = []
        for c in completions:
            if isinstance(c, dict):
                texts.append(c.get("content", ""))
            elif isinstance(c, list):
                # list of message dicts
                texts.append("".join(m.get("content", "") for m in c if isinstance(m, dict)))
            else:
                texts.append(str(c))
        return plasmid_reward_fn(
            prompts, texts, lookup_df,
            alpha=curriculum_state["alpha"],
            category_index=category_index,
        )

    # GRPO configuration — TRL 0.16+ parameter names
    grpo_config = GRPOConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps if config.max_steps else -1,

        # Sampling parameters
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,

        # GRPO-specific
        num_iterations=config.num_iterations,
        beta=config.beta,
        epsilon=config.epsilon,
        loss_type=config.loss_type,

        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        report_to=report_to,

        # System
        bf16=config.bf16,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,

        # vLLM (disabled by default for small model)
        use_vllm=config.use_vllm,
    )

    # Create GRPO trainer
    log.info("Initializing GRPO trainer...")
    callbacks = [
        GRPOMetricsCallback(curriculum_state),
        CurriculumCallback(curriculum_state, config),
    ]
    if report_to == "mlflow":
        callbacks.append(GRPOLineageCallback(config))

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )

    # Train
    log.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    final_dir = config.output_dir / "final"
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log.info("GRPO training complete!")


if __name__ == "__main__":
    main()
