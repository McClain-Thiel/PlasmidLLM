"""PPO training script for PlasmidLM post-training with sequence alignment rewards.

PPO uses a learned value function as baseline instead of group statistics (GRPO) or
leave-one-out estimates (RLOO). This avoids the low-diversity problem where DNA models
generate near-identical sequences from the same prompt (reward_std ≈ 0.004).

Usage:
    python scripts/train_ppo.py configs/ppo_g6big.py
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import datasets
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from src.plasmid_llm.models.hf_plasmid_lm import (
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMModel,
    PlasmidLMTokenizer,
)
from post_training.reward import load_motif_lookup
from post_training.ppo_models import PlasmidRewardWrapper, PlasmidValueModel

# Register custom model with AutoConfig/AutoModel so TRL can find it
AutoConfig.register("plasmid_lm", PlasmidLMConfig)
AutoModel.register(PlasmidLMConfig, PlasmidLMModel)
AutoModelForCausalLM.register(PlasmidLMConfig, PlasmidLMForCausalLM)
AutoTokenizer.register(PlasmidLMConfig, slow_tokenizer_class=PlasmidLMTokenizer)

import transformers
transformers.PlasmidLMForCausalLM = PlasmidLMForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PPORunConfig:
    """PPO-specific configuration for PlasmidLM post-training."""

    # Data paths
    model_checkpoint: Path = field(default_factory=lambda: Path("checkpoint"))
    training_pairs: Path = field(default_factory=lambda: Path("training_pairs.parquet"))
    motif_lookup: Path = field(default_factory=lambda: Path("motif_registry.parquet"))

    # PPO hyperparameters
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    total_episodes: Optional[int] = None
    max_steps: int = 5000

    # Generation
    response_length: int = 1024
    temperature: float = 1.0
    local_rollout_forward_batch_size: int = 8

    # PPO-specific
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    whiten_rewards: bool = False

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output/ppo"))
    save_steps: int = 500
    logging_steps: int = 1
    seed: int = 42
    bf16: bool = True

    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "plasmid_post_training"


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_prompt_dataset(parquet_path: str, tokenizer, filter_hard_tokens: bool = True) -> datasets.Dataset:
    """Build a pre-tokenized HF Dataset with 'input_ids' column for PPOTrainer.

    PPOTrainer expects pre-tokenized prompts (unlike GRPO/RLOO which tokenize internally).
    """
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

    if "has_hard_tokens" in col_names and filter_hard_tokens:
        has_hard = table.column("has_hard_tokens").to_pylist()
        indices = [i for i, h in enumerate(has_hard) if h]
        table = table.take(indices)
    elif "reward_motifs" in col_names and filter_hard_tokens:
        motifs = table.column("reward_motifs").to_pylist()
        indices = [i for i, m in enumerate(motifs) if m]
        table = table.take(indices)

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

    # Tokenize prompts — PPOTrainer expects pre-tokenized 'input_ids'
    def tokenize(examples):
        return tokenizer(examples["prompt"], padding=False, truncation=False)

    ds = datasets.Dataset.from_dict({"prompt": prompts})
    ds = ds.map(tokenize, batched=True, remove_columns=["prompt"])
    return ds


class PPOMetricsCallback(TrainerCallback):
    """Print key PPO metrics to stdout on each log step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        step = state.global_step
        parts = [f"step={step}"]
        for key in ["objective/scores", "objective/kl", "loss/policy", "loss/value",
                     "val/clipfrac", "val/ratio", "objective/entropy"]:
            val = logs.get(key)
            if val is not None:
                short = key.split("/")[-1]
                parts.append(f"{short}={val:.4f}")
        lr = logs.get("learning_rate")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        log.info(" | ".join(parts))


class PPOLineageCallback(TrainerCallback):
    """Log PPO config and data lineage to MLflow."""

    def __init__(self, config: PPORunConfig):
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
            params = {
                "model_checkpoint": str(self.config.model_checkpoint),
                "training_pairs": str(self.config.training_pairs),
                "motif_lookup": str(self.config.motif_lookup),
                "learning_rate": self.config.learning_rate,
                "response_length": self.config.response_length,
                "temperature": self.config.temperature,
                "kl_coef": self.config.kl_coef,
                "cliprange": self.config.cliprange,
                "vf_coef": self.config.vf_coef,
                "num_ppo_epochs": self.config.num_ppo_epochs,
                "git_commit": _get_git_commit(),
                "architecture": "plasmid_lm_ppo",
            }
            if model:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params["total_params"] = n_params
            mlflow.log_params(params)
            log.info(f"Logged {len(params)} config params to MLflow")
            self.logged_params = True
        except Exception as e:
            log.warning(f"Config lineage logging failed: {e}")

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            mlflow = self._get_mlflow()
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint_dir.exists():
                return
            if model is not None and hasattr(model, "config"):
                model.config.save_pretrained(str(checkpoint_dir))
            for parent in [Path(args.output_dir), self.config.model_checkpoint]:
                vocab_src = parent / "vocab.json"
                if vocab_src.exists():
                    shutil.copy2(vocab_src, checkpoint_dir / "vocab.json")
                    break
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


def load_config(config_path: Path) -> PPORunConfig:
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
    parser = argparse.ArgumentParser(description="Train PlasmidLM with PPO")
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

    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    log.info(f"Config loaded: {config.model_checkpoint.name}")

    # ── Load model and tokenizer ─────────────────────────────────────────────

    log.info(f"Loading model from {config.model_checkpoint}")
    policy = PlasmidLMForCausalLM.from_pretrained(str(config.model_checkpoint))
    tokenizer = PlasmidLMTokenizer.from_pretrained(str(config.model_checkpoint))
    tokenizer.padding_side = "left"

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info(f"Policy model loaded: {n_params:,} trainable params")

    # Reference model — frozen copy for KL divergence
    log.info("Loading reference model...")
    ref_policy = PlasmidLMForCausalLM.from_pretrained(str(config.model_checkpoint))

    # ── Value model — backbone from pretrained + new scalar head ─────────────

    log.info("Building value model...")
    value_backbone = PlasmidLMModel(policy.config)
    value_backbone.load_state_dict(policy.model.state_dict())
    value_model = PlasmidValueModel(value_backbone, policy.config.hidden_size)
    log.info(f"Value model: {sum(p.numel() for p in value_model.parameters()):,} params")

    # ── Reward model — non-neural wrapper around Smith-Waterman alignment ────

    log.info(f"Loading motif lookup from {config.motif_lookup}")
    lookup_df = load_motif_lookup(str(config.motif_lookup))
    log.info(f"Motif lookup loaded: {len(lookup_df)} entries, {len(lookup_df.index.unique())} unique tokens")

    reward_model = PlasmidRewardWrapper(tokenizer, lookup_df)
    log.info("Reward model: Smith-Waterman alignment wrapper")

    # ── Dataset — pre-tokenized prompts ──────────────────────────────────────

    log.info(f"Loading prompts from {config.training_pairs}")
    dataset = build_prompt_dataset(str(config.training_pairs), tokenizer, filter_hard_tokens=True)
    log.info(f"Dataset loaded: {len(dataset)} prompts with hard tokens")

    # ── MLflow setup ─────────────────────────────────────────────────────────

    report_to = "none"
    if config.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            exp = mlflow.set_experiment(config.mlflow_experiment)
            if exp is None:
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

    # ── PPO configuration ────────────────────────────────────────────────────

    from trl.experimental.ppo import PPOConfig, PPOTrainer

    ppo_config = PPOConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        total_episodes=config.total_episodes,

        # Generation
        response_length=config.response_length,
        temperature=config.temperature,
        local_rollout_forward_batch_size=config.local_rollout_forward_batch_size,

        # PPO algorithm
        num_ppo_epochs=config.num_ppo_epochs,
        num_mini_batches=config.num_mini_batches,
        kl_coef=config.kl_coef,
        cliprange=config.cliprange,
        vf_coef=config.vf_coef,
        cliprange_value=config.cliprange_value,
        gamma=config.gamma,
        lam=config.lam,
        whiten_rewards=config.whiten_rewards,

        # Logging
        num_sample_generations=0,  # disable eval sampling (no eval_dataset)
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=report_to,

        # System
        bf16=config.bf16,
        seed=config.seed,

        # Model paths (PPOConfig requires these as strings)
        sft_model_path=str(config.model_checkpoint),
        reward_model_path=str(config.model_checkpoint),
    )

    # ── Create PPO trainer ───────────────────────────────────────────────────

    log.info("Initializing PPO trainer...")
    callbacks = [PPOMetricsCallback()]
    if report_to == "mlflow":
        callbacks.append(PPOLineageCallback(config))

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    # ── Train ────────────────────────────────────────────────────────────────

    log.info("Starting PPO training...")
    log.info(f"  response_length={config.response_length}")
    log.info(f"  kl_coef={config.kl_coef}, cliprange={config.cliprange}")
    log.info(f"  num_ppo_epochs={config.num_ppo_epochs}")
    log.info(f"  batch={config.per_device_train_batch_size} x grad_accum={config.gradient_accumulation_steps}")
    trainer.train()

    # ── Save final model ─────────────────────────────────────────────────────

    final_dir = config.output_dir / "final"
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log.info("PPO training complete!")


if __name__ == "__main__":
    main()
