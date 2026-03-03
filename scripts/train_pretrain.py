"""Pretraining script for PlasmidLM.

Usage:
    python scripts/train_pretrain.py configs/pretrain_g6big.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plasmid_llm.config import PretrainingConfig
from plasmid_llm.data import PlasmidDataset, train_val_split
from plasmid_llm.models.hf_plasmid_lm import (
    PlasmidKmerTokenizer,
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMTokenizer,
    build_kmer_vocab,
)
from plasmid_llm.utils import _get_git_commit, load_config, load_env_file, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax predictions to save memory during eval."""
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    if mask.any():
        acc = (preds[mask] == labels[mask]).astype(float).mean().item()
    else:
        acc = 0.0
    return {"token_accuracy": acc}


class MLflowExtrasCallback(TrainerCallback):
    """Log perplexity and checkpoint artifacts to MLflow."""

    def __init__(self, config: PretrainingConfig):
        self.config = config
        self._mlflow = None

    def _get_mlflow(self):
        if self._mlflow is None:
            import mlflow
            self._mlflow = mlflow
        return self._mlflow

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        mlflow = self._get_mlflow()
        params = self.config.to_mlflow_params()
        params["git_commit"] = _get_git_commit()
        if model:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            params["total_params"] = n_params
        mlflow.log_params(params)
        log.info(f"Logged {len(params)} config params to MLflow")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        mlflow = self._get_mlflow()
        metrics = {}
        if "loss" in logs:
            metrics["train_perplexity"] = math.exp(min(logs["loss"], 20))
        if "eval_loss" in logs:
            metrics["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
        mlflow = self._get_mlflow()
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        if model is not None and hasattr(model, "config"):
            model.config.save_pretrained(str(checkpoint_dir))

        # Copy tokenizer + special_tokens into checkpoint for self-containment
        vocab_src = Path(args.output_dir) / "vocab.json"
        if vocab_src.exists():
            shutil.copy2(vocab_src, checkpoint_dir / "vocab.json")

        st_src = Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt"
        if st_src.exists():
            shutil.copy2(st_src, checkpoint_dir / "special_tokens.txt")

        mlflow.log_artifacts(
            str(checkpoint_dir),
            artifact_path=f"checkpoints/checkpoint-{state.global_step}",
        )
        log.info(f"Logged checkpoint-{state.global_step} to MLflow artifacts")


class PerplexityCallback(TrainerCallback):
    """Compute and log perplexity from loss for both train and eval."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        extra = {}
        if "loss" in logs:
            extra["train/perplexity"] = math.exp(min(logs["loss"], 20))
        if "eval_loss" in logs:
            extra["eval/perplexity"] = math.exp(min(logs["eval_loss"], 20))
        if extra:
            # Log via any active reporter (wandb/mlflow pick these up automatically)
            logs.update(extra)


def _build_run_name(config: PretrainingConfig) -> str:
    """Build a descriptive W&B run name from config."""
    parts = []
    if config.tokenizer_type == "kmer":
        parts.append(f"kmer{config.kmer_k}_s{config.kmer_stride}")
    else:
        parts.append("char")
    if config.use_moe:
        parts.append(f"moe{config.num_experts}x{config.moe_intermediate_size or config.intermediate_size}")
    else:
        parts.append("dense")
    parts.append(f"d{config.hidden_size}_L{config.num_hidden_layers}")
    return "_".join(parts)


def _build_wandb_tags(config: PretrainingConfig) -> list[str]:
    """Build descriptive tags for W&B run."""
    tags = ["pretraining", config.tokenizer_type]
    if config.use_moe:
        tags.append("moe")
    else:
        tags.append("dense")
    if config.tokenizer_type == "kmer":
        tags.append(f"k={config.kmer_k}")
        tags.append(f"stride={config.kmer_stride}")
    return tags


def build_tokenizer(config: PretrainingConfig) -> PlasmidLMTokenizer | PlasmidKmerTokenizer:
    """Build tokenizer from config (char-level or k-mer)."""
    with open(config.special_tokens) as f:
        special_tokens = [line.strip() for line in f if line.strip()]

    log.info(f"Loaded {len(special_tokens)} special tokens from {config.special_tokens}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.tokenizer_type == "kmer":
        vocab = build_kmer_vocab(special_tokens, k=config.kmer_k)
        vocab_file = config.output_dir / "vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=2)
        log.info(f"Created k-mer vocab with {len(vocab)} tokens (k={config.kmer_k}, stride={config.kmer_stride})")
        return PlasmidKmerTokenizer(str(vocab_file), k=config.kmer_k, stride=config.kmer_stride)
    else:
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        next_id = len(vocab)
        for base in "ATCGNatcgn":
            if base not in vocab:
                vocab[base] = next_id
                next_id += 1
        vocab_file = config.output_dir / "vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(vocab, f, indent=2)
        log.info(f"Created char vocab with {len(vocab)} tokens at {vocab_file}")
        return PlasmidLMTokenizer(str(vocab_file))


def main():
    parser = argparse.ArgumentParser(description="Pretrain PlasmidLM")
    parser.add_argument("config", type=Path, help="Path to Python config file")
    args = parser.parse_args()

    load_env_file()

    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    if not isinstance(config, PretrainingConfig):
        raise TypeError(
            f"Config must be PretrainingConfig, got {type(config).__name__}"
        )
    log.info(f"Config: {config.training_pairs.name}, output: {config.output_dir}")

    # Build tokenizer
    tokenizer = build_tokenizer(config)

    # Load dataset
    log.info(f"Loading dataset from {config.training_pairs}")
    dataset = PlasmidDataset(
        str(config.training_pairs), tokenizer, max_seq_len=config.max_seq_len
    )
    train_ds, val_ds = train_val_split(dataset, val_split=config.val_split, seed=config.seed)
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Build model
    padded_vocab = ((tokenizer.vocab_size + 7) // 8) * 8
    model_config = PlasmidLMConfig(
        vocab_size=padded_vocab,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=max(config.max_seq_len, 16384),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # MoE
        use_moe=config.use_moe,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_intermediate_size=config.moe_intermediate_size,
        aux_loss_coef=config.aux_loss_coef,
        # Tokenizer metadata
        tokenizer_type=config.tokenizer_type,
        kmer_k=config.kmer_k if config.tokenizer_type == "kmer" else None,
        kmer_stride=config.kmer_stride if config.tokenizer_type == "kmer" else None,
    )
    model = PlasmidLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} params | vocab={padded_vocab}")

    # Tracking setup
    report_to = []
    mlflow_active = setup_mlflow(config.mlflow_tracking_uri, config.mlflow_experiment)
    if mlflow_active:
        report_to.append("mlflow")

    wandb_active = False
    if config.wandb_project:
        try:
            import wandb
            run_name = config.wandb_run_name or _build_run_name(config)
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=config.to_mlflow_params(),  # reuse param dict
                tags=_build_wandb_tags(config),
            )
            report_to.append("wandb")
            wandb_active = True
            log.info(f"W&B: project={config.wandb_project}, run={run_name}")
        except Exception as e:
            log.warning(f"W&B setup failed: {e} — continuing without W&B")

    if not report_to:
        report_to = ["none"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        bf16=config.bf16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        max_grad_norm=config.max_grad_norm,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=report_to,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        PerplexityCallback(),
    ]
    if mlflow_active:
        callbacks.append(MLflowExtrasCallback(config))

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    trainer.train()

    # Save final model + tokenizer
    final_dir = config.output_dir / "final"
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    model.config.save_pretrained(str(final_dir))

    if mlflow_active:
        import mlflow
        mlflow.log_artifacts(str(final_dir), artifact_path="final_model")
        log.info("Logged final model to MLflow artifacts")

    log.info("Done!")


if __name__ == "__main__":
    main()
