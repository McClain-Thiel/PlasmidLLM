"""Config-based training script for PlasmidLM pretraining.

Usage:
    python scripts/train_with_config.py configs/pretraining_example.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
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

from plasmid_llm.config import PretrainingConfig
from plasmid_llm.data import PlasmidDataset, train_val_split
from plasmid_llm.models.hf_plasmid_lm import (
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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
    """Log perplexity metrics and checkpoint artifacts to MLflow.

    HF's built-in MLflowCallback logs metrics before custom callbacks run,
    so we log perplexity directly to MLflow ourselves. We also upload
    checkpoint directories as artifacts on each save.
    """

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
        try:
            mlflow = self._get_mlflow()
            params = self.config.to_mlflow_params()
            params["git_commit"] = _get_git_commit()
            if model:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params["total_params"] = n_params
            mlflow.log_params(params)
            log.info(f"Logged {len(params)} config params to MLflow")
        except Exception as e:
            log.warning(f"Config lineage logging failed: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        import math
        try:
            mlflow = self._get_mlflow()
            metrics = {}
            if "loss" in logs:
                metrics["train_perplexity"] = math.exp(min(logs["loss"], 20))
            if "eval_loss" in logs:
                metrics["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)
        except Exception as e:
            log.warning(f"Perplexity logging failed: {e}")

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            mlflow = self._get_mlflow()
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint_dir.exists():
                return

            # Ensure model config is saved alongside weights
            if model is not None and hasattr(model, "config"):
                model.config.save_pretrained(str(checkpoint_dir))

            # Copy tokenizer files into checkpoint so it's self-contained
            vocab_src = Path(args.output_dir) / "vocab.json"
            if vocab_src.exists():
                import shutil
                shutil.copy2(vocab_src, checkpoint_dir / "vocab.json")

            # Copy special_tokens.txt for full reproducibility
            st_src = Path(args.output_dir).parent / "data" / "special_tokens.txt"
            if not st_src.exists():
                # Try relative to script
                st_src = Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt"
            if st_src.exists():
                import shutil
                shutil.copy2(st_src, checkpoint_dir / "special_tokens.txt")

            mlflow.log_artifacts(str(checkpoint_dir), artifact_path=f"checkpoints/checkpoint-{state.global_step}")
            log.info(f"Logged checkpoint-{state.global_step} to MLflow artifacts (weights + config + tokenizer)")
        except Exception as e:
            log.warning(f"Checkpoint artifact logging failed: {e}")


def load_config(config_path: Path) -> PretrainingConfig:
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


def build_tokenizer_from_special_tokens(special_tokens_path: Path, output_dir: Path) -> PlasmidLMTokenizer:
    """Build tokenizer dynamically from special tokens file."""
    import json
    
    # Read special tokens
    with open(special_tokens_path) as f:
        special_tokens = [line.strip() for line in f if line.strip()]
    
    log.info(f"Loaded {len(special_tokens)} special tokens from {special_tokens_path}")
    
    # Create vocab
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add DNA bases (tokenizer will add these automatically too, but we're explicit)
    next_id = len(vocab)
    for base in "ATCGNatcgn":
        if base not in vocab:
            vocab[base] = next_id
            next_id += 1
    
    # Save vocab to temp file
    vocab_file = output_dir / "vocab.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump(vocab, f, indent=2)
    
    log.info(f"Created vocab with {len(vocab)} tokens at {vocab_file}")
    
    return PlasmidLMTokenizer(str(vocab_file))


def main():
    parser = argparse.ArgumentParser(description="Train PlasmidLM with config file")
    parser.add_argument("config", type=Path, help="Path to Python config file")
    args = parser.parse_args()

    # Load config
    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    log.info(f"Config: {config.training_pairs.name}, output: {config.output_dir}")

    # Build tokenizer from special tokens
    tokenizer = build_tokenizer_from_special_tokens(
        config.special_tokens, config.output_dir
    )

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
    )
    model = PlasmidLMForCausalLM(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} params | vocab={padded_vocab}")

    # Load .env for Databricks credentials
    import os
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
        log.info(f"Loaded env vars from {env_file}")

    # Setup MLflow
    report_to = "none"
    mlflow_uri = config.mlflow_tracking_uri
    if mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        report_to = "mlflow"
        log.info(f"MLflow: {mlflow_uri} / {config.mlflow_experiment}")

    # HF TrainingArguments
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
    ]
    if mlflow_uri:
        callbacks.append(MLflowExtrasCallback(config))

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    # Train
    log.info("Starting training...")
    trainer.train()

    # Save final model + tokenizer
    final_dir = config.output_dir / "final"
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    model.config.save_pretrained(str(final_dir))

    # Log final model to MLflow
    if mlflow_uri:
        import mlflow
        mlflow.log_artifacts(str(final_dir), artifact_path="final_model")
        log.info("Logged final model to MLflow artifacts")

    log.info("Done!")


if __name__ == "__main__":
    main()
