"""HF-native training script for PlasmidLM (transformer-only, no Hydra).

LEGACY: This script still works but uses command-line arguments instead of config files.
For new projects, use `train_with_config.py` with Python config files instead.

See: docs/CONFIG_TRAINING.md for the new approach.
"""

from __future__ import annotations

import argparse
import logging
import os
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

from plasmid_llm.data import PlasmidDataset, train_val_split
from plasmid_llm.models.hf_plasmid_lm import PlasmidLMConfig, PlasmidLMForCausalLM, PlasmidLMTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file for data lineage tracking."""
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]  # First 16 chars
    except Exception:
        return "unknown"


class DataLineageCallback(TrainerCallback):
    """Log data provenance and computed metrics to MLflow.
    
    Doesn't manage runs - that's handled by HF's built-in MLflow integration.
    Just adds custom data lineage params and perplexity metrics.
    """

    def __init__(self, train_args_ns):
        self.train_args_ns = train_args_ns

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            import mlflow
            a = self.train_args_ns
            
            # Log data lineage
            lineage_params = {
                "data_path": a.data_path,
                "vocab_path": a.vocab_path,
                "data_sha256": _compute_file_hash(a.data_path),
                "vocab_sha256": _compute_file_hash(a.vocab_path),
                "git_commit": _get_git_commit(),
            }
            
            # Add motif registry if provided
            if hasattr(a, "motif_registry_path") and a.motif_registry_path:
                lineage_params["motif_registry_path"] = a.motif_registry_path
                lineage_params["motif_registry_sha256"] = _compute_file_hash(a.motif_registry_path)
            
            mlflow.log_params(lineage_params)
            
        except Exception as e:
            log.warning(f"Data lineage logging failed: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        try:
            import math
            # Add perplexity to logs dict (HF's MLflow integration will pick it up)
            if "loss" in logs and "train_perplexity" not in logs:
                logs["train_perplexity"] = math.exp(min(logs["loss"], 20))
            if "eval_loss" in logs and "eval_perplexity" not in logs:
                logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 20))
        except Exception as e:
            log.warning(f"Perplexity computation failed: {e}")


class GenerationSampleCallback(TrainerCallback):
    """Generate full-length sample sequences at each evaluation and log to MLflow."""

    def __init__(self, val_dataset, tokenizer, max_new_tokens: int = 8000, n_samples: int = 3):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.n_samples = n_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        device = next(model.parameters()).device
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        lines = [f"=== Generation samples at step {state.global_step} ===\n"]

        for i in range(min(self.n_samples, len(self.val_dataset))):
            item = self.val_dataset[i]
            ids_list = item["input_ids"].tolist()

            try:
                sep_pos = ids_list.index(sep_id)
            except ValueError:
                continue

            # Include BOS + prompt + SEP as the conditioning prefix
            prompt_ids = ids_list[:sep_pos + 1]
            prompt_text = self.tokenizer.decode(ids_list[1:sep_pos])  # skip BOS for display

            # True completion (strip padding and EOS for display)
            true_ids = ids_list[sep_pos + 1:]
            for stop_id in (pad_id, eos_id):
                if stop_id in true_ids:
                    true_ids = true_ids[:true_ids.index(stop_id)]
            true_seq = self.tokenizer.decode(true_ids)

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.8,
                        do_sample=True,
                    )
                gen_ids = output_ids[0, len(prompt_ids):].tolist()
                # Strip EOS if present
                if eos_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_id)]
                gen_seq = self.tokenizer.decode(gen_ids)
            except Exception as e:
                gen_seq = f"[generation failed: {e}]"

            lines.append(f"--- Sample {i+1} ---")
            lines.append(f"Prompt ({len(prompt_ids)-1} tokens): {prompt_text}")
            lines.append(f"True length:      {len(true_seq)} bp")
            lines.append(f"Generated length: {len(gen_seq)} bp")
            lines.append(f"True sequence:\n{true_seq}")
            lines.append(f"Generated sequence:\n{gen_seq}\n")

        sample_path = os.path.join(args.output_dir, f"samples_step{state.global_step}.txt")
        with open(sample_path, "w") as f:
            f.write("\n".join(lines))
        try:
            import mlflow
            mlflow.log_artifact(sample_path)
        except Exception as e:
            log.warning(f"Failed to log generation samples: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PlasmidLM with HF Trainer")

    # Data
    p.add_argument("--data_path", required=True, help="Path to parquet file")
    p.add_argument("--vocab_path", required=True, help="Path to vocab.json")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--val_split", type=float, default=0.05)

    # Model
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--num_hidden_layers", type=int, default=10)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=1536)

    # Training
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # MLflow
    p.add_argument("--mlflow_tracking_uri", type=str, default=None)
    p.add_argument("--mlflow_experiment", type=str, default=None)
    p.add_argument("--motif_registry_path", type=str, default=None, 
                   help="Path to motif registry (for data lineage tracking)")

    # Resume / early stopping
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--gradient_checkpointing", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # --- MLflow setup (use HF's built-in integration) ---
    report_to = "none"
    if args.mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)
        report_to = "mlflow"
        log.info(f"MLflow tracking: {args.mlflow_tracking_uri}, experiment: {args.mlflow_experiment}")

    # --- Tokenizer + Dataset ---
    log.info(f"Loading tokenizer from {args.vocab_path}")
    tokenizer = PlasmidLMTokenizer(args.vocab_path)

    log.info(f"Loading dataset from {args.data_path}")
    dataset = PlasmidDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    train_ds, val_ds = train_val_split(dataset, val_split=args.val_split, seed=args.seed)
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # --- Model ---
    padded_vocab = ((tokenizer.vocab_size + 7) // 8) * 8
    config = PlasmidLMConfig(
        vocab_size=padded_vocab,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_seq_len, 16384),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = PlasmidLMForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} trainable params | vocab={padded_vocab}")

    # --- HF TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=report_to,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # --- Trainer ---
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        GenerationSampleCallback(val_ds, tokenizer),
    ]
    
    # Add data lineage tracking if MLflow is enabled
    if args.mlflow_tracking_uri:
        callbacks.append(DataLineageCallback(args))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    # --- Train ---
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save final model + tokenizer in HF format ---
    final_dir = os.path.join(args.output_dir, "final")
    log.info(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    log.info("Done!")


if __name__ == "__main__":
    main()
