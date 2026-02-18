"""HF-native training script for PlasmidLM (transformer-only, no Hydra)."""

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


class PlasmidMLflowCallback(TrainerCallback):
    """Log model metadata tags to MLflow at training start."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        try:
            import mlflow
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.set_tags({
                "architecture": "plasmid_lm_transformer",
                "git_commit": _get_git_commit(),
                "total_params": str(n_params),
                "hidden_size": str(model.config.hidden_size),
                "num_layers": str(model.config.num_hidden_layers),
                "num_heads": str(model.config.num_attention_heads),
            })
        except Exception as e:
            log.warning(f"MLflow tag logging failed: {e}")


class GenerationSampleCallback(TrainerCallback):
    """Generate sample sequences at each evaluation and log to MLflow."""

    def __init__(self, val_dataset, tokenizer: PlasmidLMTokenizer, max_new_tokens: int = 512, n_samples: int = 3):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.n_samples = n_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        try:
            import mlflow
        except Exception:
            return

        if model is None:
            return

        model.eval()
        device = next(model.parameters()).device
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        lines = [f"=== Generation samples at step {state.global_step} ===\n"]

        for i in range(min(self.n_samples, len(self.val_dataset))):
            item = self.val_dataset[i]
            ids_list = item["input_ids"].tolist()

            try:
                sep_pos = ids_list.index(sep_id)
            except ValueError:
                continue

            prompt_ids = ids_list[:sep_pos + 1]
            prompt_text = self.tokenizer.decode(ids_list[:sep_pos])

            true_ids = ids_list[sep_pos + 1:]
            if pad_id in true_ids:
                true_ids = true_ids[:true_ids.index(pad_id)]
            true_text = self.tokenizer.decode(true_ids)

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            try:
                output_ids = model.generate(input_tensor, max_new_tokens=self.max_new_tokens, temperature=0.8)
                gen_ids = output_ids[0, len(prompt_ids):].tolist()
                gen_text = self.tokenizer.decode(gen_ids)
            except Exception as e:
                gen_text = f"[generation failed: {e}]"

            lines.append(f"--- Sample {i+1} ---")
            lines.append(f"Prompt:    {prompt_text[:200]}")
            lines.append(f"True:      {true_text[:300]}...")
            lines.append(f"Generated: {gen_text[:300]}...")
            lines.append(f"True len:  {len(true_text)} | Gen len: {len(gen_text)}\n")

        sample_path = os.path.join(args.output_dir, f"samples_step{state.global_step}.txt")
        with open(sample_path, "w") as f:
            f.write("\n".join(lines))
        try:
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

    # Resume / early stopping
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    # --- MLflow env vars (must be set before Trainer init) ---
    report_to = "none"
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        report_to = "mlflow"
    if args.mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment

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
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            PlasmidMLflowCallback(),
            GenerationSampleCallback(val_ds, tokenizer, max_new_tokens=512),
        ],
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
