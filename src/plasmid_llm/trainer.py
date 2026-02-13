"""Unified training loop with MLflow logging for all architectures."""

from __future__ import annotations

import csv
import logging
import os
import psutil
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from plasmid_llm.utils.metrics import compute_perplexity

log = logging.getLogger(__name__)

# Try to import pynvml for real GPU utilization; fall back gracefully
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_system_metrics(device: torch.device) -> dict[str, float]:
    """Collect CPU and GPU metrics including real SM utilization."""
    metrics = {
        "system/cpu_percent": psutil.cpu_percent(),
        "system/ram_percent": psutil.virtual_memory().percent,
    }
    if device.type == "cuda":
        idx = device.index or 0
        mem_alloc = torch.cuda.memory_allocated(idx) / 1e9
        mem_reserved = torch.cuda.memory_reserved(idx) / 1e9
        mem_total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        metrics["system/gpu_mem_allocated_gb"] = round(mem_alloc, 2)
        metrics["system/gpu_mem_reserved_gb"] = round(mem_reserved, 2)
        metrics["system/gpu_mem_total_gb"] = round(mem_total, 2)
        metrics["system/gpu_mem_percent"] = round(mem_reserved / mem_total * 100, 1)

        if _HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["system/gpu_utilization_percent"] = util.gpu
                metrics["system/gpu_mem_utilization_percent"] = util.memory
            except Exception:
                pass
    return metrics


def _build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    # Separate weight decay for non-bias/norm params
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=cfg.max_steps,
            pct_start=cfg.warmup_steps / cfg.max_steps,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


class Trainer:
    """Handles training loop, evaluation, checkpointing, and MLflow logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        tokenizer=None,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        self.optimizer = _build_optimizer(model, cfg.train)
        self.scheduler = _build_scheduler(self.optimizer, cfg.train)

        self.use_amp = cfg.train.precision in ("bf16", "fp16") and self.device.type == "cuda"
        # bf16 doesn't need loss scaling; fp16 does
        self.use_scaler = cfg.train.precision == "fp16" and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_scaler)
        self.amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.tokens_seen = 0

        # Early stopping
        self.patience = getattr(cfg.train, "patience", 10)
        self._evals_without_improvement = 0

        # Throughput tracking
        self._step_start_time = None
        self._tokens_in_batch = 0

        # Running averages for smoother logging
        self._loss_sum = 0.0
        self._loss_count = 0

        # Training history for artifact export
        self._history: list[dict] = []

        checkpoint_dir = Path(cfg.train.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def _setup_mlflow(self) -> None:
        # Disable MLflow's automatic system metrics to avoid duplicates
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        run_name = self.cfg.mlflow.run_name or f"{self.cfg.model.arch}_{self.global_step}"
        mlflow.start_run(run_name=run_name)

        # Log full config as params
        flat_cfg = dict(OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True))
        mlflow.log_params(_flatten_dict(flat_cfg))

        # Log tags
        mlflow.set_tags({
            "architecture": self.cfg.model.arch,
            "git_commit": _get_git_commit(),
            "total_params": str(_count_params(self.model)),
            "dataset_size": str(len(self.train_loader.dataset)),
            "max_seq_len": str(self.cfg.data.max_seq_len),
        })

        # Log config as artifact
        config_path = self.checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        mlflow.log_artifact(str(config_path))

    def train(self) -> float:
        """Run the full training loop. Returns best validation loss."""
        self._setup_mlflow()

        if self.cfg.train.resume_from:
            self._load_checkpoint(self.cfg.train.resume_from)

        log.info(
            f"Training {self.cfg.model.arch} | "
            f"{_count_params(self.model):,} params | "
            f"device={self.device}"
        )

        train_iter = iter(self.train_loader)
        self.model.train()
        epoch_start = time.time()

        try:
            while self.global_step < self.cfg.train.max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                self._step_start_time = time.time()

                # Count non-padding tokens in this batch
                self._tokens_in_batch = int(batch["attention_mask"].sum().item())

                loss, grad_norm, token_acc = self._train_step(batch)
                self.global_step += 1
                self.tokens_seen += self._tokens_in_batch

                step_time = time.time() - self._step_start_time
                tokens_per_sec = self._tokens_in_batch / step_time

                # Running average
                self._loss_sum += loss
                self._loss_count += 1

                # Eval — check first so we don't double-log train metrics
                is_eval_step = self.global_step % self.cfg.train.eval_every == 0
                is_log_step = self.global_step % self.cfg.train.log_every == 0

                if is_eval_step:
                    avg_train_loss = self._loss_sum / self._loss_count
                    avg_train_ppl = compute_perplexity(avg_train_loss)

                    val_loss, val_ppl, val_acc = self._evaluate()

                    metrics = {
                        "train/loss": loss,
                        "train/loss_avg": avg_train_loss,
                        "train/perplexity": compute_perplexity(loss),
                        "train/perplexity_avg": avg_train_ppl,
                        "train/token_accuracy": token_acc,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": self.tokens_seen,
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/token_accuracy": val_acc,
                        "val/train_gap": val_loss - avg_train_loss,
                    }
                    metrics.update(_get_system_metrics(self.device))
                    mlflow.log_metrics(metrics, step=self.global_step)

                    log.info(
                        f"step={self.global_step} loss={loss:.4f} avg_loss={avg_train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_acc={val_acc:.3f} "
                        f"gap={val_loss - avg_train_loss:.4f} tok/s={tokens_per_sec:.0f}"
                    )

                    self._history.append({
                        "step": self.global_step,
                        "train_loss": round(loss, 5),
                        "train_loss_avg": round(avg_train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "val_ppl": round(val_ppl, 3),
                        "val_acc": round(val_acc, 4),
                        "train_val_gap": round(val_loss - avg_train_loss, 5),
                        "lr": self.scheduler.get_last_lr()[0],
                        "tokens_seen": self.tokens_seen,
                    })

                    # Reset running averages after eval
                    self._loss_sum = 0.0
                    self._loss_count = 0

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._evals_without_improvement = 0
                        self._save_checkpoint("best.pt")
                        mlflow.log_artifact(str(self.checkpoint_dir / "best.pt"))
                    else:
                        self._evals_without_improvement += 1

                    # Generate samples at eval time
                    if self.tokenizer is not None:
                        self._log_generation_samples()

                    self.model.train()

                    # Early stopping
                    if self.patience > 0 and self._evals_without_improvement >= self.patience:
                        log.info(
                            f"Early stopping: no improvement for {self.patience} evals "
                            f"(best val_loss={self.best_val_loss:.4f})"
                        )
                        break

                elif is_log_step:
                    lr = self.scheduler.get_last_lr()[0]
                    train_ppl = compute_perplexity(loss)
                    avg_loss = self._loss_sum / self._loss_count
                    metrics = {
                        "train/loss": loss,
                        "train/loss_avg": avg_loss,
                        "train/perplexity": train_ppl,
                        "train/token_accuracy": token_acc,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": self.tokens_seen,
                    }
                    # Log system metrics less frequently (every 10th log step)
                    if self.global_step % (self.cfg.train.log_every * 10) == 0:
                        metrics.update(_get_system_metrics(self.device))
                    mlflow.log_metrics(metrics, step=self.global_step)

                    log.info(
                        f"step={self.global_step} loss={loss:.4f} ppl={train_ppl:.2f} "
                        f"acc={token_acc:.3f} lr={lr:.2e} grad_norm={grad_norm:.3f} "
                        f"tok/s={tokens_per_sec:.0f}"
                    )

                # Periodic checkpoint
                if self.global_step % self.cfg.train.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")

        finally:
            # Log final summary metrics
            mlflow.log_metrics({
                "summary/best_val_loss": self.best_val_loss,
                "summary/best_val_ppl": compute_perplexity(self.best_val_loss),
                "summary/total_tokens": self.tokens_seen,
                "summary/total_steps": self.global_step,
                "summary/wall_time_hours": round((time.time() - epoch_start) / 3600, 2),
            })

            # Save and log training history CSV
            self._save_history()

            mlflow.end_run()

        return self.best_val_loss

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

        # Compute token-level accuracy before backward
        with torch.no_grad():
            preds = outputs["logits"][:, :-1, :].argmax(dim=-1)
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            if mask.any():
                token_acc = (preds[mask] == labels[mask]).float().mean().item()
            else:
                token_acc = 0.0

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip
        ).item()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), grad_norm, token_acc

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            total_loss += outputs["loss"].item()

            # Token accuracy
            preds = outputs["logits"][:, :-1, :].argmax(dim=-1)
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            if mask.any():
                total_correct += (preds[mask] == labels[mask]).sum().item()
                total_tokens += mask.sum().item()

            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = compute_perplexity(avg_loss)
        acc = total_correct / max(total_tokens, 1)
        return avg_loss, ppl, acc

    @torch.no_grad()
    def _log_generation_samples(self, n_samples: int = 3) -> None:
        """Generate sample sequences and log as MLflow artifact."""
        if self.tokenizer is None:
            return

        self.model.eval()

        # Grab a few prompts from the val set
        val_dataset = self.val_loader.dataset
        indices = list(range(min(n_samples, len(val_dataset))))

        samples_path = self.checkpoint_dir / f"samples_step{self.global_step}.txt"
        with open(samples_path, "w") as f:
            f.write(f"=== Generation samples at step {self.global_step} ===\n\n")
            for i in indices:
                item = val_dataset[i]
                input_ids = item["input_ids"]

                # Find the SEP token to get just the prompt
                sep_id = self.tokenizer.sep_token_id
                ids_list = input_ids.tolist()
                try:
                    sep_pos = ids_list.index(sep_id)
                except ValueError:
                    continue

                prompt_ids = ids_list[:sep_pos + 1]
                prompt_text = self.tokenizer.decode(ids_list[:sep_pos])

                # True completion (from sep to first pad or end)
                pad_id = self.tokenizer.pad_token_id
                true_ids = ids_list[sep_pos + 1:]
                if pad_id in true_ids:
                    true_ids = true_ids[:true_ids.index(pad_id)]
                true_text = self.tokenizer.decode(true_ids)

                # Generate
                input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

                # Cast model for generation if needed (mamba requires matching dtypes)
                try:
                    output_ids = self.model.generate(
                        input_tensor,
                        max_new_tokens=min(512, self.cfg.data.max_seq_len),
                        temperature=0.8,
                    )
                    gen_ids = output_ids[0, len(prompt_ids):].tolist()
                    gen_text = self.tokenizer.decode(gen_ids)
                except Exception as e:
                    gen_text = f"[generation failed: {e}]"

                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"Prompt:    {prompt_text[:200]}\n")
                f.write(f"True:      {true_text[:300]}...\n")
                f.write(f"Generated: {gen_text[:300]}...\n")
                f.write(f"True len:  {len(true_text)} | Gen len: {len(gen_text)}\n\n")

        mlflow.log_artifact(str(samples_path))

    def _save_history(self) -> None:
        """Save training history as CSV artifact."""
        if not self._history:
            return
        history_path = self.checkpoint_dir / "training_history.csv"
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._history[0].keys())
            writer.writeheader()
            writer.writerows(self._history)
        mlflow.log_artifact(str(history_path))

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "tokens_seen": self.tokens_seen,
                "config": OmegaConf.to_container(self.cfg),
            },
            path,
        )
        log.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.tokens_seen = ckpt.get("tokens_seen", 0)
        log.info(f"Resumed from step {self.global_step}")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict for MLflow param logging."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = str(v)
    return items
