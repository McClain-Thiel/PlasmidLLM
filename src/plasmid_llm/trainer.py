"""Unified training loop with MLflow logging for all architectures."""

from __future__ import annotations

import logging
import os
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


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = _build_optimizer(model, cfg.train)
        self.scheduler = _build_scheduler(self.optimizer, cfg.train)

        self.use_amp = cfg.train.precision in ("bf16", "fp16") and self.device.type == "cuda"
        # bf16 doesn't need loss scaling; fp16 does
        self.use_scaler = cfg.train.precision == "fp16" and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_scaler)
        self.amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16

        self.global_step = 0
        self.best_val_loss = float("inf")

        checkpoint_dir = Path(cfg.train.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def _setup_mlflow(self) -> None:
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

        try:
            while self.global_step < self.cfg.train.max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss, grad_norm = self._train_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.cfg.train.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    mlflow.log_metrics(
                        {
                            "train/loss": loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                        },
                        step=self.global_step,
                    )
                    log.info(
                        f"step={self.global_step} loss={loss:.4f} lr={lr:.2e} "
                        f"grad_norm={grad_norm:.3f}"
                    )

                # Eval
                if self.global_step % self.cfg.train.eval_every == 0:
                    val_loss, val_ppl = self._evaluate()
                    mlflow.log_metrics(
                        {"val/loss": val_loss, "val/perplexity": val_ppl},
                        step=self.global_step,
                    )
                    log.info(f"  val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best.pt")
                        mlflow.log_artifact(str(self.checkpoint_dir / "best.pt"))

                    self.model.train()

                # Periodic checkpoint
                if self.global_step % self.cfg.train.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")

        finally:
            mlflow.log_metric("best_val_loss", self.best_val_loss)
            mlflow.end_run()

        return self.best_val_loss

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip
        ).item()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), grad_norm

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
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
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = compute_perplexity(avg_loss)
        return avg_loss, ppl

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
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
