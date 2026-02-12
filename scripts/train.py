"""Hydra entry point for training PlasmidLLM models."""

import logging

import hydra
from omegaconf import DictConfig

from plasmid_llm.config import register_configs
from plasmid_llm.data import PlasmidDataset, build_dataloaders
from plasmid_llm.models import build_model
from plasmid_llm.tokenizer import PlasmidTokenizer
from plasmid_llm.trainer import Trainer

log = logging.getLogger(__name__)

# Register structured configs with Hydra
register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    log.info(f"Architecture: {cfg.model.arch}")
    log.info(f"Config:\n{cfg}")

    # Tokenizer
    tokenizer = PlasmidTokenizer(cfg.data.vocab_path)
    log.info(f"Vocab size: {tokenizer.vocab_size}")

    # Dataset
    dataset = PlasmidDataset(
        parquet_path=cfg.data.parquet_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
    )
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=cfg.train.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.data.seed,
        num_workers=cfg.train.num_workers,
    )
    log.info(f"Dataset: {len(dataset)} samples, train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    # Model
    model = build_model(cfg.model, tokenizer.vocab_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {n_params:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, cfg)
    best_val_loss = trainer.train()

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    main()
