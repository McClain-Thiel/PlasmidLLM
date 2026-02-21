"""GRPO training script for PlasmidLM post-training with sequence alignment rewards.

Usage:
    python scripts/train_grpo.py configs/grpo_example.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plasmid_llm.config import PostTrainingConfig
from src.plasmid_llm.data import PlasmidPromptsDataset
from src.plasmid_llm.models.hf_plasmid_lm import PlasmidLMForCausalLM, PlasmidLMTokenizer
from post_training.reward import load_motif_lookup, plasmid_reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class GRPOLineageCallback(TrainerCallback):
    """Log GRPO config and data lineage to MLflow."""

    def __init__(self, config: PostTrainingConfig):
        self.config = config
        self.logged_params = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or self.logged_params:
            return
        
        try:
            import mlflow

            # Log all config params
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

    # Load config
    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    log.info(f"Config loaded: {config.model_checkpoint.name}")

    # Load model and tokenizer
    log.info(f"Loading model from {config.model_checkpoint}")
    model = PlasmidLMForCausalLM.from_pretrained(str(config.model_checkpoint))
    tokenizer = PlasmidLMTokenizer.from_pretrained(str(config.model_checkpoint))
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model loaded: {n_params:,} trainable params")

    # Load motif lookup for reward function
    log.info(f"Loading motif lookup from {config.motif_lookup}")
    lookup_df = load_motif_lookup(str(config.motif_lookup))
    log.info(f"Motif lookup loaded: {len(lookup_df)} entries, {len(lookup_df.index.unique())} unique tokens")

    # Load prompts dataset
    log.info(f"Loading prompts from {config.training_pairs}")
    dataset = PlasmidPromptsDataset(
        str(config.training_pairs),
        tokenizer,
        filter_hard_tokens=True,
    )
    log.info(f"Dataset loaded: {len(dataset)} prompts with hard tokens")

    # Setup MLflow
    report_to = "none"
    if config.mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        report_to = "mlflow"
        log.info(f"MLflow: {config.mlflow_tracking_uri} / {config.mlflow_experiment}")

    # Define reward function wrapper
    def reward_fn(samples: list[str], prompts: list[str], **kwargs) -> list[float]:
        """
        Reward function for GRPO.
        
        Args:
            samples: Generated sequences (completions)
            prompts: Original prompts
        
        Returns:
            List of rewards in [0, 1]
        """
        return plasmid_reward_fn(prompts, samples, lookup_df)

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps if config.max_steps else -1,
        
        # Sampling parameters
        num_generations=config.num_generations_per_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        
        # GRPO-specific
        num_ppo_epochs=config.num_ppo_epochs,
        kl_coef=config.kl_coef,
        clip_range=config.clip_range,
        vf_coef=config.vf_coef,
        
        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        report_to=report_to,
        
        # System
        bf16=config.bf16,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
    )

    # Create GRPO trainer
    log.info("Initializing GRPO trainer...")
    callbacks = []
    if config.mlflow_tracking_uri:
        callbacks.append(GRPOLineageCallback(config))
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_function=reward_fn,
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
