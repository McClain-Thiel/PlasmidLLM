"""Rejection Sampling Fine-Tuning (RFT / Expert Iteration) for PlasmidLM.

Iterative approach:
  1. Generate completions for all prompts using the current model
  2. Score each with motif alignment reward
  3. Keep only high-reward completions (above threshold)
  4. SFT on the filtered (prompt, completion) pairs
  5. Repeat for N iterations

This avoids GRPO's requirement for within-group diversity — the model just needs
to produce some good completions across the full prompt set.

Usage:
    python scripts/train_rft.py configs/rft_g6big.py
"""

from __future__ import annotations

import argparse
import gc
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
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plasmid_llm.models.hf_plasmid_lm import (
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMTokenizer,
)
from post_training.reward import load_motif_lookup, plasmid_reward_fn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RFTConfig:
    """Configuration for RFT post-training."""

    # Data paths
    model_checkpoint: Path
    training_pairs: Path
    motif_lookup: Path

    # RFT iterations
    num_rft_iterations: int = 5       # outer loop iterations
    reward_threshold: float = 0.15    # keep completions with reward >= this

    # Generation
    gen_batch_size: int = 8           # batch size for generation
    max_completion_length: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    num_samples_per_prompt: int = 1   # completions per prompt (1 = fast, >1 = more coverage)
    max_prompts_per_iter: int = 5000  # cap prompts per iteration to bound time

    # SFT training
    learning_rate: float = 5e-5
    sft_epochs: int = 1
    sft_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096        # tokenized prompt + completion

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output/rft"))
    save_steps: int = 200
    logging_steps: int = 10
    seed: int = 42
    bf16: bool = True

    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "plasmid_rft"

    def __post_init__(self):
        self.model_checkpoint = Path(self.model_checkpoint)
        self.training_pairs = Path(self.training_pairs)
        self.motif_lookup = Path(self.motif_lookup)
        self.output_dir = Path(self.output_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_prompts(parquet_path: str, max_prompts: int = 0) -> list[str]:
    """Load prompt strings from training pairs parquet."""
    meta = pq.ParquetFile(parquet_path).schema.names
    need_cols = []
    for c in ["prompt", "token_prompt", "reward_motifs"]:
        if c in meta:
            need_cols.append(c)

    log.info(f"Reading columns {need_cols} from {parquet_path}")
    table = pq.read_table(parquet_path, columns=need_cols)
    col_names = table.column_names

    # Filter to prompts with reward motifs
    if "reward_motifs" in col_names:
        motifs = table.column("reward_motifs").to_pylist()
        indices = [i for i, m in enumerate(motifs) if m]
        table = table.take(indices)

    # Extract prompt column
    if "prompt" in col_names:
        prompts = table.column("prompt").to_pylist()
    elif "token_prompt" in col_names:
        prompts = table.column("token_prompt").to_pylist()
    else:
        raise ValueError(f"No prompt column found. Available: {col_names}")

    # Append <SEP> for generation
    prompts = [p + "<SEP>" for p in prompts]

    if max_prompts > 0 and len(prompts) > max_prompts:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(prompts), max_prompts, replace=False)
        prompts = [prompts[i] for i in indices]

    log.info(f"Loaded {len(prompts)} prompts")
    return prompts


def prepare_checkpoint_for_vllm(checkpoint_dir: str) -> None:
    """Copy model source files and add auto_map to config.json so vLLM can load."""
    import json

    checkpoint_path = Path(checkpoint_dir)
    src_dir = Path(__file__).resolve().parent.parent / "src" / "plasmid_llm" / "models" / "hf_plasmid_lm"

    # Copy model source files
    for fname in ["configuration_plasmid_lm.py", "modeling_plasmid_lm.py", "tokenization_plasmid_lm.py"]:
        src = src_dir / fname
        dst = checkpoint_path / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            log.info(f"  Copied {fname} to checkpoint dir")

    # Add auto_map to config.json if missing
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        if "auto_map" not in cfg:
            cfg["auto_map"] = {
                "AutoConfig": "configuration_plasmid_lm.PlasmidLMConfig",
                "AutoModel": "modeling_plasmid_lm.PlasmidLMModel",
                "AutoModelForCausalLM": "modeling_plasmid_lm.PlasmidLMForCausalLM",
                "AutoTokenizer": ["tokenization_plasmid_lm.PlasmidLMTokenizer", None],
            }
            config_path.write_text(json.dumps(cfg, indent=2))
            log.info("  Added auto_map to config.json")


def generate_completions_vllm(
    model_path: str,
    tokenizer: PlasmidLMTokenizer,
    prompts: list[str],
    config: RFTConfig,
) -> list[tuple[str, str]]:
    """Generate completions using vLLM for 10-20x throughput."""
    from vllm import LLM, SamplingParams

    # Ensure checkpoint has model source files + auto_map for trust_remote_code
    prepare_checkpoint_for_vllm(model_path)

    log.info(f"Initializing vLLM with model from {model_path}...")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        max_model_len=config.max_completion_length + 512,  # prompt + completion
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_completion_length,
    )

    log.info(f"Generating {len(prompts)} completions with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for prompt, output in zip(prompts, outputs):
        completion = output.outputs[0].text
        results.append((prompt, completion))

    log.info(f"vLLM generated {len(results)} completions")

    # Free vLLM GPU memory so SFT can use it
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


@torch.no_grad()
def generate_completions_native(
    model: PlasmidLMForCausalLM,
    tokenizer: PlasmidLMTokenizer,
    prompts: list[str],
    config: RFTConfig,
) -> list[tuple[str, str]]:
    """Generate completions using native model.generate() (slower, no vLLM needed)."""
    model.eval()
    device = next(model.parameters()).device
    results = []
    import time as _time
    _gen_start = _time.time()

    for start in range(0, len(prompts), config.gen_batch_size):
        batch_prompts = prompts[start:start + config.gen_batch_size]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model.generate(
            **encoded,
            max_new_tokens=config.max_completion_length,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i, prompt in enumerate(batch_prompts):
            prompt_len = encoded["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
            completion_ids = outputs[i][prompt_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
            results.append((prompt, completion))

        batch_idx = start // config.gen_batch_size
        n_batches = (len(prompts) + config.gen_batch_size - 1) // config.gen_batch_size
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = _time.time() - _gen_start
            eta = (elapsed / max(batch_idx + 1, 1)) * (n_batches - batch_idx - 1) if elapsed > 0 else 0
            log.info(f"Generated {len(results)}/{len(prompts)} completions "
                     f"(batch {batch_idx+1}/{n_batches}, ETA {eta/60:.1f}min)")

    log.info(f"Generated {len(results)} completions total")
    return results


def score_and_filter(
    pairs: list[tuple[str, str]],
    lookup_df,
    threshold: float,
) -> tuple[list[tuple[str, str, float]], dict]:
    """Score completions and filter by reward threshold.

    Returns (filtered_pairs, stats_dict).
    """
    prompts = [p for p, _ in pairs]
    completions = [c for _, c in pairs]

    rewards = plasmid_reward_fn(prompts, completions, lookup_df)

    all_rewards = np.array(rewards)
    filtered = []
    for (prompt, completion), reward in zip(pairs, rewards):
        if reward >= threshold:
            filtered.append((prompt, completion, reward))

    # Compute percentiles for threshold tuning
    pcts = np.percentile(all_rewards, [10, 25, 50, 75, 90]) if len(all_rewards) > 0 else [0]*5
    n_zero = int((all_rewards == 0).sum())

    stats = {
        "total": len(pairs),
        "filtered": len(filtered),
        "keep_rate": len(filtered) / max(len(pairs), 1),
        "mean_reward": float(all_rewards.mean()),
        "median_reward": float(np.median(all_rewards)),
        "max_reward": float(all_rewards.max()),
        "min_reward": float(all_rewards.min()),
        "std_reward": float(all_rewards.std()),
        "mean_filtered_reward": float(np.mean([r for _, _, r in filtered])) if filtered else 0.0,
        "p10": float(pcts[0]), "p25": float(pcts[1]), "p50": float(pcts[2]),
        "p75": float(pcts[3]), "p90": float(pcts[4]),
        "n_zero": n_zero,
    }
    return filtered, stats


def build_sft_dataset(
    filtered_pairs: list[tuple[str, str, float]],
    tokenizer: PlasmidLMTokenizer,
) -> datasets.Dataset:
    """Build SFT dataset from filtered (prompt, completion, reward) triples.

    Each example is the full text: prompt + completion (tokenized together).
    SFT loss is computed on all tokens (prompt included — matches pretrain objective).
    """
    texts = []
    for prompt, completion, _ in filtered_pairs:
        # The model was pretrained on "prompt<SEP>completion<EOS>"
        # The prompt already has <SEP>, so just concatenate
        full = prompt + completion
        # Ensure it ends with <EOS> if not already
        if not full.rstrip().endswith("<EOS>"):
            full = full.rstrip() + "<EOS>"
        texts.append(full)

    return datasets.Dataset.from_dict({"text": texts})


# ── Main ──────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> RFTConfig:
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
    parser = argparse.ArgumentParser(description="Train PlasmidLM with RFT")
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

    config = load_config(args.config)
    log.info(f"RFT Config: {config.num_rft_iterations} iterations, threshold={config.reward_threshold}")

    # Setup MLflow
    mlflow_enabled = False
    if config.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            exp = mlflow.set_experiment(config.mlflow_experiment)
            if exp is None:
                exp_id = mlflow.create_experiment(config.mlflow_experiment)
                mlflow.set_experiment(experiment_id=exp_id)
            mlflow_enabled = True
            log.info(f"MLflow: {config.mlflow_tracking_uri} / {config.mlflow_experiment}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # Check vLLM availability
    use_vllm = False
    try:
        import vllm
        use_vllm = True
        log.info("vLLM available — using for generation (10-20x faster)")
    except ImportError:
        log.info("vLLM not available — using native model.generate()")

    # Load model and tokenizer
    log.info(f"Loading model from {config.model_checkpoint}")
    model = PlasmidLMForCausalLM.from_pretrained(str(config.model_checkpoint))
    tokenizer = PlasmidLMTokenizer.from_pretrained(str(config.model_checkpoint))
    tokenizer.padding_side = "left"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} params")

    # Load motif lookup for reward scoring
    log.info(f"Loading motif lookup from {config.motif_lookup}")
    lookup_df = load_motif_lookup(str(config.motif_lookup))
    log.info(f"Motif lookup: {len(lookup_df)} entries")

    # Load all prompts
    prompts = load_prompts(str(config.training_pairs), max_prompts=config.max_prompts_per_iter)

    # Track model checkpoint path for each iteration
    current_checkpoint = config.model_checkpoint

    if mlflow_enabled:
        import mlflow
        mlflow.start_run(run_name=f"rft_{_get_git_commit()}")
        mlflow.log_params({
            "model_checkpoint": str(config.model_checkpoint),
            "num_rft_iterations": config.num_rft_iterations,
            "reward_threshold": config.reward_threshold,
            "max_completion_length": config.max_completion_length,
            "temperature": config.temperature,
            "num_samples_per_prompt": config.num_samples_per_prompt,
            "learning_rate": config.learning_rate,
            "sft_epochs": config.sft_epochs,
            "git_commit": _get_git_commit(),
        })

    for iteration in range(1, config.num_rft_iterations + 1):
        log.info(f"\n{'='*60}")
        log.info(f"RFT ITERATION {iteration}/{config.num_rft_iterations}")
        log.info(f"{'='*60}")

        # Shuffle prompts each iteration
        rng = np.random.RandomState(config.seed + iteration)
        iter_prompts = list(prompts)
        rng.shuffle(iter_prompts)

        # Phase 1: Generate completions
        log.info(f"Phase 1: Generating completions for {len(iter_prompts)} prompts...")
        all_pairs = []
        for sample_idx in range(config.num_samples_per_prompt):
            if use_vllm:
                # Move model to CPU so vLLM can claim GPU
                model.to("cpu")
                torch.cuda.empty_cache()
                pairs = generate_completions_vllm(
                    str(current_checkpoint), tokenizer, iter_prompts, config
                )
            else:
                model.to("cuda" if torch.cuda.is_available() else "cpu")
                pairs = generate_completions_native(model, tokenizer, iter_prompts, config)
            all_pairs.extend(pairs)
            if config.num_samples_per_prompt > 1:
                log.info(f"  Sample {sample_idx+1}/{config.num_samples_per_prompt}: {len(pairs)} completions")

        # Phase 2: Score and filter
        log.info(f"Phase 2: Scoring {len(all_pairs)} completions...")
        filtered, stats = score_and_filter(all_pairs, lookup_df, config.reward_threshold)

        log.info(f"  Reward distribution: mean={stats['mean_reward']:.4f}, "
                 f"median={stats['median_reward']:.4f}, "
                 f"max={stats['max_reward']:.4f}, std={stats['std_reward']:.4f}")
        log.info(f"  Percentiles: p10={stats['p10']:.4f} p25={stats['p25']:.4f} "
                 f"p50={stats['p50']:.4f} p75={stats['p75']:.4f} p90={stats['p90']:.4f}")
        log.info(f"  Zero-reward: {stats['n_zero']}/{stats['total']}")
        log.info(f"  Kept {stats['filtered']}/{stats['total']} completions "
                 f"({stats['keep_rate']:.1%}) above threshold {config.reward_threshold}")
        log.info(f"  Filtered mean reward: {stats['mean_filtered_reward']:.4f}")

        if mlflow_enabled:
            mlflow.log_metrics({
                f"rft/mean_reward": stats["mean_reward"],
                f"rft/max_reward": stats["max_reward"],
                f"rft/std_reward": stats["std_reward"],
                f"rft/keep_rate": stats["keep_rate"],
                f"rft/filtered_mean_reward": stats["mean_filtered_reward"],
                f"rft/num_filtered": stats["filtered"],
            }, step=iteration)

        if len(filtered) < 10:
            log.warning(f"Only {len(filtered)} examples passed threshold — "
                        f"lowering threshold to median ({stats['median_reward']:.4f})")
            filtered, stats = score_and_filter(all_pairs, lookup_df, stats["median_reward"])
            log.info(f"  Re-filtered: {len(filtered)} examples")

        if len(filtered) == 0:
            log.error("No examples passed filter — skipping SFT")
            continue

        # Phase 3: SFT on filtered data
        log.info(f"Phase 3: SFT on {len(filtered)} high-reward examples...")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        sft_dataset = build_sft_dataset(filtered, tokenizer)

        iter_output = config.output_dir / f"iter_{iteration}"
        sft_config = SFTConfig(
            output_dir=str(iter_output),
            num_train_epochs=config.sft_epochs,
            per_device_train_batch_size=config.sft_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_length=config.max_seq_length,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            bf16=config.bf16,
            seed=config.seed,
            report_to="mlflow" if mlflow_enabled else "none",
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=sft_dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        log.info(f"  SFT loss: {train_result.training_loss:.4f}")

        if mlflow_enabled:
            mlflow.log_metric("rft/sft_loss", train_result.training_loss, step=iteration)

        # Save iteration checkpoint
        checkpoint_dir = iter_output / "checkpoint"
        trainer.save_model(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

        # Copy vocab, special tokens, and config for self-containment
        # (vLLM needs config.json + vocab.json to reload model in next iteration)
        for src_dir in [config.model_checkpoint, current_checkpoint]:
            vocab_src = Path(src_dir) / "vocab.json"
            if vocab_src.exists():
                shutil.copy2(vocab_src, checkpoint_dir / "vocab.json")
                break
        st_src = Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt"
        if st_src.exists():
            shutil.copy2(st_src, checkpoint_dir / "special_tokens.txt")
        # Save model config (HF format) — needed by vLLM's AutoModel loader
        if hasattr(model, "config"):
            model.config.save_pretrained(str(checkpoint_dir))

        log.info(f"  Saved checkpoint to {checkpoint_dir}")
        current_checkpoint = checkpoint_dir

        if mlflow_enabled:
            try:
                mlflow.log_artifacts(str(checkpoint_dir), artifact_path=f"rft_iter_{iteration}")
            except Exception as e:
                log.warning(f"MLflow artifact upload failed: {e}")

        # Clean up for next iteration
        del trainer, sft_dataset
        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"\nRFT complete! Final model at {current_checkpoint}")

    if mlflow_enabled:
        mlflow.end_run()


if __name__ == "__main__":
    main()
