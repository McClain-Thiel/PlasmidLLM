"""Fresh kmer6 pretraining; requires a successful independent split audit."""

import json
import os
from pathlib import Path

from plasmid_llm.config import PretrainingConfig
from pretraining.build_cluster_split import sha256_file

ROOT = Path(__file__).resolve().parents[2]
split = Path(os.environ["PLASMID_SPLIT_DIR"])
audit = json.loads(Path(os.environ["PLASMID_SPLIT_AUDIT"]).read_text())
if not audit["passed"]:
    raise ValueError("Cluster partition audit did not pass")
for name in ("train", "validation", "test"):
    if sha256_file(split / f"{name}.parquet") != audit["partition_sha256"][name]:
        raise ValueError(f"Audit does not match {name} data")
seed = int(os.environ.get("PLASMID_SEED", "42"))
name = f"{os.environ['PLASMID_RUN_ID']}_pretrain_s{seed}"
microbatch = int(os.environ.get("PLASMID_PRETRAIN_MICROBATCH", "8"))
if microbatch < 1 or 64 % microbatch:
    raise ValueError("Pretraining microbatch must divide the fixed batch of 64")

config = PretrainingConfig(
    training_pairs=split / "train.parquet", validation_pairs=split / "validation.parquet",
    special_tokens=ROOT / "data" / "special_tokens.txt",
    tokenizer_type="kmer", kmer_k=6, kmer_stride=3,
    hidden_size=384, num_hidden_layers=10, num_attention_heads=8, intermediate_size=1536,
    max_seq_len=4096, circular_rotation=True, soft_token_dropout=0.2,
    minority_target_fraction=0.1, minority_max_weight=4.0,
    output_dir=Path("checkpoints") / name, per_device_train_batch_size=microbatch,
    gradient_accumulation_steps=64 // microbatch, learning_rate=3e-4,
    max_steps=int(os.environ.get("PLASMID_STEPS", "100000")), warmup_steps=1000,
    eval_steps=int(os.environ.get("PLASMID_EVAL_EVERY", "500")),
    save_steps=int(os.environ.get("PLASMID_SAVE_EVERY", "5000")), save_total_limit=None,
    early_stopping_patience=None, load_best_model_at_end=False,
    bf16=True, seed=seed, wandb_project="PlasmidLLM", wandb_run_name=name,
)
