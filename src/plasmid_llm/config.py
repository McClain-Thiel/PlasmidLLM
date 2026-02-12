"""Structured Hydra configs for PlasmidLLM."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    parquet_path: str = "s3://phd-research-storage-1758274488/addgene_clean/tokenization/training_pairs.parquet"
    vocab_path: str = "s3://phd-research-storage-1758274488/addgene_clean/tokenization/token_vocabulary.json"
    max_seq_len: int = 4096
    val_split: float = 0.05
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    eval_every: int = 500
    save_every: int = 5000
    log_every: int = 10
    num_workers: int = 4
    precision: str = "bf16"
    scheduler: str = "cosine"
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""


@dataclass
class MlflowConfig:
    tracking_uri: str = "databricks"
    experiment_name: str = "/Users/${oc.env:USER,default}/PlasmidLLM"
    run_name: str = ""


@dataclass
class TransformerConfig:
    arch: str = "transformer"
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1


@dataclass
class TransformerLargeConfig:
    arch: str = "transformer"
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.1


@dataclass
class MambaConfig:
    arch: str = "mamba"
    d_model: int = 256
    n_layers: int = 16
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class DiffusionConfig:
    arch: str = "diffusion"
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    timesteps: int = 1000
    noise_schedule: str = "cosine"
    loss_type: str = "cross_entropy"
    dropout: float = 0.1


@dataclass
class PlasmidLLMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: Any = field(default_factory=TransformerConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=PlasmidLLMConfig)
    cs.store(group="data", name="default", node=DataConfig)
    cs.store(group="train", name="default", node=TrainConfig)
    cs.store(group="model", name="transformer", node=TransformerConfig)
    cs.store(group="model", name="transformer_large", node=TransformerLargeConfig)
    cs.store(group="model", name="mamba", node=MambaConfig)
    cs.store(group="model", name="diffusion", node=DiffusionConfig)
    cs.store(group="mlflow", name="databricks", node=MlflowConfig)
