"""Structured Hydra configs for PlasmidLLM."""

from dataclasses import dataclass, field
from typing import Any

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
    patience: int = 10
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""
    loss_type: str = "cross_entropy"      # "cross_entropy" | "focal"
    focal_gamma: float = 2.0              # focal loss focusing parameter
    label_smoothing: float = 0.0          # label smoothing factor


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
class Transformer500MConfig:
    arch: str = "transformer"
    d_model: int = 1280
    n_layers: int = 32
    n_heads: int = 16
    d_ff: int = 5120
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
class TransformerFiLMConfig:
    arch: str = "transformer_film"
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    sep_token_id: int = 3


@dataclass
class TransformerConvConfig:
    arch: str = "transformer_conv"
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 8
    d_ff: int = 1536
    conv_kernel_size: int = 7
    dropout: float = 0.1


@dataclass
class TransformerConvLargeConfig:
    arch: str = "transformer_conv"
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    conv_kernel_size: int = 7
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
class DiffusionSmallConfig:
    arch: str = "diffusion"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    timesteps: int = 1000
    noise_schedule: str = "cosine"
    loss_type: str = "cross_entropy"
    dropout: float = 0.1


@dataclass
class EncoderDecoderConfig:
    arch: str = "encoder_decoder"
    d_model: int = 384
    n_enc_layers: int = 4      # Shallow - just encoding short metadata
    n_dec_layers: int = 12     # Deep - generating 8000 DNA tokens
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    sep_token_id: int = 3
    max_seq_len: int = 8192


@dataclass
class PlasmidLLMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: Any = field(default_factory=TransformerConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    # Don't register top-level schema — it prevents cross-architecture model switching
    cs.store(group="data", name="default", node=DataConfig)
    cs.store(group="train", name="default", node=TrainConfig)
    cs.store(group="model", name="transformer", node=TransformerConfig)
    cs.store(group="model", name="transformer_large", node=TransformerLargeConfig)
    cs.store(group="model", name="transformer_500m", node=Transformer500MConfig)
    cs.store(group="model", name="mamba", node=MambaConfig)
    cs.store(group="model", name="transformer_film", node=TransformerFiLMConfig)
    cs.store(group="model", name="transformer_conv", node=TransformerConvConfig)
    cs.store(group="model", name="transformer_conv_large", node=TransformerConvLargeConfig)
    cs.store(group="model", name="diffusion", node=DiffusionConfig)
    cs.store(group="model", name="diffusion_small", node=DiffusionSmallConfig)
    cs.store(group="model", name="encoder_decoder", node=EncoderDecoderConfig)
    cs.store(group="mlflow", name="databricks", node=MlflowConfig)
