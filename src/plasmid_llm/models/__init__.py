"""Model registry for PlasmidLLM architectures."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def build_model(config: Any, vocab_size: int, loss_fn: nn.Module | None = None) -> nn.Module:
    """Instantiate a model from config. Config must have an `arch` field."""
    arch = config.arch if hasattr(config, "arch") else config["arch"]
    if arch not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")
    return MODEL_REGISTRY[arch](config, vocab_size, loss_fn=loss_fn)


# Import model modules to trigger registration
from plasmid_llm.models import transformer  # noqa: F401, E402
from plasmid_llm.models import diffusion  # noqa: F401, E402
from plasmid_llm.models import transformer_variants  # noqa: F401, E402
from plasmid_llm.models import encoder_decoder  # noqa: F401, E402

# Optional mamba import (requires mamba-ssm package)
try:
    from plasmid_llm.models import mamba  # noqa: F401, E402
except ImportError:
    pass
