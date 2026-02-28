"""Ray-based distributed post-training for PlasmidLM.

Usage:
    python scripts/train_ray.py configs/ray_g6big.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig
from post_training.ray.orchestrator import Orchestrator
from plasmid_llm.utils import load_config, load_env_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train PlasmidLM with Ray-based RL")
    parser.add_argument("config", type=Path, help="Path to Python config file")
    args = parser.parse_args()

    # Load .env for Databricks/MLflow credentials
    load_env_file()

    # Load config
    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    if not isinstance(config, RayPostTrainingConfig):
        raise TypeError(
            f"Config must be RayPostTrainingConfig, got {type(config).__name__}"
        )
    log.info(f"Config loaded: checkpoint={config.model_checkpoint.name}, "
             f"algorithm={config.algorithm}, max_steps={config.max_steps}")

    # Run training
    orchestrator = Orchestrator(config)
    orchestrator.train()


if __name__ == "__main__":
    main()
