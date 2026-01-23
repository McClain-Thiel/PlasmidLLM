from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class EnvConfig:
    catalog: str
    schemas: Dict[str, str]
    volumes: Dict[str, str]
    tables: Dict[str, str]


@dataclass(frozen=True)
class TokenConfig:
    thresholds: Dict[str, int]
    critical_amr: list[str]
    meta_tokens: list[dict]
    tagging: Dict[str, int]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_env_config(env_yaml_path: str = "config/env.yaml") -> EnvConfig:
    raw = load_yaml(env_yaml_path)
    return EnvConfig(
        catalog=raw["catalog"],
        schemas=raw["schemas"],
        volumes=raw["volumes"],
        tables=raw["tables"],
    )


def load_token_config(token_yaml_path: str = "config/token_config.yaml") -> TokenConfig:
    raw = load_yaml(token_yaml_path)
    return TokenConfig(
        thresholds=raw["thresholds"],
        critical_amr=raw["critical_amr"],
        meta_tokens=raw["meta_tokens"],
        tagging=raw["tagging"],
    )

