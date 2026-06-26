# dataset_builder/config.py
"""Configuration loading and access."""

import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


@dataclass
class DatasetConfig:
    """Typed access to the YAML configuration."""
    raw: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw["meta"]["name"]

    @property
    def task(self) -> str:
        return self.raw["meta"]["task"]

    @property
    def fps(self) -> int:
        return self.raw["meta"]["fps"]
    
    @property
    def kinematics(self) -> Dict:
        return self.raw.get("kinematics", {})

    @property
    def cache(self) -> Dict:
        return self.raw.get("cache", {})

    @property
    def episodes(self) -> Dict:
        return self.raw.get("episodes", {})

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw["cache"]["base_dir"])

    @property
    def cache_flags(self) -> Dict[str, bool]:
        cache = self.raw["cache"]
        return {
            "front_rgb": cache.get("front_rgb", False),
            "wrist_rgb": cache.get("wrist_rgb", False),
            "front_pc": cache.get("front_pc", False),
            "wrist_pc": cache.get("wrist_pc", False),
        }

    @property
    def pc_config(self) -> Dict:
        return self.raw["point_cloud"]

    @property
    def proprio_config(self) -> Dict:
        return self.raw["proprioception"]

    @property
    def hand_config(self) -> Dict:
        return self.proprio_config.get("hand", {})

    @property
    def action_config(self) -> Dict:
        return self.raw["actions"]

    @property
    def progress_config(self) -> Dict:
        return self.raw.get("progress", {"enabled": False})

    @property
    def post_processing_config(self) -> Dict:
        return self.raw.get("post_processing", {})

    @property
    def output_config(self) -> Dict:
        return self.raw["output"]
    
    @property
    def force_config(self) -> Dict:
        return self.hand_config.get("force", {})

    @property
    def aux_action_config(self) -> list:
        return self.raw.get("auxiliary_actions", [])

def load_config(config_path: str) -> DatasetConfig:
    """Load YAML config from disk and return typed wrapper."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # TODO: Add schema validation (pydantic or jsonschema)
    return DatasetConfig(raw=raw)