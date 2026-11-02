"""Load config/config.yaml once and resolve project-relative paths."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel: str) -> Path:
    """Resolve a config path relative to the project root."""
    return ROOT / rel
