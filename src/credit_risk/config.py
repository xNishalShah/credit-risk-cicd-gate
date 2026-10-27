from __future__ import annotations

from pathlib import Path
import yaml

REQUIRED = {
    "environment", "data", "training", "features", "tracking", "quality_gate"
}


def load_config(path: str | Path) -> dict:
    """Load and fail fast on the settings that make a run reproducible."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    missing = REQUIRED - set(config or {})
    if missing:
        raise ValueError(f"Invalid config: missing top-level keys: {sorted(missing)}")
    if config["training"]["test_size"] <= 0 or config["training"]["test_size"] >= 1:
        raise ValueError("Invalid config: training.test_size must be between 0 and 1")
    if not 0 <= config["features"]["utilization_high_cutoff"] <= 1:
        raise ValueError("Invalid config: utilization_high_cutoff must be between 0 and 1")
    if config["quality_gate"]["metric"] != "pr_auc":
        raise ValueError("Invalid config: only pr_auc is an approved primary gate metric")
    return config
