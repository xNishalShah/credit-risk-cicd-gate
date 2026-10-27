"""Run the v1/v2 data drill at three seeds and write slide-ready statistics."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from credit_risk.train import run


def execute(base: dict, root: Path, label: str, file_name: str, seeds: list[int]) -> list[float]:
    scores = []
    for seed in seeds:
        config = deepcopy(base)
        config["data"]["path"] = f"data/raw/{file_name}"
        config["data"]["version"] = f"credit-card-{label}"
        config["training"]["random_seed"] = seed
        config_path = root / "artifacts" / "generated_configs" / f"{label}_seed_{seed}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        result = run(str(config_path), f"artifacts/data_drill/{label}_seed_{seed}.json")
        scores.append(result["best_run"]["pr_auc"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(args.config).resolve().parents[1]
    base = yaml.safe_load(Path(args.config).read_text())
    v1 = execute(base, root, "v1", "credit_card_customers_v1.csv", args.seeds)
    v2 = execute(base, root, "v2", "credit_card_customers_v2.csv", args.seeds)
    mean_v1, mean_v2 = float(np.mean(v1)), float(np.mean(v2))
    rules = {"no_drop": 0.00, "recommended_minus_0_01": -0.01, "permissive_minus_0_03": -0.03}
    summary = {
        "metric": "pr_auc",
        "v1": {"scores": v1, "mean": mean_v1, "sample_std": float(np.std(v1, ddof=1))},
        "v2": {"scores": v2, "mean": mean_v2, "sample_std": float(np.std(v2, ddof=1))},
        "mean_delta_v2_minus_v1": mean_v2 - mean_v1,
        "gate_rule_outcomes": {
            name: {"permitted_floor": mean_v1 + tolerance, "accepted_v2_runs": sum(score >= mean_v1 + tolerance for score in v2), "total_v2_runs": len(v2)}
            for name, tolerance in rules.items()
        },
    }
    output = root / "artifacts" / "data_drill" / "summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
