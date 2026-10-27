"""Produce ten tracked model runs: two model families across five splits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from credit_risk.train import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    args = parser.parse_args()
    root = Path(args.config).resolve().parents[1]
    base = yaml.safe_load(Path(args.config).read_text())
    generated = root / "artifacts" / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        config = deepcopy(base)
        config["training"]["random_seed"] = seed
        config_path = generated / f"full_seed_{seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        run(str(config_path), f"artifacts/full_runs/seed_{seed}.json")
    print(f"Completed {len(args.seeds) * len(base['training']['models'])} tracked runs.")


if __name__ == "__main__":
    main()
