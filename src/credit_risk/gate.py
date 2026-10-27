from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from credit_risk.config import load_config


def evaluate(candidate: dict, baseline: dict, config: dict) -> tuple[bool, list[str]]:
    gate = config["quality_gate"]
    metric = gate["metric"]
    c, b = candidate["best_run"], baseline["best_run"]
    reasons = []
    if c[metric] < b[metric] + gate["minimum_baseline_delta"]:
        reasons.append(f"{metric} {c[metric]:.4f} is below permitted baseline {b[metric] + gate['minimum_baseline_delta']:.4f}")
    if c["recall_at_precision"] < gate["minimum_recall_at_precision"]:
        reasons.append("recall at 50% precision is below the safety floor")
    return not reasons, reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    candidate, baseline = json.loads(Path(args.candidate).read_text()), json.loads(Path(args.baseline).read_text())
    passed, reasons = evaluate(candidate, baseline, config)
    print("## Credit Risk Quality Gate")
    print(f"**{'PASS' if passed else 'FAIL'}** | candidate PR-AUC: {candidate['best_run']['pr_auc']:.4f} | baseline PR-AUC: {baseline['best_run']['pr_auc']:.4f}")
    for reason in reasons: print(f"- {reason}")
    sys.exit(0 if passed else 1)
