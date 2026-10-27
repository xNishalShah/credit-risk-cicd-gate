from credit_risk.gate import evaluate


CONFIG = {"quality_gate": {"metric": "pr_auc", "minimum_baseline_delta": -0.01, "minimum_recall_at_precision": 0.45}}


def test_gate_rejects_material_regression():
    candidate = {"best_run": {"pr_auc": 0.60, "recall_at_precision": 0.80}}
    baseline = {"best_run": {"pr_auc": 0.70, "recall_at_precision": 0.80}}
    passed, reasons = evaluate(candidate, baseline, CONFIG)
    assert not passed
    assert "below permitted baseline" in reasons[0]


def test_intentional_demo_failure():
    assert False, "Intentional CI failure for demonstration"
