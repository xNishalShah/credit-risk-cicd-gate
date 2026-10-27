"""Register the best Logistic Regression run from the full experiment."""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()
experiment = client.get_experiment_by_name("credit-risk-full")
if experiment is None:
    raise SystemExit("Run scripts/run_seed_sweep.py first.")
runs = mlflow.search_runs([experiment.experiment_id], order_by=["metrics.pr_auc DESC"])
logistic = runs[runs["tags.mlflow.runName"] == "logistic_regression"].iloc[0]
result = mlflow.register_model(f"runs:/{logistic.run_id}/model", "CreditRiskDefaultModel")
print(f"registered_model=CreditRiskDefaultModel")
print(f"version={result.version}")
print(f"run_id={logistic.run_id}")
print(f"pr_auc={logistic['metrics.pr_auc']:.6f}")
