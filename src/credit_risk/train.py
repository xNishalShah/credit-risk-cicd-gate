from __future__ import annotations

import argparse
import json
from pathlib import Path


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import load_config
from credit_risk.features import engineer_features


def repository_root(config_path: str | Path) -> Path:
    """Find the project root for normal configs and generated experiment configs."""
    resolved = Path(config_path).resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "src" / "credit_risk").is_dir() and (parent / "configs").is_dir():
            return parent
    raise ValueError(f"Could not locate project root from config path: {config_path}")


def build_model(name: str, params: dict):
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)
    raise ValueError(f"Unsupported model family: {name}")


def recall_at_precision(y_true, probabilities, min_precision=0.50):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    valid = recall[precision >= min_precision]
    return float(valid.max()) if len(valid) else 0.0


def run(config_path: str, output_path: str = "artifacts/candidate_metrics.json") -> dict:
    config = load_config(config_path)
    root = repository_root(config_path)
    csv_path = root / config["data"]["path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset is absent: {csv_path}. Run `dvc pull` first.")
    frame = engineer_features(pd.read_csv(csv_path), config["features"])
    target = config["data"]["target"]
    X = frame.drop(columns=[target, "Customer_ID"])
    y = frame[target]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"], stratify=y,
        random_state=config["training"]["random_seed"],
    )
    runs = []
    try:
        import mlflow
        mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
        mlflow.set_experiment(config["tracking"]["experiment_name"])
    except ImportError:
        mlflow = None
    for candidate in config["training"]["models"]:
        pipe = Pipeline([("preprocess", preprocess), ("model", build_model(candidate["name"], candidate["params"]))])
        pipe.fit(X_train, y_train)
        probability = pipe.predict_proba(X_test)[:, 1]
        metrics = {"model": candidate["name"], "params": candidate["params"],
                     "pr_auc": round(float(average_precision_score(y_test, probability)), 6),
                     "roc_auc": round(float(roc_auc_score(y_test, probability)), 6),
                     "recall_at_precision": round(recall_at_precision(y_test, probability), 6)}
        runs.append(metrics)
        if mlflow:
            with mlflow.start_run(run_name=candidate["name"]):
                mlflow.log_params({f"model.{k}": v for k, v in candidate["params"].items()})
                mlflow.log_param("model.family", candidate["name"])
                mlflow.log_metrics({k: v for k, v in metrics.items() if k in {"pr_auc", "roc_auc", "recall_at_precision"}})
                mlflow.set_tags({"data_version": config["data"]["version"], "config_path": str(config_path), "environment": config["environment"]})
                mlflow.log_dict(config, "config.yaml")
                mlflow.sklearn.log_model(pipe, artifact_path="model")
    best = max(runs, key=lambda x: x["pr_auc"])
    result = {"data_version": config["data"]["version"], "config": str(config_path), "best_run": best, "all_runs": runs}
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/candidate_metrics.json")
    args = parser.parse_args()
    run(args.config, args.output)
