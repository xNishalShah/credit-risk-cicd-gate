"""Create the required Income_Bracket error-rate breakdown for slide 8."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import load_config
from credit_risk.features import engineer_features
from credit_risk.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.config).resolve().parents[1]
    frame = engineer_features(pd.read_csv(root / config["data"]["path"]), config["features"])
    target = config["data"]["target"]
    X, y = frame.drop(columns=[target, "Customer_ID"]), frame[target]
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["training"]["test_size"], stratify=y, random_state=config["training"]["random_seed"])
    fitted = []
    for candidate in config["training"]["models"]:
        pipe = Pipeline([("preprocess", preprocess), ("model", build_model(candidate["name"], candidate["params"]))])
        pipe.fit(X_train, y_train)
        probabilities = pipe.predict_proba(X_test)[:, 1]
        fitted.append((average_precision_score(y_test, probabilities), candidate["name"], probabilities))
    _, model_name, probabilities = max(fitted, key=lambda item: item[0])
    predictions = (probabilities >= 0.5).astype(int)
    rows = []
    for group in sorted(X_test["Income_Bracket"].dropna().unique()):
        mask = X_test["Income_Bracket"] == group
        truth, predicted = y_test[mask], predictions[mask]
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        rows.append({"income_bracket": group, "support": int(mask.sum()), "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else None, "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None})
    report = {"data_version": config["data"]["version"], "model": model_name, "threshold": 0.5, "breakdown": rows}
    out = root / "artifacts" / "fairness_income_bracket.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
