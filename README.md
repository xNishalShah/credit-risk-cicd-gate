# Credit Risk CI CD Quality Gate

This repository prevents a credit-risk model change from merging when it fails tests or regresses against a protected approved baseline. It is designed for the supplied credit-card customer datasets.

## Repository map

| Location | Purpose |
|---|---|
| `configs/` | Validated dev and CI settings, including model choices and gate tolerance. |
| `src/credit_risk/` | Feature engineering, training, configuration validation, and gate logic. |
| `tests/` | Unit tests for features, configuration, and regression rejection. |
| `data/raw/*.dvc` | DVC pointers; raw customer data is deliberately excluded from Git. |
| `baselines/` | Protected approved performance record used by CI. |
| `.github/workflows/` | Pull-request CI workflow. |
| `docs/` | Feature definition and the gate/fairness rationale. |

## Quick start

```bash
# MLflow 2.17 uses PyArrow versions that do not support Python 3.14.
# Use the supplied Python 3.11 Conda environment instead.
conda env create -f environment.yml
conda activate credit-risk-cicd-gate
pip install dvc
# Configure the team's DVC remote once, then retrieve the exact data version
dvc pull data/raw/credit_card_customers_v1.csv.dvc
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m credit_risk.train --config configs/dev.yaml
PYTHONPATH=src python -m credit_risk.gate --config configs/ci.yaml --candidate artifacts/candidate_metrics.json --baseline baselines/approved.json
```

## Data and reproducibility

The CSVs are not committed to ordinary Git history. Their DVC pointer files bind the file names, sizes, and content hashes. Commit a code or configuration change separately, commit a new `.dvc` pointer when the data changes, and tag approved releases. Checking out a release tag followed by `dvc pull` restores the paired code, configuration, and data.

To introduce v2, add the supplied `credit_card_customers_v2.csv` through DVC, change only `data.path` and `data.version` in a reviewed YAML file, and run the three-seed comparison before changing the approved baseline.

## Model and gate decisions

The pipeline median-imputes numeric values, most-frequent-imputes categoricals, one-hot encodes categories, and learns each transform on training data only. It evaluates logistic regression and random forest. PR-AUC is the selection metric because the target is imbalanced; ROC-AUC and recall at 50 percent precision are reported as supporting metrics.

The CI gate allows a PR-AUC decrease of at most 0.01 to avoid blocking ordinary sampling variability and requires recall at 50 percent precision of at least 0.45. The `main` branch must protect `baselines/approved.json`: only a reviewed promotion PR with linked experiment evidence may change it.

## Lifecycle mapping

| Lifecycle stage | Project evidence |
|---|---|
| Plan | `docs/gate_and_fairness_memo.md`, task rationale, and reviewed issue/PR. |
| Code | Feature and pipeline code in `src/`, tested in `tests/`. |
| Build and test | GitHub Actions installs dependencies and runs pytest. |
| Release | Reviewed baseline-promotion PR and Git tag. |
| Deploy | Model registration and serving integration are the next operational step. |
| Operate and monitor | Monthly retraining, data version comparison, fairness report, and gate history. |

ML differs from ordinary software because data and feature definitions are versioned inputs; metric variation means the gate needs a justified tolerance; and fairness/drift monitoring can invalidate a model even when the code is unchanged. A live deployment endpoint and production monitoring service are intentionally outside this classroom repository, but their hand-off criteria are documented above.

## Required evidence to produce before submission

1. Run at least 10 tracked configurations/seeds with MLflow (or export runs from the local `mlruns` directory).
2. Open three PRs: a passing safe change, a deliberately degraded candidate, and a deliberately failing test.
3. Run v1 and v2 at seeds 42, 43, and 44; record the distribution of PR-AUC differences.
4. Add the final model's `Income_Bracket` false-positive/false-negative breakdown to the memo, then have both partners review the recommendation.

## Producing the final presentation evidence

With the Conda environment active, first make both supplied data versions available locally:

```bash
cp ../datasets/project1_credit_risk/credit_card_customers_v1.csv data/raw/
cp ../datasets/project1_credit_risk/credit_card_customers_v2.csv data/raw/
PYTHONPATH=src python scripts/run_seed_sweep.py
PYTHONPATH=src python scripts/compare_data_versions.py
PYTHONPATH=src python scripts/fairness_report.py
```

The seed sweep creates 10 MLflow runs (five seeds multiplied by two model families). Copy the printed v1/v2 means and sample standard deviations, the `gate_rule_outcomes` table, and the `breakdown` table into the final slide. The JSON files in `artifacts/` make the shown numbers traceable.
