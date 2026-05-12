"""
Train and persist the delay forecasting pipeline.

Run from the repo root:
    python scripts/train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.forecast_model import (
    METRICS_PATH, MODEL_PATH, train_and_evaluate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "data" / "training_data.csv"


def main():
    if not TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"{TRAINING_DATA} not found. "
            "Run scripts/generate_synthetic_data.py first."
        )

    training_df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded {len(training_df)} projects from "
          f"{TRAINING_DATA.relative_to(REPO_ROOT)}")
    print()

    result = train_and_evaluate(training_df)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(result["pipeline"], MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH.relative_to(REPO_ROOT)}")

    metrics_payload = {
        "winner_name":        result["winner_name"],
        "train_metrics":      result["train_metrics"],
        "test_metrics":       result["test_metrics"],
        "all_models":         result["all_models"],
        "curves":             result["curves"],
        "feature_importance": result["feature_importance"],
        "split":              result["split"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2, default=str)
    print(f"Saved metrics  -> {METRICS_PATH.relative_to(REPO_ROOT)}")
    print()

    print(f"Selected model: {result['winner_name']}")
    print()
    print(f"Train fold ({result['split']['train_size']} projects, "
          f"{result['split']['earliest_train_date']} → "
          f"{result['split']['latest_train_date']}):")
    for k, v in result["train_metrics"].items():
        print(f"  {k:>12}: {v}")
    print()
    print(f"Test fold  ({result['split']['test_size']} projects, "
          f"{result['split']['earliest_test_date']} → "
          f"{result['split']['latest_test_date']}):")
    for k, v in result["test_metrics"].items():
        print(f"  {k:>12}: {v}")
    print()
    print("Model comparison (test ROC AUC):")
    for name, m in result["all_models"].items():
        print(f"  {name:>22}: {m['test']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()