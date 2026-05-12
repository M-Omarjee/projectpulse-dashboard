"""
Delay forecasting model for ProjectPulse.

Trained on data/training_data.csv (200 historical NHS projects with known
delay outcomes). Evaluated using a temporal train/test split: earlier 70%
of projects (by start date) train, later 30% are held out — simulating
deployment to projects that started after the model was built.

The "60% planned-duration snapshot" design (see scripts/generate_synthetic_data.py)
ensures no feature leaks the eventual outcome: Actual_Duration is fixed at 60%
of Planned_Duration for every row, so it carries no signal about the eventual
delay status. This is the key methodological fix vs the original implementation,
which had Actual_Duration in both the features AND the target.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Configuration ----------------------------------------------------------
NUMERIC_FEATURES = [
    "Planned_Duration",
    "Planned_Cost",
    "Actual_Cost",      # spend to date at the 60% snapshot
    "Completion_Pct",   # actual completion at the 60% snapshot
]
CATEGORICAL_FEATURES = [
    "Category",
    "Status",
    "Risk_Level",
]
TARGET = "Was_Delayed"
TIME_COL = "Start_Date"
TEMPORAL_TRAIN_FRACTION = 0.70

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "forecast_pipeline.joblib"
METRICS_PATH = REPO_ROOT / "models" / "forecast_metrics.json"


# --- Pipeline construction --------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model_type: str = "rf") -> Pipeline:
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif model_type == "lr":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([("preprocess", _build_preprocessor()), ("clf", clf)])


# --- Training & evaluation --------------------------------------------------
def temporal_split(df: pd.DataFrame, train_fraction: float = TEMPORAL_TRAIN_FRACTION):
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cutoff = int(len(df) * train_fraction)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _evaluate(pipe: Pipeline, df: pd.DataFrame) -> dict:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int).values
    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "delay_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }


def _curves(pipe: Pipeline, df: pd.DataFrame) -> dict:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int).values
    probs = pipe.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    prec, rec, _ = precision_recall_curve(y, probs)
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() > 0:
            calib.append({
                "bin_mid":   float((bins[b] + bins[b + 1]) / 2),
                "predicted": float(probs[mask].mean()),
                "actual":    float(y[mask].mean()),
                "count":     int(mask.sum()),
            })
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr":  {"precision": prec.tolist(), "recall": rec.tolist()},
        "calibration": calib,
    }


def _feature_importance(pipe: Pipeline) -> list[dict]:
    clf = pipe.named_steps["clf"]
    preprocessor: ColumnTransformer = pipe.named_steps["preprocess"]
    feature_names = (
        NUMERIC_FEATURES
        + list(preprocessor.named_transformers_["cat"]
                            .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_.tolist()
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0]).tolist()
    else:
        return []
    pairs = sorted(zip(feature_names, importances),
                   key=lambda p: p[1], reverse=True)
    return [{"feature": str(f), "importance": float(i)} for f, i in pairs]


def train_and_evaluate(training_df: pd.DataFrame) -> dict:
    """Temporal split → fit RF + LR → evaluate. Return everything for downstream use."""
    train_df, test_df = temporal_split(training_df)
    results = {}
    for name, model_type in [("random_forest", "rf"), ("logistic_regression", "lr")]:
        pipe = build_pipeline(model_type)
        pipe.fit(
            train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            train_df[TARGET].astype(int).values,
        )
        results[name] = {
            "pipeline":      pipe,
            "train_metrics": _evaluate(pipe, train_df),
            "test_metrics":  _evaluate(pipe, test_df),
        }
    winner_name = max(results, key=lambda k: results[k]["test_metrics"]["roc_auc"])
    winner = results[winner_name]
    return {
        "winner_name":  winner_name,
        "pipeline":     winner["pipeline"],
        "train_metrics": winner["train_metrics"],
        "test_metrics":  winner["test_metrics"],
        "all_models": {
            name: {"train": r["train_metrics"], "test": r["test_metrics"]}
            for name, r in results.items()
        },
        "curves":             _curves(winner["pipeline"], test_df),
        "feature_importance": _feature_importance(winner["pipeline"]),
        "split": {
            "train_size":          len(train_df),
            "test_size":           len(test_df),
            "train_fraction":      TEMPORAL_TRAIN_FRACTION,
            "earliest_train_date": train_df[TIME_COL].min(),
            "latest_train_date":   train_df[TIME_COL].max(),
            "earliest_test_date":  test_df[TIME_COL].min(),
            "latest_test_date":    test_df[TIME_COL].max(),
        },
    }


# --- Inference (used by dashboard & FastAPI) --------------------------------
def _load_pipeline() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. "
            "Run `python scripts/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_delays(df: pd.DataFrame) -> pd.DataFrame:
    """Score an in-flight portfolio using the persisted pipeline."""
    pipe = _load_pipeline()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probs = pipe.predict_proba(X)[:, 1]
    out = df[["Project_ID", "Phase", "Status"]].copy()
    out["Delay_Probability"] = (probs * 100).round(1)
    return out.sort_values("Delay_Probability", ascending=False).reset_index(drop=True)