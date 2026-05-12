"""
generate_synthetic_data.py
--------------------------
Generates two CSV files for the ProjectPulse dashboard:

  data/training_data.csv  -- 200 historical NHS projects with KNOWN outcomes.
                             Each row is a snapshot taken at 60% of the
                             project's PLANNED duration; the label
                             (Was_Delayed) records whether the project
                             ultimately finished late.

  data/project_data.csv   -- 5 in-flight projects with the same schema, used
                             as the live demo portfolio in the dashboard.

The forecast model is trained on training_data.csv (with a temporal
train/test split) and then applied to project_data.csv at inference time.
This separation is what makes the delay probabilities in the dashboard
genuine out-of-sample predictions rather than in-sample classifications.

Run from the repo root:
    python scripts/generate_synthetic_data.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Reproducibility --------------------------------------------------------
RNG = np.random.default_rng(seed=42)

# --- Domain constants -------------------------------------------------------
CATEGORIES = ["Digital", "Estates", "Transformation"]
CATEGORY_WEIGHTS = [0.40, 0.30, 0.30]
RISK_LEVELS = ["Low", "Medium", "High"]

# Hand-tuned to look vaguely like the NHS project portfolio reality:
# Estates projects run longer, cost more, and are more delay-prone than
# Digital or Transformation work.
CATEGORY_PROFILES = {
    "Digital": {
        "duration_days":   (180, 540),
        "cost_gbp":        (200_000, 3_000_000),
        "risk_weights":    [0.35, 0.45, 0.20],
        "base_delay_prob": 0.25,
    },
    "Estates": {
        "duration_days":   (365, 1095),
        "cost_gbp":        (1_000_000, 30_000_000),
        "risk_weights":    [0.20, 0.45, 0.35],
        "base_delay_prob": 0.40,
    },
    "Transformation": {
        "duration_days":   (180, 720),
        "cost_gbp":        (100_000, 2_000_000),
        "risk_weights":    [0.30, 0.50, 0.20],
        "base_delay_prob": 0.30,
    },
}

# The snapshot is taken at this fraction of the PLANNED duration.
# Using planned (not actual) duration is critical: it means
# Pct_Of_Planned_Elapsed is fixed at 0.6 for every row and cannot leak
# the eventual outcome.
SNAPSHOT_FRACTION = 0.60


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def phase_from_completion(pct: float) -> str:
    """Map completion % to a PRINCE2-flavoured phase label."""
    if pct < 15:   return "Initiation"
    if pct < 35:   return "Design"
    if pct < 65:   return "Build"
    if pct < 85:   return "Testing"
    if pct < 100:  return "Deployment"
    return "Closed"


def generate_project(idx: int, in_flight: bool, today: date) -> dict:
    """
    Generate one synthetic project.

    Parameters
    ----------
    idx        : project number used to build Project_ID
    in_flight  : if True, project is still running today (outcome unknown).
                 If False, project has completed in the past (outcome known).
    today      : reference date for in-flight projects
    """
    category = RNG.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
    profile  = CATEGORY_PROFILES[category]

    planned_duration = int(RNG.integers(*profile["duration_days"]))
    planned_cost     = float(RNG.uniform(*profile["cost_gbp"]))
    risk_level       = RNG.choice(RISK_LEVELS, p=profile["risk_weights"])

    # --- Latent eventual outcome (what the model has to recover) ----------
    base = profile["base_delay_prob"]
    risk_effect = {"Low": 0.0, "Medium": 0.6, "High": 1.2}[risk_level]
    delay_logit = (
        np.log(base / (1 - base))
        + risk_effect
        + 0.001 * (planned_duration - 365)        # longer projects drift more
        + RNG.normal(0, 0.5)                      # irreducible noise
    )
    will_be_delayed = bool(RNG.random() < sigmoid(delay_logit))

    # --- Observation snapshot at SNAPSHOT_FRACTION ------------------------
    snapshot_day = int(planned_duration * SNAPSHOT_FRACTION)
    expected_completion_pct = SNAPSHOT_FRACTION * 100

    if will_be_delayed:
        # A doomed project shows realistic early-warning signs at 60% mark.
        completion_pct = max(0, expected_completion_pct + RNG.normal(-10, 6))
        cost_to_date   = planned_cost * SNAPSHOT_FRACTION * RNG.uniform(1.02, 1.18)
        final_duration = planned_duration * RNG.uniform(1.08, 1.55)
        final_cost     = planned_cost * RNG.uniform(1.05, 1.35)
    else:
        completion_pct = min(100, expected_completion_pct + RNG.normal(2, 4))
        cost_to_date   = planned_cost * SNAPSHOT_FRACTION * RNG.uniform(0.93, 1.04)
        final_duration = planned_duration * RNG.uniform(0.93, 1.04)
        final_cost     = planned_cost * RNG.uniform(0.95, 1.05)

    # --- Human-assigned status label at snapshot --------------------------
    schedule_variance = (completion_pct - expected_completion_pct) / expected_completion_pct
    if schedule_variance > -0.05:
        status = "On Track"
    elif schedule_variance > -0.20:
        status = "At Risk"
    else:
        status = "Delayed"

    phase = phase_from_completion(completion_pct)

    # --- Start date: roll backwards from snapshot -------------------------
    if in_flight:
        start_date = today - timedelta(days=snapshot_day)
    else:
        completion_offset = int(RNG.integers(180, 1095))
        completion_date = today - timedelta(days=completion_offset)
        start_date = completion_date - timedelta(days=int(final_duration))

    return {
        "Project_ID":            f"NHS-{idx:03d}",
        "Category":              str(category),
        "Start_Date":            start_date.isoformat(),
        "Phase":                 phase,
        "Status":                status,
        "Risk_Level":            str(risk_level),
        "Planned_Duration":      int(planned_duration),
        "Planned_Cost":          round(float(planned_cost), 2),
        "Actual_Duration":       int(snapshot_day),
        "Actual_Cost":           round(float(cost_to_date), 2),
        "Completion_Pct":        round(float(completion_pct), 1),
        # Outcome fields: NaN for in-flight, known for historical
        "Final_Actual_Duration": (None if in_flight else int(final_duration)),
        "Final_Actual_Cost":     (None if in_flight else round(float(final_cost), 2)),
        "Was_Delayed":           (None if in_flight else int(will_be_delayed)),
    }


def main():
    today = date(2026, 5, 12)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    # 200 historical projects (training/evaluation)
    training_rows = [generate_project(i + 1, in_flight=False, today=today)
                     for i in range(200)]
    training_df = pd.DataFrame(training_rows)
    training_df = training_df.sort_values("Start_Date").reset_index(drop=True)
    training_df["Project_ID"] = [f"NHS-{i+1:03d}" for i in range(len(training_df))]
    training_path = data_dir / "training_data.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Wrote {len(training_df):>4} rows -> {training_path.relative_to(repo_root)}")

    # 5 in-flight projects (live demo portfolio)
    demo_rows = [generate_project(i + 1, in_flight=True, today=today)
                 for i in range(5)]
    demo_df = pd.DataFrame(demo_rows)
    demo_path = data_dir / "project_data.csv"
    demo_df.to_csv(demo_path, index=False)
    print(f"Wrote {len(demo_df):>4} rows -> {demo_path.relative_to(repo_root)}")

    # Sanity summary
    print()
    print(f"Training delay rate: {training_df['Was_Delayed'].mean():.1%}")
    print("Training category breakdown:")
    print(training_df["Category"].value_counts().to_string())
    print()
    print("Delay rate by category (training):")
    print(training_df.groupby("Category")["Was_Delayed"].mean().round(3).to_string())


if __name__ == "__main__":
    main()