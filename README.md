# 📈 ProjectPulse

**Programme-level delay forecasting for NHS-style project portfolios.**

A FastAPI + Streamlit application that forecasts the probability of project delay using a model trained with a temporal train/test split on 200 synthetic NHS projects. **Test ROC AUC 0.88, Brier 0.10** on the held-out fold.

[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://projectpulse-dashboard-hit6abrlfmkhp3pzbjzeqv.streamlit.app)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

🔗 **Live demo:** https://projectpulse-dashboard-hit6abrlfmkhp3pzbjzeqv.streamlit.app

![Overview](docs/Overview.png)

---

## Why this exists

Most portfolio dashboards stop at counting projects and colour-coding RAG status. ProjectPulse adds a probabilistic forecast — *given what a programme manager can see today, how likely is each project to finish late?* — and surfaces the full methodology (model comparison, ROC, calibration, feature contribution) in the dashboard itself, so the prediction is auditable rather than opaque.

## What's in the app

| Tab | Contents |
|---|---|
| **📊 Overview** | Portfolio KPIs, per-project delay probability, risk register, executive summary |
| **💰 Finance** | Budget vs Actual by project, variance table with conditional colouring, CSV export |
| **🔬 Methodology** | Model comparison, ROC + PR curves, calibration plot, feature contribution, evaluation notes |

## Methodology

The delay forecast is the differentiating piece, so the design choices are worth being explicit about:

- **Temporal train/test split.** 200 historical synthetic projects are sorted by start date; earliest 70% train the model, latest 30% are held out. This simulates deployment to projects that started *after* the model was built — a stricter and more realistic evaluation than random k-fold.
- **Snapshot at 60% of planned duration.** Every training row records the project's state at exactly 60% of its *planned* duration. Because the elapsed fraction is fixed by construction, no feature can leak the eventual delay status. This was a deliberate fix to a target-leakage flaw in an earlier iteration.
- **Model selection on held-out AUC.** The training pipeline fits both Random Forest and Logistic Regression and selects the better model on the test fold automatically. On current data, Logistic Regression wins (0.884 vs 0.846).
- **Calibration is surfaced.** A decile calibration plot lets users verify the predicted probabilities behave like real probabilities — not just rank scores.
- **Explicit about synthetic data.** The data generator is open (`scripts/generate_synthetic_data.py`) and documented. The model demonstrates methodology, not real NHS forecasting.

**Current performance (held-out 60-project test fold):**

| Metric | Value |
|---|---|
| ROC AUC | 0.884 |
| PR AUC | 0.762 |
| Brier score | 0.102 |
| Selected model | Logistic Regression |

## Architecture

```
              ┌────────────────────────────┐
              │     src/  (data layer)     │
              │  utils · forecast_model    │
              │  risk_analysis · summary   │
              └─────────────┬──────────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
       ┌─────▼──────┐               ┌──────▼─────────┐
       │  FastAPI   │               │   Streamlit    │
       │  (REST)    │               │  (dashboard)   │
       │ src/main.py│               │dashboard_app.py│
       └────────────┘               └────────────────┘
```

Both surfaces share one source of truth. The Streamlit app calls the data layer directly (in-process) for performance; FastAPI exposes the same logic over REST for integration use cases.

## Quick start

```bash
git clone https://github.com/M-Omarjee/projectpulse-dashboard.git
cd projectpulse-dashboard

python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# Launch the dashboard (uses pre-trained model committed to the repo)
streamlit run dashboard_app.py
```

The dashboard opens at http://localhost:8501.

**To regenerate the synthetic data and retrain the model:**

```bash
python -m scripts.generate_synthetic_data
python -m scripts.train_model
```

**To run the FastAPI surface independently:**

```bash
uvicorn src.main:app --reload
```

REST endpoints at http://127.0.0.1:8000 — try `/kpis`, `/forecast`, `/risks`, `/summary`, `/finance`.

## Repository structure

```
projectpulse-dashboard/
├── dashboard_app.py              # Streamlit entrypoint
├── data/
│   ├── training_data.csv         # 200 historical projects (synthetic)
│   └── project_data.csv          # 5 in-flight projects (live demo)
├── models/
│   ├── forecast_pipeline.joblib  # Persisted trained pipeline
│   └── forecast_metrics.json     # Evaluation artefacts for Methodology tab
├── scripts/
│   ├── generate_synthetic_data.py
│   └── train_model.py
├── src/
│   ├── main.py                   # FastAPI app
│   ├── utils.py                  # Data loading + KPI computation
│   ├── forecast_model.py         # Delay forecasting pipeline
│   ├── risk_analysis.py          # Composite risk score
│   └── summary_generator.py      # Executive summary text generation
├── docs/                         # Screenshots
└── requirements.txt
```

## Limitations & next steps

- **Synthetic data, not NHS data.** The training set is procedurally generated to be plausible. Production use would require retraining on real outcomes data, fairness analysis across project categories, and ongoing drift monitoring.
- **No time-series view yet.** Projects are scored at a single snapshot. A view of how each project's probability changes over its lifecycle (the "Pulse" in the name) is the next product addition.
- **No project drill-down.** Clicking a project doesn't yet open a detail view.
- **Rule-based executive summary.** The summary text is template-driven, not LLM-generated. Swapping in an LLM narrative grounded in the underlying KPIs is planned.

## License

MIT.

## Author

Built by [Muhammed Omarjee](https://github.com/M-Omarjee) — UK foundation doctor pivoting to health tech and clinical AI.