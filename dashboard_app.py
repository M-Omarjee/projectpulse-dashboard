# dashboard_app.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import load_project_data, kpis as compute_kpis, finance_table
from src.forecast_model import predict_delays, METRICS_PATH
from src.risk_analysis import risk_score
from src.summary_generator import generate_summary

REPO_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="ProjectPulse Dashboard", layout="wide")
st.title("📈 ProjectPulse Dashboard")
st.caption(f"Data refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# --- Data loaders -----------------------------------------------------------
@st.cache_data(ttl=300)
def load_all():
    df = load_project_data()
    kpi_metrics = compute_kpis(df)
    return {
        "kpis": kpi_metrics,
        "forecast": predict_delays(df),
        "risks": risk_score(df),
        "finance": finance_table(df),
        "summary": generate_summary(kpi_metrics),
    }


@st.cache_data(ttl=300)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


try:
    data = load_all()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

kpis_d   = data["kpis"]
forecast = data["forecast"]
risks    = data["risks"]
finance  = data["finance"]
summary  = data["summary"]

project_meta = risks[["Project_ID", "Phase", "Status"]].drop_duplicates()
if not finance.empty:
    finance = finance.merge(project_meta, on="Project_ID", how="left")

# --- Sidebar filters --------------------------------------------------------
st.sidebar.header("Filters")
phase_sel  = st.sidebar.multiselect("Project Phase", sorted(project_meta["Phase"].dropna().unique()))
status_sel = st.sidebar.multiselect("Status",        sorted(project_meta["Status"].dropna().unique()))

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if "Phase" in df.columns and phase_sel:
        df = df[df["Phase"].isin(phase_sel)]
    if "Status" in df.columns and status_sel:
        df = df[df["Status"].isin(status_sel)]
    return df

forecast_f = apply_filters(forecast.copy())
risks_f    = apply_filters(risks.copy())
finance_f  = apply_filters(finance.copy()) if not finance.empty else finance

# --- Tabs -------------------------------------------------------------------
overview, finance_tab, methodology_tab = st.tabs(
    ["📊 Overview", "💰 Finance", "🔬 Methodology"]
)

# ============================================================ OVERVIEW
with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", kpis_d["projects"])
    c2.metric("Avg Completion", f"{kpis_d['avg_completion_pct']:.1f}%")
    c3.metric("Cost Variance", f"{kpis_d['cost_variance_pct']:.1f}%")
    c4.metric("Time Variance", f"{kpis_d['time_variance_pct']:.1f}%")

    st.markdown("### Delay Probability by Project")
    if not forecast_f.empty:
        fig = px.bar(
            forecast_f, x="Project_ID", y="Delay_Probability",
            color="Status", barmode="group", text="Delay_Probability"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title="% Delay Probability", xaxis_title="", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Probabilities from a logistic-regression model trained on 200 "
            "historical NHS projects with a temporal train/test split. "
            "See the **Methodology** tab for full evaluation."
        )
    else:
        st.info("No forecast rows after filters.")

    st.markdown("### Risk Register")
    st.dataframe(risks_f, use_container_width=True)

    st.markdown("### Executive Summary")
    if isinstance(summary, dict):
        st.write(summary.get("executive_summary", summary))
    else:
        st.write(summary)

# ============================================================ FINANCE
with finance_tab:
    st.subheader("💰 Budget vs Actual")
    if not finance_f.empty:
        fig = px.bar(finance_f, x="Project_ID", y=["Budget", "Actual"], barmode="group", title=None)
        fig.update_layout(yaxis_title="£", xaxis_title="")
        fig.update_yaxes(tickprefix="£", separatethousands=True)
        st.plotly_chart(fig, use_container_width=True)

        nice = finance_f.copy()
        nice["Variance_Pct"] = nice["Variance_Pct"].round(1)

        def color_variance(val):
            if val > 0: return "background-color:#ffe5e5"
            if val < 0: return "background-color:#e6ffea"
            return "background-color:#f3f3f3"

        styled = (
            nice.style
            .format({
                "Budget": "£{:,.0f}", "Actual": "£{:,.0f}",
                "Variance_Amount": "£{:,.0f}", "Variance_Pct": "{:.1f}%"
            })
            .apply(lambda col: [color_variance(v) if col.name in ["Variance_Amount", "Variance_Pct"] else "" for v in col], axis=0)
        )
        st.markdown("#### Variance Table")
        st.dataframe(styled, use_container_width=True)

        csv = nice.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Finance CSV", csv, "finance.csv", "text/csv")
    else:
        st.info("Finance data not available or filtered out.")

# ============================================================ METHODOLOGY
with methodology_tab:
    metrics = load_metrics()
    if metrics is None:
        st.warning(
            "No trained model found. Run `python -m scripts.train_model` to "
            "train the delay forecasting pipeline and generate evaluation metrics."
        )
        st.stop()

    st.subheader("How the delay forecasting model is trained and evaluated")
    st.markdown(
        "The model in the **Overview** tab is not fit on the live portfolio. "
        "It is trained on 200 historical NHS projects (synthetic, generator in "
        "`scripts/generate_synthetic_data.py`) using a **temporal train/test "
        "split**: the earliest 70% of projects by start date train the model, "
        "and the latest 30% are held out for evaluation. This simulates "
        "deploying the model to projects that started after training data was "
        "collected."
    )

    split = metrics["split"]
    test = metrics["test_metrics"]
    winner = metrics["winner_name"].replace("_", " ").title()

    st.markdown(f"**Selected model:** {winner} — chosen automatically as the "
                f"better of two candidates (Random Forest vs Logistic Regression) "
                f"on held-out test ROC AUC.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test ROC AUC", f"{test['roc_auc']:.3f}",
              help="Discrimination: probability the model ranks a delayed project above an on-time one.")
    c2.metric("Test PR AUC", f"{test['pr_auc']:.3f}",
              help="Average precision across recall thresholds; useful for imbalanced outcomes.")
    c3.metric("Brier Score", f"{test['brier']:.3f}",
              help="Mean squared error of probabilities (0 perfect, 0.25 random).")
    c4.metric("Test Set Size", f"{test['n']} projects")

    # Model comparison
    st.markdown("### Model comparison")
    comp = pd.DataFrame([
        {
            "Model":       name.replace("_", " ").title(),
            "Train AUC":   round(v["train"]["roc_auc"], 3),
            "Test AUC":    round(v["test"]["roc_auc"], 3),
            "Test Brier":  round(v["test"]["brier"], 3),
            "Test F1":     round(v["test"]["f1"], 3),
            "Selected":    "✓" if name == metrics["winner_name"] else "",
        }
        for name, v in metrics["all_models"].items()
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ROC + PR
    st.markdown("### Performance curves (held-out test set)")
    col_roc, col_pr = st.columns(2)
    with col_roc:
        roc = metrics["curves"]["roc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                     name=f"{winner} (AUC = {test['roc_auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash"), name="Random"))
        fig_roc.update_layout(title="ROC curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              legend=dict(x=0.4, y=0.1))
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_pr:
        pr = metrics["curves"]["pr"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                    name=f"AP = {test['pr_auc']:.3f}"))
        fig_pr.update_layout(title="Precision-Recall curve",
                             xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    # Calibration
    st.markdown("### Calibration")
    st.markdown(
        "Are the predicted probabilities meaningful, or just relative scores? "
        "Each point shows the average actual delay rate among test projects "
        "whose predicted probability fell into that bin. Points near the "
        "diagonal indicate the probabilities are well-calibrated."
    )
    calib_df = pd.DataFrame(metrics["curves"]["calibration"])
    if not calib_df.empty:
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=calib_df["predicted"], y=calib_df["actual"],
                                     mode="lines+markers", name="Observed",
                                     marker=dict(size=calib_df["count"] * 2 + 6)))
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash"), name="Perfect calibration"))
        fig_cal.update_layout(xaxis_title="Predicted probability of delay",
                              yaxis_title="Observed delay rate in bin",
                              xaxis=dict(range=[0, 1]),
                              yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_cal, use_container_width=True)

    # Feature importance
    fi = metrics.get("feature_importance", [])
    if fi:
        st.markdown("### Feature contribution")
        fi_df = pd.DataFrame(fi).head(12)
        fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h")
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"},
                             xaxis_title="Magnitude of contribution to delay prediction",
                             yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(
            "For logistic regression, magnitude of the standardised "
            "coefficient. For random forest, mean decrease in impurity."
        )

    # Methodology notes
    with st.expander("📝 Methodology notes & limitations"):
        st.markdown(f"""
- **Split**: {split['train_size']} training projects ({split['earliest_train_date']} → {split['latest_train_date']}), {split['test_size']} test projects ({split['earliest_test_date']} → {split['latest_test_date']}).
- **Snapshot design**: every training row records the project's state at exactly 60% of its *planned* duration. Because the elapsed fraction is fixed by construction, no feature can leak the eventual delay status — fixing the target-leakage problem in the original implementation.
- **Features**: planned duration, planned cost, spend to date, completion %, project category, current status, risk level.
- **Target**: did the project finally complete late (`Final_Actual_Duration > Planned_Duration`).
- **Synthetic data caveats**: this is *generated* portfolio data designed to be plausible, not real NHS data. The model demonstrates the methodology; absolute numbers should not be used for real forecasting without retraining on real data.
- **What this is**: a methodologically defensible *demonstration* of programme-level delay forecasting with proper evaluation, calibration, and explainability.
- **What this is not**: a production-grade clinical or operational AI model. Real deployment would require larger n, real outcomes data, fairness analysis across project types, and ongoing monitoring for drift.
        """)