# dashboard_app.py
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import plotly.express as px

from src.utils import load_project_data, kpis as compute_kpis, finance_table
from src.forecast_model import predict_delays
from src.risk_analysis import risk_score
from src.summary_generator import generate_summary

st.set_page_config(page_title="ProjectPulse Dashboard", layout="wide")
st.title("📈 ProjectPulse Dashboard")
st.caption(f"Data refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# --- Data layer ---
# Streamlit calls the underlying Python functions directly rather than HTTP-
# fetching from FastAPI. The FastAPI app (src/main.py) still exposes the same
# logic over REST for integration use cases, but is not required at runtime.

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

# Attach Phase/Status to Finance using risks table meta
project_meta = risks[["Project_ID", "Phase", "Status"]].drop_duplicates()
if not finance.empty:
    finance = finance.merge(project_meta, on="Project_ID", how="left")

# --- Sidebar filters (global) ---
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

# --- Tabs ---
overview, finance_tab = st.tabs(["📊 Overview", "💰 Finance"])

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
    else:
        st.info("No forecast rows after filters.")

    st.markdown("### Risk Register")
    st.dataframe(risks_f, use_container_width=True)

    st.markdown("### Executive Summary")
    if isinstance(summary, dict):
        st.write(summary.get("executive_summary", summary))
    else:
        st.write(summary)

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
            if val > 0:
                return "background-color:#ffe5e5"
            if val < 0:
                return "background-color:#e6ffea"
            return "background-color:#f3f3f3"

        styled = (
            nice.style
            .format({
                "Budget": "£{:,.0f}",
                "Actual": "£{:,.0f}",
                "Variance_Amount": "£{:,.0f}",
                "Variance_Pct": "{:.1f}%"
            })
            .apply(lambda col: [color_variance(v) if col.name in ["Variance_Amount", "Variance_Pct"] else "" for v in col], axis=0)
        )

        st.markdown("#### Variance Table")
        st.dataframe(styled, use_container_width=True)

        csv = nice.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Finance CSV", csv, "finance.csv", "text/csv")
    else:
        st.info("Finance data not available or filtered out.")