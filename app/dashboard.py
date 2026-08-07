"""
dashboard.py
============
PredictGuard — Interactive Streamlit Dashboard.

Page 1: Machine View  (Single machine diagnostics, SHAP features, dispatch advice)
Page 2: Fleet View    (Fleet risk breakdown, top risky machines, component failure summary, cost stats)

Usage:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
import site
from pathlib import Path

# Bootstrap user site-packages
for _sp in ([site.getusersitepackages()]
            if isinstance(site.getusersitepackages(), str)
            else site.getusersitepackages()):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PredictGuardPipeline

# ---------------------------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLORS = {
    "LOW": "#2ECC71",
    "MEDIUM": "#F1C40F",
    "HIGH": "#E67E22",
    "CRITICAL": "#E74C3C",
}

# ---------------------------------------------------------------------------
# Load pipeline & data
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    import pathlib
    try:
        pathlib.WindowsPath()
    except NotImplementedError:
        pathlib.WindowsPath = pathlib.PosixPath
    models_dir = PROJECT_ROOT / "models"
    return PredictGuardPipeline.load(models_dir)


@st.cache_data
def load_test_data():
    test_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if test_path.exists():
        return pd.read_parquet(test_path)
    # Fallback to dev dataset sample if test parquet is missing
    dev_path = PROJECT_ROOT / "data" / "processed" / "development.parquet"
    if dev_path.exists():
        return pd.read_parquet(dev_path).sample(10000, random_state=42)
    return pd.DataFrame()


@st.cache_data
def load_fleet_summary():
    summary_path = PROJECT_ROOT / "reports" / "fleet_risk_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


@st.cache_data
def load_cost_data():
    cost_path = PROJECT_ROOT / "reports" / "cost_analysis.csv"
    if cost_path.exists():
        return pd.read_csv(cost_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 10px; padding: 15px; border-left: 5px solid #3B82F6; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .card-low { border-left-color: #2ECC71 !important; }
    .card-medium { border-left-color: #F1C40F !important; }
    .card-high { border-left-color: #E67E22 !important; }
    .card-critical { border-left-color: #E74C3C !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=60)
st.sidebar.title("PredictGuard AI")
page = st.sidebar.radio("Navigation", ["🔍 Machine View", "📊 Fleet View", "⚙️ System Info"])

pipeline = load_pipeline()
df_test = load_test_data()
df_fleet = load_fleet_summary()

# ===========================================================================
# PAGE 1: MACHINE VIEW
# ===========================================================================
if page == "🔍 Machine View":
    st.markdown('<div class="main-header">Machine Health Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Individual telemetry analysis, SHAP driver attribution, and dispatch advice.</div>', unsafe_allow_html=True)

    if df_test.empty:
        st.error("No test data found.")
        st.stop()

    machines = sorted(df_test["machineID"].unique())
    selected_machine = st.sidebar.selectbox("Select Machine ID", machines)

    machine_df = df_test[df_test["machineID"] == selected_machine].sort_values("datetime", ascending=False)
    latest_row = machine_df.iloc[0]

    # Run inference for this machine hour
    reports = pipeline.predict_report(
        machine_df.iloc[[0]],
        machine_ids=[selected_machine],
        timestamps=[latest_row.get("datetime", "")],
        include_shap_top_n=7,
    )
    rep = reports[0]

    # Risk Banner Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Machine ID", str(selected_machine))
    with c2:
        st.metric("Calibrated Prob", f"{rep['failure_probability']:.1%}")
    with c3:
        st.metric("Risk Score", f"{rep['risk_score']} / 100")
    with c4:
        tier = rep["risk_tier"]
        st.metric("Risk Tier", tier)
    with c5:
        st.metric("Decision", rep["dispatch_decision"])

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("⚡ Risk Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rep["failure_probability"] * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TIER_COLORS.get(rep["risk_tier"], "#3B82F6")},
                "steps": [
                    {"range": [0, 20], "color": "#E8F8F5"},
                    {"range": [20, 50], "color": "#FEF9E7"},
                    {"range": [50, 75], "color": "#FDF2E9"},
                    {"range": [75, 100], "color": "#FDEDEC"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": pipeline.optimal_threshold * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("🚨 Recommended Dispatch Action")
        if rep["dispatch_decision"] == "DISPATCH":
            st.error(f"**ACTION REQUIRED (DISPATCH)**: {rep['recommended_action']}")
        else:
            st.success(f"**NORMAL OPERATIONS (MONITOR)**: {rep['recommended_action']}")

        st.markdown(f"**Component Prediction:** `{rep['predicted_component']}` (Confidence: {rep['component_confidence']:.1%})")

    with col_right:
        st.subheader("🔬 Top SHAP Feature Contributions")
        shap_feats = rep.get("top_shap_features", [])
        if shap_feats:
            sdf = pd.DataFrame(shap_feats)
            sdf["impact"] = np.where(sdf["shap_value"] > 0, "Increases Failure Risk", "Decreases Failure Risk")
            fig_shap = px.bar(
                sdf,
                x="shap_value",
                y="feature",
                orientation="h",
                color="impact",
                color_discrete_map={"Increases Failure Risk": "#E74C3C", "Decreases Failure Risk": "#2ECC71"},
                title="SHAP Attribution (Impact on Log-Odds)",
            )
            fig_shap.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP details unavailable.")

    # Sensor Telemetry Snapshot
    st.subheader("📈 Core Telemetry Features")
    sensor_cols = [c for c in machine_df.columns if any(s in c for s in ["volt", "rotate", "pressure", "vibration"])]
    if sensor_cols:
        sub_df = machine_df[["datetime"] + sensor_cols[:6]].head(24)
        st.dataframe(sub_df, use_container_width=True)

# ===========================================================================
# PAGE 2: FLEET VIEW
# ===========================================================================
elif page == "📊 Fleet View":
    st.markdown('<div class="main-header">Fleet Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fleet-wide risk distribution, top vulnerable assets, and cost-dispatch metrics.</div>', unsafe_allow_html=True)

    if df_fleet.empty:
        st.warning("Fleet summary not generated yet.")
        st.stop()

    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Assets Monitored", len(df_fleet))
    with c2:
        critical_count = (df_fleet["risk_tier"] == "CRITICAL").sum()
        st.metric("Critical Assets", f"{critical_count}", delta=f"{critical_count/len(df_fleet):.1%}", delta_color="inverse")
    with c3:
        avg_risk = df_fleet["avg_risk_score"].mean()
        st.metric("Fleet Avg Risk Score", f"{avg_risk:.1f} / 100")
    with c4:
        opt_thresh = pipeline.optimal_threshold
        st.metric("Optimal Dispatch Threshold", f"{opt_thresh:.2f}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🛡️ Asset Risk Tier Breakdown")
        tier_counts = df_fleet["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_pie = px.pie(
            tier_counts,
            names="Tier",
            values="Count",
            color="Tier",
            color_discrete_map=TIER_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("🛠️ Predicted Failing Component Distribution")
        comp_counts = df_fleet["predicted_component"].value_counts().reset_index()
        comp_counts.columns = ["Component", "Count"]
        fig_comp = px.bar(
            comp_counts,
            x="Component",
            y="Count",
            color="Component",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_comp.update_layout(height=320)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Top High-Risk Assets Requiring Attention")
    top_risky = df_fleet.sort_values("max_risk_score", ascending=False).head(10)
    st.dataframe(
        top_risky[["machineID", "max_risk_score", "latest_risk_score", "risk_tier", "predicted_component", "failure_rate"]],
        use_container_width=True,
    )

    # Cost Analysis Curve
    df_cost = load_cost_data()
    if not df_cost.empty:
        st.markdown("---")
        st.subheader("💰 Cost Optimization Curve (FN Cost = $10,000 | FP Cost = $500)")
        fig_cost = px.line(
            df_cost,
            x="threshold",
            y="expected_cost",
            title="Expected Maintenance Cost vs Probability Threshold",
            labels={"threshold": "Decision Threshold", "expected_cost": "Expected Financial Cost ($)"},
        )
        fig_cost.add_vline(x=pipeline.optimal_threshold, line_dash="dash", line_color="red", annotation_text=f"Optimal ({pipeline.optimal_threshold:.2f})")
        fig_cost.update_layout(height=350)
        st.plotly_chart(fig_cost, use_container_width=True)

# ===========================================================================
# PAGE 3: SYSTEM INFO
# ===========================================================================
else:
    st.markdown('<div class="main-header">PredictGuard System Metadata</div>', unsafe_allow_html=True)
    st.json(pipeline.metadata)
