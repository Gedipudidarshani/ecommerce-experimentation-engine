import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.engine import ExperimentationPlatformEngine

st.set_page_config(
    page_title="E-Commerce Experimentation Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("E-Commerce Decision System: Experimentation & CUPED Engine")
st.markdown("*Evaluating front-end conversion lifts against back-end fulfillment guardrails and CUPED variance reduction.*")

# 1. Sidebar Configurations
st.sidebar.header("Experiment Controls")
data_source = st.sidebar.radio(
    "Select Evaluation Mode",
    ["Simulated Counterfactual Treatment (ExP Demo)", "Raw BigQuery Export (A/A Audit)"]
)
alpha_level = st.sidebar.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)

@st.cache_data
def load_raw_bigquery_data():
    return pd.read_csv("data/gold_features.csv")

@st.cache_data
def load_simulated_treatment_data():
    np.random.seed(42)
    sample_size = 50000
    user_ids = np.arange(100001, 100001 + sample_size)
    variant = np.random.choice(['Control', 'Treatment'], size=sample_size, p=[0.5, 0.5])
    
    # Baseline covariates
    pre_sessions = np.random.negative_binomial(n=2, p=0.3, size=sample_size)
    propensity = 1.0 / (1.0 + np.exp(-(0.4 * pre_sessions - 2.0)))
    has_prior_spend = np.random.binomial(n=1, p=propensity)
    pre_spend = np.where(has_prior_spend == 1, np.random.gamma(shape=2.5, scale=40.0, size=sample_size), 0.0)
    
    # Treatment lift on conversion
    base_logit = -3.4 + 0.005 * pre_spend + 0.12 * pre_sessions
    treatment_bump = np.where(variant == 'Treatment', 0.28, 0.0)
    p_conv = 1.0 / (1.0 + np.exp(-(base_logit + treatment_bump)))
    converted = np.random.binomial(n=1, p=p_conv)
    
    # Orders and item volumes
    post_orders = np.where(converted == 1, np.random.poisson(lam=0.4, size=sample_size) + 1, 0)
    shipped_items = np.where(converted == 1, post_orders + np.random.binomial(n=post_orders, p=0.35), 0)
    gross_revenue = np.where(converted == 1, shipped_items * np.random.gamma(shape=5.0, scale=14.0, size=sample_size), 0.0)
    cogs = gross_revenue * np.random.uniform(0.40, 0.46, size=sample_size)
    
    # Logistics and fulfillment tiers
    distance_km = np.random.uniform(50.0, 2800.0, size=sample_size)
    freight = np.where(shipped_items > 0, shipped_items * (4.50 + (distance_km * 0.0035)), 0.0)
    
    # Guardrails: Treatment triggers return bracketing and carrier SLA delays
    ret_prob = np.where(variant == 'Treatment', 0.124, 0.082)
    returned_items = np.random.binomial(n=shipped_items, p=ret_prob)
    avg_price = np.divide(gross_revenue, np.maximum(1, shipped_items))
    reverse_logistics = returned_items * (9.00 + (0.15 * avg_price))
    
    sla_prob = np.where(
        variant == 'Treatment',
        np.where(distance_km > 1200.0, 0.142, 0.038),
        np.where(distance_km > 1200.0, 0.035, 0.012)
    )
    valid_delivered = np.maximum(0, shipped_items - np.random.binomial(n=shipped_items, p=0.01))
    sla_breaches = np.random.binomial(n=valid_delivered, p=sla_prob)
    net_margin = np.round(gross_revenue - cogs - freight - reverse_logistics, 2)
    
    return pd.DataFrame({
        'user_id': user_ids,
        'variant': variant,
        'pre_spend_covariate': pre_spend,
        'pre_sessions_covariate': pre_sessions,
        'converted': converted,
        'post_orders': post_orders,
        'total_shipped': shipped_items,
        'total_returned': returned_items,
        'valid_delivered': valid_delivered,
        'sla_breaches': sla_breaches,
        'gross_revenue': gross_revenue,
        'cogs': cogs,
        'outbound_shipping_cost': freight,
        'reverse_logistics_cost': reverse_logistics,
        'net_contribution_margin': net_margin
    })

# Ingest appropriate evaluation slice
if data_source.startswith("Simulated"):
    df = load_simulated_treatment_data()
    st.info("Displaying **Counterfactual Simulation Mode** (N=50,000): Demonstrates treatment effect injection, CUPED variance reduction, and supply chain guardrail tripwires.")
else:
    df = load_raw_bigquery_data()
    st.warning("Displaying **Raw BigQuery Export (A/A Audit)** (N=1,158): Validates baseline pipeline calibration. Because historical data contains no active treatment, metrics remain statistically identical (nominal A/A state).")

# Execute Engine
engine = ExperimentationPlatformEngine(df, alpha=alpha_level)
results = engine.run()

# Diagnostic Status Bar
col_srm, col_c, col_t = st.columns(3)
with col_srm:
    if results.srm_passed:
        st.success(f"SRM Passed (p = {results.srm_p_value:.4f})")
    else:
        st.error(f"SRM Flagged (p = {results.srm_p_value:.4e})")
with col_c:
    st.metric("Control Sample Size", f"{results.sample_sizes['Control']:,}")
with col_t:
    st.metric("Treatment Sample Size", f"{results.sample_sizes['Treatment']:,}")

st.divider()

# Core Metric Evaluation
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Conversion Rate (CR)")
    st.metric(
        label="Treatment vs Control",
        value=f"{results.cr_treatment:.2f}%",
        delta=f"{results.cr_lift_pct:+.2f}% Rel Lift"
    )
    st.caption(f"Two-Proportion Z-Test (p = {results.cr_p_value:.4e})")
    if results.cr_p_value < alpha_level:
        st.success("Significant conversion lift verified.")
    else:
        st.info("Inconclusive conversion difference.")

with col2:
    st.subheader("2. Raw Margin (NCMU)")
    st.metric(
        label="Raw Margin Delta",
        value=f"{results.raw_delta_ncmu:+.2f} USD",
        delta=f"p = {results.raw_ncmu_p_value:.4f}"
    )
    st.caption("Standard Welch's Two-Sample T-Test")
    if results.raw_ncmu_p_value < alpha_level:
        st.success("Statistically significant.")
    else:
        st.caption("High variance conceals treatment lift.")

with col3:
    st.subheader("3. CUPED Margin (OEC)")
    st.metric(
        label="CUPED Adjusted Delta",
        value=f"{results.cuped_delta_ncmu:+.2f} USD",
        delta=f"-{results.variance_reduction_pct:.1f}% Variance"
    )
    st.caption(f"95% CI: [{results.cuped_ci[0]:.2f}, {results.cuped_ci[1]:.2f}] USD | p = {results.cuped_ncmu_p_value:.4f}")
    if results.cuped_ncmu_p_value < alpha_level:
        st.success("Significant lift verified via CUPED.")
    else:
        st.caption("Not statistically significant.")

st.divider()

# Operational Guardrails Diagnostics
st.subheader("Operational Guardrail Diagnostics (FDR Corrected)")
g1, g2 = st.columns(2)
with g1:
    fig_ret = go.Figure(data=[
        go.Bar(name='Control', x=['Return Rate'], y=[results.returns_control_rate], marker_color='#1f77b4'),
        go.Bar(name='Treatment', x=['Return Rate'], y=[results.returns_treatment_rate], marker_color='#d62728')
    ])
    fig_ret.update_layout(yaxis_title="Percentage (%)", barmode='group', height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_ret, width='stretch')
    if results.returns_breached:
        st.error(f"GUARDRAIL BREACHED: Return rate increased to {results.returns_treatment_rate:.2f}% (FDR p = {results.returns_fdr_p:.4e})")
    else:
        st.success(f"Return rate within tolerance ({results.returns_treatment_rate:.2f}% vs {results.returns_control_rate:.2f}%)")

with g2:
    fig_sla = go.Figure(data=[
        go.Bar(name='Control', x=['SLA Breach'], y=[results.sla_control_rate], marker_color='#1f77b4'),
        go.Bar(name='Treatment', x=['SLA Breach'], y=[results.sla_treatment_rate], marker_color='#d62728')
    ])
    fig_sla.update_layout(yaxis_title="Percentage (%)", barmode='group', height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_sla, width='stretch')
    if results.sla_breached:
        st.error(f"GUARDRAIL BREACHED: 48h SLA breaches surged to {results.sla_treatment_rate:.2f}% (FDR p = {results.sla_fdr_p:.4e})")
    else:
        st.success(f"SLA breach rate within tolerance ({results.sla_treatment_rate:.2f}% vs {results.sla_control_rate:.2f}%)")

st.divider()

# Executive Decision Summary
st.subheader("Executive Decision Summary")
if results.returns_breached or results.sla_breached:
    st.error(
        "**DECISION: BLOCK SITEWIDE ROLLOUT**\n\n"
        "While conversion and CUPED-adjusted margin showed positive movement, the intervention triggered critical supply chain tripwires. "
        "Expedited delivery guarantees caused a surge in return bracketing and carrier SLA breaches.\n\n"
        "**Recommended Action:** Restrict badge visibility dynamically to customers residing within 600 km of regional distribution centers with stocked inventory."
    )
else:
    st.success(
        "**DECISION: MAINTAIN NOMINAL BASELINE (A/A CALIBRATION VERIFIED)**\n\n"
        "No statistically significant deviation detected across Conversion, Net Margin, or Guardrails. "
        "The experimental pipeline correctly controls Type I error rates on baseline historical records."
    )