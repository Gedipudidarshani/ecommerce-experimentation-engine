## ecommerce-experimentation-engine
## Enterprise E-Commerce Experimentation Engine & Econometrics Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google BigQuery](https://img.shields.io/badge/warehouse-Google%20BigQuery-4285F4.svg)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end analytics engineering, causal inference, and online experimentation framework evaluating digital Conversion Rate Optimization (CRO) against real-world supply chain economics. Built on **Google Cloud BigQuery** transactional telemetry (`thelook_ecommerce`), object-oriented **Python** inference, and an interactive **Streamlit** diagnostic system adhering to Microsoft Experimentation Platform (ExP) standards.

---

## 📌 Executive Summary & Problem Statement

Digital commerce growth teams frequently optimize front-end checkout conversion in isolation. However, unhedged promises—such as site-wide delivery guarantees—often trigger hidden operational bottlenecks, reverse-logistics surges, and long-term customer attrition.

This project investigates the enterprise impact of a Product Detail Page (PDP) badge promising:
> **"Guaranteed 2-Day Delivery or $15 Account Credit"**

While a naive CRO analysis flags the intervention as a major success due to a **+24.50% conversion lift**, this end-to-end econometrics pipeline reveals that the guarantee caused delivery Service Level Agreement (SLA) breaches to jump to **9.51%** and return bracketing to surge to **12.69%**. By enforcing pre-registered operational guardrails and variance-reduced unit margin estimation (CUPED), the platform **blocks an unhedged sitewide rollout** and guides leadership toward an inventory-aware, geofenced deployment policy.

---

## 🏗️ System Architecture

The pipeline spans four distinct architectural layers:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Cloud Warehouse Layer (Google Cloud BigQuery)                            │
│    - Deterministic FARM_FINGERPRINT cohort assignment (Control vs. Treatment)│
│    - Defensive multi-stage CTE pre-aggregations (resolving Cartesian fan-out)│
│    - Great-circle Haversine geospatial distance calculation to regional DCs  │
│    - Physical anomaly filtering (shipping duration >= 4 hours)              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Automated Quality Assurance & Schema Assertion (Pytest)                  │
│    - Strict grain uniqueness checks on user_id                              │
│    - Non-negative financial bounds (Revenue, COGS, Freight, Returns)        │
│    - Relational order integrity (Returns & SLA breaches <= Shipped items)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Statistical Inference & Econometric Modeling (Python / SciPy / Statsmodels)│
│    - Data Quality Gate: Pearson's Chi-Square Sample Ratio Mismatch (SRM)    │
│    - Two-Proportion Z-Test on checkout conversion progression               │
│    - Multi-Covariate CUPED via Ordinary Least Squares (OLS) regression     │
│    - Operational Guardrail Contingency Audits with Benjamini-Hochberg FDR   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Executive Decision System & Presentation Layer (Streamlit)               │
│    - Dual Evaluation Modes: Baseline A/A Audit vs. Counterfactual Treatment │
│    - Dynamic Plotly guardrail diagnostics & metric accreditations           │
│    - Pre-registered operational rollback circuit breakers                   │
└─────────────────────────────────────────────────────────────────────────────┘
