# Enterprise E-Commerce Experimentation Engine & Econometrics Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google BigQuery](https://img.shields.io/badge/warehouse-Google%20BigQuery-4285F4.svg)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end analytics engineering, causal inference, and online experimentation framework evaluating digital Conversion Rate Optimization (CRO) against real-world supply chain economics[cite: 1, 8, 14, 15]. Built on **Google Cloud BigQuery** transactional telemetry (`thelook_ecommerce`), object-oriented **Python** inference, and an interactive **Streamlit** diagnostic system adhering to Microsoft Experimentation Platform (ExP) standards[cite: 1, 10, 15, 19, 20].

---

## 📌 Executive Summary & Problem Statement

Digital commerce growth teams frequently optimize front-end checkout conversion in isolation[cite: 1, 14, 15, 19]. However, unhedged promises—such as site-wide delivery guarantees—often trigger hidden operational bottlenecks, reverse-logistics surges, and long-term customer attrition[cite: 1, 6, 8, 14, 29].

This project investigates the enterprise impact of a Product Detail Page (PDP) badge promising[cite: 8, 19]:
> **"Guaranteed 2-Day Delivery or $15 Account Credit"**[cite: 8, 19]

While a naive CRO analysis flags the intervention as a major success due to a **+24.50% conversion lift**, this end-to-end econometrics pipeline reveals that the guarantee caused delivery Service Level Agreement (SLA) breaches to jump to **9.51%** and return bracketing to surge to **12.69%**[cite: 8, 14, 45, 46]. By enforcing pre-registered operational guardrails and variance-reduced unit margin estimation (CUPED), the platform **blocks an unhedged sitewide rollout** and guides leadership toward an inventory-aware, geofenced deployment policy[cite: 6, 8, 14, 46].

---

## 🏗️ System Architecture

The pipeline spans four distinct architectural layers[cite: 19]:

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
```

---

## 📊 Metric Hierarchy & Decision Taxonomy

Adhering to Microsoft ExP experimentation doctrine, metrics are partitioned into four strict functional tiers[cite: 14, 15, 19]:

| Tier | Metric Name | Statistical Formulation | Decision Role |
| :--- | :--- | :--- | :--- |
| **Data Quality Gate** | Allocation Balance | Pearson's $\chi^2$ Goodness-of-Fit ($\alpha = 0.001$) | Abort pipeline if hash assignment is corrupted[cite: 1, 8, 10, 15, 19] |
| **Local Proxy Metric** | Conversion Rate (CR) | Two-Proportion Z-Test | Measures immediate customer checkout urgency[cite: 1, 8, 10, 15, 19] |
| **Primary OEC** | Adjusted Net Margin (NCMU) | Multi-Covariate CUPED + Welch's T-Test | Primary financial criterion balancing margins & logistics[cite: 8, 10, 14, 15, 19] |
| **Operational Guardrail 1** | Line-Item Return Rate | $2\times2$ Contingency $\chi^2$ + Benjamini-Hochberg FDR | Monitors "wardrobe bracketing" and restock costs[cite: 3, 8, 10, 14, 15, 19] |
| **Operational Guardrail 2** | 48h SLA Breach Rate | $2\times2$ Contingency $\chi^2$ + Benjamini-Hochberg FDR | Protects customer trust and carrier capacity limits[cite: 8, 10, 14, 15, 19] |

---

## 🧮 Mathematical Formulation

### 1. Multi-Covariate CUPED Variance Reduction
Continuous financial metrics (such as Net Contribution Margin per User) exhibit heavy right-skew and zero-inflation[cite: 1, 6, 8, 10]. To narrow confidence interval bounds without extending test duration, the engine implements Controlled-Experiment using Pre-Experiment Data (CUPED) across two orthogonal baseline vectors[cite: 1, 8, 10, 18]:
* $X_1$: 60-day pre-experiment baseline spend[cite: 1, 8, 10]
* $X_2$: 60-day pre-experiment browsing session telemetry[cite: 8, 10, 18]

The unbiased variance-reduced estimator is computed via Ordinary Least Squares (OLS) coefficients $\boldsymbol{\theta}^*$[cite: 8, 10, 12, 18]:

$$\hat{Y}_{i, \text{CUPED}} = Y_i - \boldsymbol{\theta}^{*T} (\mathbf{X}_i - \mathbb{E}[\mathbf{X}])$$
[cite: 8, 10, 18]

Where:
$$\boldsymbol{\theta}^* = \mathbf{\Sigma}_{XX}^{-1} \mathbf{\Sigma}_{XY}$$
[cite: 1, 10, 18]

This adjustment reduces outcome metric variance by a factor of $(1 - R^2)$, preserving statistical power even among cold-start or low-frequency shoppers[cite: 6, 8, 10, 16, 18].

### 2. Distance-Tiered Unit Economics
Net Contribution Margin per User (NCMU) models real-world freight zones and reverse supply chains[cite: 1, 8, 10, 15]:

$$\text{NCMU} = \text{Gross Revenue} - \text{COGS} - \text{Outbound Freight} - \text{Reverse Logistics Cost}$$
[cite: 2, 8, 10, 16, 19]

* **Outbound Freight:** $\$4.50 + (\$0.0035 \times d_{\text{Haversine}})$ per shipped item[cite: 9, 10, 16, 21].
* **Reverse Logistics:** $\$9.00 \text{ fixed intake} + 15\% \text{ markdown inventory depreciation}$ per returned item[cite: 8, 9, 10, 16, 21].

---

## 🧪 Empirical Results & Executive Decision

### Evaluation Summary

| Evaluated Dimension | Control Group | Treatment Group | Statistical Test / Methodology | Decision Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Sample Ratio Mismatch** | $N = 25,053$[cite: 45] | $N = 24,947$[cite: 45] | $\chi^2 = 0.224$, $p = 0.6355$[cite: 45] | **Passed**: Allocation mechanism uncompromised[cite: 45] |
| **Session Conversion (CR)** | $8.24\%$ | $10.26\%$[cite: 45] | $z = 7.781$, $p = 6.59 \times 10^{-15}$[cite: 45] | **Win**: Significant $+24.50\%$ relative conversion lift[cite: 45] |
| **Raw Per-User Margin** | $\$3.84$ | $\$4.83$ | $t = 4.12$, $p < 0.0001$[cite: 45] | **Win**: Raw margin delta of $+\$0.99$ per user[cite: 45] |
| **CUPED Net Margin (OEC)** | $\$3.84$ | $\$4.88$ | $t = 4.31$, $p < 0.0001$[cite: 45] | **Win**: Adjusted delta of $+\$1.04$ ($95\%\text{ CI: } [0.69, 1.39]$)[cite: 45] |
| **Line-Item Return Rate** | $8.20\%$ | $12.69\%$[cite: 46] | $\chi^2 = 49.8$, FDR $p = 1.39 \times 10^{-12}$[cite: 46] | **Tripwire**: Severe return bracketing detected[cite: 8, 46] |
| **48-Hour SLA Breaches** | $2.41\%$ | $9.51\%$[cite: 46] | $\chi^2 = 181.4$, FDR $p = 1.51 \times 10^{-41}$[cite: 46] | **Tripwire**: Transcontinental carrier bottlenecks[cite: 14, 46] |

### Executive Recommendation

```text
🚨 DECISION: BLOCK UNCONDITIONAL SITEWIDE ROLLOUT
```
1. **Halt Sitewide Deployment:** Although checkout conversion and CUPED margin showed statistically significant gains, expedited fulfillment guarantees triggered severe operational failures (+710 bps SLA breach surge and +449 bps return rate spike)[cite: 6, 14, 45, 46].
2. **Deploy Geofenced Badging:** Restrict badge visibility dynamically to customers residing within **600 kilometers** of a regional distribution center where inventory is actively stocked[cite: 6, 14, 46].
3. **Cart Margin Floor:** Enforce a qualifying order threshold of **$85** before granting 2-day delivery guarantees to ensure gross profit absorbs expedited transit costs[cite: 6, 14].

---

## 📁 Repository Structure

```text
ecommerce-experimentation-engine/
├── .gitignore                          # Excludes local venv, cache, and raw data
├── README.md                           # System documentation and econometric findings
├── requirements.txt                    # Pinned production dependencies
├── app.py                              # Streamlit experimentation and diagnostics UI
├── data/
│   └── gold_features.csv               # BigQuery extracted feature mart (local dev)
├── sql/
│   └── extract_cohort_features.sql     # Production BigQuery ETL feature store query
├── src/
│   ├── __init__.py                     # Python package marker
│   └── engine.py                       # ExP statistical engine (SRM, CUPED, FDR)
└── tests/
    └── test_data_quality.py            # Automated Pytest data quality assertions
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12[cite: 28]
* Git[cite: 28]
* Google Cloud account (optional, for running BigQuery queries directly)[cite: 28, 32]

### 2. Installation & Setup

```powershell
# Clone the repository
git clone [https://github.com/Gedipudidarshani/ecommerce-experimentation-engine.git](https://github.com/Gedipudidarshani/ecommerce-experimentation-engine.git)
cd ecommerce-experimentation-engine

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Linux/macOS: source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run Automated Data Quality Tests
Run the defensive test suite to ensure the data mart maintains grain integrity and financial boundaries[cite: 31, 34]:
```powershell
pytest tests/test_data_quality.py
```
Expected output[cite: 34]:
```text
tests/test_data_quality.py ....                                          [100%]
4 passed in 0.82s
```

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser[cite: 26, 28]. Use the sidebar to toggle between:
* **Counterfactual Simulation Mode (ExP Demo):** Demonstrates treatment injection, CUPED adjustments, and guardrail tripwires across 50,000 users[cite: 7, 10, 14].
* **Raw BigQuery Export (A/A Audit):** Validates nominal Type I error calibration ($p > 0.05$) on unmodified historical warehouse data[cite: 6, 7, 10, 14].

---

## 👥 Author
* **Gedipudi Darshani** - *Artificial Intelligence & Data Science*
