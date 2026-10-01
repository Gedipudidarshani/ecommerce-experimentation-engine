from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class EvaluationResults:
    srm_passed: bool
    srm_p_value: float
    sample_sizes: Dict[str, int]
    cr_control: float
    cr_treatment: float
    cr_lift_pct: float
    cr_p_value: float
    raw_delta_ncmu: float
    raw_ncmu_p_value: float
    cuped_delta_ncmu: float
    cuped_ncmu_p_value: float
    variance_reduction_pct: float
    cuped_ci: Tuple[float, float]
    returns_breached: bool
    returns_control_rate: float
    returns_treatment_rate: float
    returns_fdr_p: float
    sla_breached: bool
    sla_control_rate: float
    sla_treatment_rate: float
    sla_fdr_p: float


class ExperimentationPlatformEngine:
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        self.df = data.copy()
        self.alpha = alpha

    def run(self) -> EvaluationResults:
        ctrl_mask = self.df['variant'] == 'Control'
        treat_mask = self.df['variant'] == 'Treatment'
        n_c = int(ctrl_mask.sum())
        n_t = int(treat_mask.sum())
        total_n = n_c + n_t

        # 1. Sample Ratio Mismatch (SRM) Goodness-of-Fit
        _, p_srm = stats.chisquare(f_obs=[n_c, n_t], f_exp=[total_n / 2.0, total_n / 2.0])
        srm_pass = bool(p_srm >= 0.001)

        # 2. Conversion Z-Test
        conv_c = int(self.df.loc[ctrl_mask, 'converted'].sum())
        conv_t = int(self.df.loc[treat_mask, 'converted'].sum())
        _, p_conv = proportions_ztest([conv_t, conv_c], [n_t, n_c], alternative='two-sided')
        cr_c = conv_c / max(1, n_c)
        cr_t = conv_t / max(1, n_t)
        cr_lift = ((cr_t - cr_c) / max(1e-9, cr_c)) * 100.0

        # 3. Raw Welch's Two-Sample T-Test (NCMU)
        y_c = self.df.loc[ctrl_mask, 'net_contribution_margin'].to_numpy()
        y_t = self.df.loc[treat_mask, 'net_contribution_margin'].to_numpy()
        _, p_raw = stats.ttest_ind(y_t, y_c, equal_var=False)
        raw_delta = float(np.mean(y_t) - np.mean(y_c))

        # 4. Multi-Covariate CUPED (Pre-Spend + Pre-Sessions)
        cov_matrix = self.df[['pre_spend_covariate', 'pre_sessions_covariate']].to_numpy()
        y_target = self.df['net_contribution_margin'].to_numpy()

        X_design = sm.add_constant(cov_matrix)
        ols_model = sm.OLS(y_target, X_design).fit()
        thetas = ols_model.params[1:]

        cov_centered = cov_matrix - np.mean(cov_matrix, axis=0)
        self.df['cuped_margin'] = y_target - np.dot(cov_centered, thetas)

        y_cuped_c = self.df.loc[ctrl_mask, 'cuped_margin'].to_numpy()
        y_cuped_t = self.df.loc[treat_mask, 'cuped_margin'].to_numpy()
        _, p_cuped = stats.ttest_ind(y_cuped_t, y_cuped_c, equal_var=False)
        cuped_delta = float(np.mean(y_cuped_t) - np.mean(y_cuped_c))

        var_raw_t = float(np.var(y_t, ddof=1)) if len(y_t) > 1 else 1.0
        var_cuped_t = float(np.var(y_cuped_t, ddof=1)) if len(y_cuped_t) > 1 else 1.0
        var_red_pct = max(0.0, (1.0 - (var_cuped_t / max(1e-9, var_raw_t))) * 100.0)

        se_cuped = np.sqrt((var_cuped_t / max(1, n_t)) + (float(np.var(y_cuped_c, ddof=1)) / max(1, n_c)))
        ci_lower = cuped_delta - 1.96 * se_cuped
        ci_upper = cuped_delta + 1.96 * se_cuped

        # 5. Operational Guardrails with Benjamini-Hochberg FDR
        ship_c = int(self.df.loc[ctrl_mask, 'total_shipped'].sum())
        ship_t = int(self.df.loc[treat_mask, 'total_shipped'].sum())
        ret_c = int(self.df.loc[ctrl_mask, 'total_returned'].sum())
        ret_t = int(self.df.loc[treat_mask, 'total_returned'].sum())

        table_ret = [[ret_t, max(0, ship_t - ret_t)], [ret_c, max(0, ship_c - ret_c)]]
        _, p_ret, _, _ = stats.chi2_contingency(table_ret)

        breach_c = int(self.df.loc[ctrl_mask, 'sla_breaches'].sum())
        breach_t = int(self.df.loc[treat_mask, 'sla_breaches'].sum())
        table_sla = [[breach_t, max(0, ship_t - breach_t)], [breach_c, max(0, ship_c - breach_c)]]
        _, p_sla, _, _ = stats.chi2_contingency(table_sla)

        reject_flags, fdr_pvals, _, _ = multipletests([p_ret, p_sla], alpha=self.alpha, method='fdr_bh')

        rate_ret_c = (ret_c / max(1, ship_c)) * 100.0
        rate_ret_t = (ret_t / max(1, ship_t)) * 100.0
        rate_sla_c = (breach_c / max(1, ship_c)) * 100.0
        rate_sla_t = (breach_t / max(1, ship_t)) * 100.0

        return EvaluationResults(
            srm_passed=srm_pass,
            srm_p_value=float(p_srm),
            sample_sizes={'Control': n_c, 'Treatment': n_t},
            cr_control=cr_c * 100.0,
            cr_treatment=cr_t * 100.0,
            cr_lift_pct=cr_lift,
            cr_p_value=float(p_conv),
            raw_delta_ncmu=raw_delta,
            raw_ncmu_p_value=float(p_raw),
            cuped_delta_ncmu=cuped_delta,
            cuped_ncmu_p_value=float(p_cuped),
            variance_reduction_pct=var_red_pct,
            cuped_ci=(ci_lower, ci_upper),
            returns_breached=bool(reject_flags[0] and rate_ret_t > rate_ret_c),
            returns_control_rate=rate_ret_c,
            returns_treatment_rate=rate_ret_t,
            returns_fdr_p=float(fdr_pvals[0]),
            sla_breached=bool(reject_flags[1] and rate_sla_t > rate_sla_c),
            sla_control_rate=rate_sla_c,
            sla_treatment_rate=rate_sla_t,
            sla_fdr_p=float(fdr_pvals[1])
        )