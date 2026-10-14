"""Fast synthetic unit tests for causal simulation, TWFE solvers, Wald pre-trend tests, and CLV quadrature.

These tests execute in under 5 seconds total, require no warehouse or external raw data,
and remain unmarked so they run in standard fast CI pipelines.
"""

import os
import tempfile
import duckdb
from linearmodels.panel import PanelOLS
import numpy as np
import pandas as pd
from scipy.special import betaln
from lifetimes import BetaGeoFitter

from causal.event_study import run_event_study
from causal.simulate_policy import simulate_policy_intervention
from causal.twfe_did import _estimate_twfe_did_sparse


def test_synthetic_simulator_panel_structure_and_realized_effects() -> None:
    """Tests policy intervention simulator panel structure and realized effects on synthetic data."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        tmp_db = f.name
    if os.path.exists(tmp_db):
        os.remove(tmp_db)

    try:
        # Create small synthetic fact_orders: 50 customers over 30 weekly periods
        conn = duckdb.connect(tmp_db)
        records = []
        np.random.seed(42)
        dates = pd.date_range("2011-01-02", "2011-08-01", freq="W-SUN")
        for cust_id in range(1, 51):
            for d in dates:
                if np.random.rand() < 0.5:
                    records.append({
                        "customer_id": cust_id,
                        "invoice_no": f"INV-{cust_id}-{d.strftime('%Y%m%d')}",
                        "invoice_date": d,
                        "line_total": float(np.random.uniform(20.0, 100.0)),
                    })
        df_orders = pd.DataFrame(records)
        conn.execute("CREATE TABLE fact_orders AS SELECT * FROM df_orders")
        conn.close()

        treatment_share = 0.40
        aov_change = 0.10
        order_rate_change = 0.15

        panel_df = simulate_policy_intervention(
            db_path=tmp_db,
            intervention_date="2011-05-01",
            treatment_share=treatment_share,
            order_rate_change=order_rate_change,
            aov_change=aov_change,
            churn_scenario="baseline",
            top_spend_churn_cumulative=0.20,
            random_seed=42,
            persist_to_db=False,
        )

        # 1. Structure assertions
        assert len(panel_df) > 0, "Expected non-empty panel"
        assert panel_df["customer_id"].nunique() == 50, "Expected all 50 synthetic customers"
        required_cols = [
            "customer_id",
            "week_date",
            "is_treated",
            "is_post_period",
            "counterfactual_weekly_spend",
            "observed_weekly_spend",
            "realized_effect_spend",
            "counterfactual_weekly_orders",
            "observed_weekly_orders",
        ]
        for col in required_cols:
            assert col in panel_df.columns, f"Missing column: {col}"

        # 2. Treatment assignment share assertion (exactly 40%)
        cust_treat = panel_df.groupby("customer_id")["is_treated"].first()
        actual_share = float(cust_treat.mean())
        assert abs(actual_share - treatment_share) < 0.05, f"Treatment share {actual_share} deviates from {treatment_share}"

        # 3. Pre-period & control non-interference
        pre_or_ctrl = (panel_df["is_treated"] == 0) | (panel_df["is_post_period"] == 0)
        assert (panel_df.loc[pre_or_ctrl, "realized_effect_spend"] == 0.0).all(), "Pre/control effects must be 0"

        # 4. Positive-purchase week AOV scaling: observed spend = cf_spend * (1 + aov_change)
        active_tr_pos = (
            (panel_df["is_treated"] == 1)
            & (panel_df["is_post_period"] == 1)
            & (~panel_df["is_churned"])
            & (panel_df["counterfactual_weekly_orders"] > 0)
        )
        if active_tr_pos.sum() > 0:
            obs_spend = panel_df.loc[active_tr_pos, "observed_weekly_spend"].values
            cf_spend = panel_df.loc[active_tr_pos, "counterfactual_weekly_spend"].values
            assert np.allclose(obs_spend, cf_spend * (1.0 + aov_change), rtol=1e-3)

    finally:
        if os.path.exists(tmp_db):
            os.remove(tmp_db)


def test_synthetic_twfe_solver_vs_panel_ols() -> None:
    """Compares the sparse TWFE FWL solver against linearmodels.PanelOLS on synthetic panel data (diff < 1e-6)."""
    n_entities = 100
    n_periods = 20
    np.random.seed(123)

    cust_ids = np.repeat(np.arange(1, n_entities + 1), n_periods)
    week_dates = np.tile(pd.date_range("2011-01-01", periods=n_periods, freq="W-SUN"), n_entities)

    alpha_i = np.random.normal(50.0, 10.0, size=n_entities)
    alpha_map = dict(zip(range(1, n_entities + 1), alpha_i))
    gamma_t = np.random.normal(0.0, 5.0, size=n_periods)
    gamma_map = dict(zip(pd.date_range("2011-01-01", periods=n_periods, freq="W-SUN"), gamma_t))

    treated_custs = set(np.random.choice(range(1, n_entities + 1), size=int(0.4 * n_entities), replace=False))
    is_treated = np.array([1 if c in treated_custs else 0 for c in cust_ids])
    is_post = np.array([1 if w >= pd.to_datetime("2011-03-01") else 0 for w in week_dates])
    treat_x_post = is_treated * is_post

    true_tau = 7.50
    noise = np.random.normal(0.0, 2.0, size=n_entities * n_periods)
    y = np.array([alpha_map[c] + gamma_map[w] for c, w in zip(cust_ids, week_dates)]) + true_tau * treat_x_post + noise

    df = pd.DataFrame({
        "customer_id": cust_ids,
        "week_date": week_dates,
        "is_treated": is_treated,
        "is_post_period": is_post,
        "treat_x_post": treat_x_post,
        "y": y,
    })

    # 1. Fit with custom sparse solver
    sparse_res = _estimate_twfe_did_sparse(df, outcome_col="y")

    # 2. Fit with PanelOLS
    df_p = df.set_index(["customer_id", "week_date"])
    mod = PanelOLS(df_p["y"], df_p[["treat_x_post"]], entity_effects=True, time_effects=True, drop_absorbed=True)
    p_res = mod.fit(cov_type="clustered", cluster_entity=True)

    diff_att = abs(sparse_res["estimated_att"] - float(p_res.params["treat_x_post"]))
    diff_se = abs(sparse_res["std_err"] - float(p_res.std_errors["treat_x_post"]))

    assert diff_att < 1e-6, f"ATT difference {diff_att:.2e} exceeds 1e-6 tolerance"
    assert diff_se < 1e-6, f"SE difference {diff_se:.2e} exceeds 1e-6 tolerance"


def test_synthetic_wald_pre_trend_random_vs_confounded() -> None:
    """Verifies that the joint Wald pre-trend test passes on random assignment and rejects on confounded assignment."""
    np.random.seed(42)
    n_entities = 100
    n_periods = 20
    cust_ids = np.repeat(np.arange(1, n_entities + 1), n_periods)
    dates_unique = pd.date_range("2011-01-02", periods=n_periods, freq="W-SUN")
    week_dates = np.tile(dates_unique, n_entities)

    treated_custs = set(np.random.choice(range(1, n_entities + 1), size=int(0.4 * n_entities), replace=False))
    is_treated = np.array([1 if c in treated_custs else 0 for c in cust_ids])

    alpha_i = np.random.normal(50.0, 5.0, size=n_entities)
    alpha_map = dict(zip(range(1, n_entities + 1), alpha_i))
    gamma_t = np.random.normal(0.0, 2.0, size=n_periods)
    gamma_map = dict(zip(dates_unique, gamma_t))

    t0 = dates_unique[10]
    t0_str = t0.strftime("%Y-%m-%d")

    # Case A: Random assignment (parallel pre-trends + noise)
    y_rand = np.array([alpha_map[c] + gamma_map[w] for c, w in zip(cust_ids, week_dates)]) + np.random.normal(0.0, 1.5, size=n_entities * n_periods)
    df_rand = pd.DataFrame({
        "customer_id": cust_ids,
        "week_date": week_dates,
        "is_treated": is_treated,
        "observed_weekly_spend_winsor": y_rand,
    })

    res_rand = run_event_study(panel_df=df_rand, intervention_date=t0_str, max_leads=5, max_lags=5)
    pval_rand = res_rand["joint_test_pval"]
    assert pval_rand > 0.05, f"Expected random assignment to pass parallel trends (p > 0.05), got p={pval_rand:.4f}"

    # Case B: Confounded assignment (differential pre-period trend of +3.0/wk for treated units)
    weeks_rel = np.array([(w - t0).days // 7 for w in week_dates])
    pre_mask = weeks_rel < 0
    confound_trend = np.where(pre_mask & (is_treated == 1), 3.0 * weeks_rel, 0.0)
    y_conf = y_rand + confound_trend

    df_conf = pd.DataFrame({
        "customer_id": cust_ids,
        "week_date": week_dates,
        "is_treated": is_treated,
        "observed_weekly_spend_winsor": y_conf,
    })

    res_conf = run_event_study(panel_df=df_conf, intervention_date=t0_str, max_leads=5, max_lags=5)
    pval_conf = res_conf["joint_test_pval"]
    assert pval_conf < 0.01, f"Expected confounded assignment to reject parallel trends (p < 0.01), got p={pval_conf:.4e}"


def test_synthetic_quadrature_vs_closed_form() -> None:
    """Verifies that Gauss-Legendre quadrature matches the analytical closed form for F=0 when a+b > 1 (diff < 1e-5)."""
    # Known fixed parameters with a + b > 1
    r, alpha, a, b = 0.5, 8.0, 1.2, 2.5
    assert a + b > 1.0

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.params_ = pd.Series({"r": r, "alpha": alpha, "a": a, "b": b})

    t_vals = np.array([10.0, 30.0, 90.0, 180.0, 365.0])
    holdout_days = 90.0

    # 1. Closed form evaluation
    closed_form = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            holdout_days, np.zeros_like(t_vals), np.zeros_like(t_vals), t_vals
        )
    )

    # 2. Custom 100-node Gauss-Legendre Quadrature
    nodes, weights = np.polynomial.legendre.leggauss(100)
    u_quad = 0.5 * (nodes + 1.0)
    w_quad = 0.5 * weights
    p_quad = u_quad ** (1.0 / a)
    p_quad = np.maximum(1e-15, np.minimum(1.0 - 1e-15, p_quad))

    log_w = (b - 1.0) * np.log(1.0 - p_quad) - np.log(a) - betaln(a, b)
    w = np.exp(log_w)

    t_mat = t_vals[:, None]
    p_mat = p_quad[None, :]

    z = p_mat * holdout_days / (alpha + t_mat)
    ratio = (alpha + t_mat) / (alpha + t_mat + p_mat * holdout_days)
    term = np.where(z < 1e-6, r * holdout_days / (alpha + t_mat) * (1.0 - 0.5 * (r + 1) * z), (1.0 - ratio**r) / p_mat)

    quad_res = np.sum(term * w[None, :] * w_quad[None, :], axis=1)

    max_abs_diff = float(np.max(np.abs(closed_form - quad_res)))
    assert max_abs_diff < 1e-5, f"Quadrature difference {max_abs_diff:.2e} exceeds 1e-5 tolerance"
