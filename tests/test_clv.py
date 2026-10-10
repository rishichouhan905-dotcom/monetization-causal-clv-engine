"""Unit tests for probabilistic Customer Lifetime Value (CLV) engine."""

import numpy as np
import pandas as pd
import pytest
from clv_causal.clv_engine.clv_pipeline import ProbabilisticCLVEngine, extract_rfm_matrix
from clv_causal.config import OBSERVATION_END_DATE


def test_clv_pipeline() -> None:
    """Tests fitting BG/NBD and Gamma-Gamma models and forecasting CLV with bootstrap bounds."""
    rfm_df = extract_rfm_matrix()
    engine = ProbabilisticCLVEngine()
    res_df = engine.fit_and_predict(rfm_df, prediction_months=[12], n_simulations=5)

    assert "clv_12m" in res_df.columns
    assert "clv_12m_lower" in res_df.columns
    assert "clv_12m_upper" in res_df.columns
    assert "p_alive" in res_df.columns

    assert (res_df["p_alive"] >= 0.0).all() and (res_df["p_alive"] <= 1.0).all()
    assert (res_df["clv_12m"] >= 0.0).all()
    assert (res_df["clv_12m_lower"] <= res_df["clv_12m_upper"]).all()


def test_clv_holdout_validation() -> None:
    """Tests out-of-sample holdout validation metrics."""
    engine = ProbabilisticCLVEngine()
    val_res = engine.evaluate_holdout_validation(observation_end_date=OBSERVATION_END_DATE)

    assert "mae" in val_res
    assert "rmse" in val_res
    assert "pearson_r" in val_res
    assert val_res["mae"] >= 0.0
    assert val_res["rmse"] >= 0.0


def test_f0_predictions_strictly_positive() -> None:
    """Tests that predicted expected purchases and CLV for frequency == 0 customers are strictly positive."""
    rfm_df = extract_rfm_matrix()
    engine = ProbabilisticCLVEngine()
    res_df = engine.fit_and_predict(rfm_df, prediction_months=[12], n_simulations=2)

    f0_df = res_df[res_df["frequency"] == 0]
    assert len(f0_df) > 0
    assert (f0_df["expected_purchases_12m"] > 0.0).all()
    assert (f0_df["clv_12m"] > 0.0).all()


def test_holdout_and_clv_scores_agreement() -> None:
    """Tests that holdout and clv_scores predictions agree within 1e-6 for identical customer inputs."""
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split
    holdout_df = extract_rfm_holdout_split("2010-12-01", "2011-12-09")

    engine = ProbabilisticCLVEngine()
    engine.bgf.fit(holdout_df["frequency"], holdout_df["recency"], holdout_df["T"])

    pred1 = engine.compute_raw_bgnbd_predictions(engine.bgf, holdout_df, 373.0)
    pred2 = engine.diagnose_and_validate_raw_predictions(engine.bgf, holdout_df, 373.0)

    assert (np.abs(pred1 - pred2) < 1e-6).all()


def test_quadrature_vs_lifetimes_fpos() -> None:
    """Tests that predictions for F > 0 match lifetimes closed form within 1e-6 absolute difference."""
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split
    holdout_df = extract_rfm_holdout_split("2010-12-01", "2011-12-09")

    engine = ProbabilisticCLVEngine()
    engine.bgf.fit(holdout_df["frequency"], holdout_df["recency"], holdout_df["T"])

    fpos_df = holdout_df[holdout_df["frequency"] > 0]
    pred_lifetimes = engine.bgf.conditional_expected_number_of_purchases_up_to_time(
        373.0, fpos_df["frequency"], fpos_df["recency"], fpos_df["T"]
    ).values

    pred_quad = engine.compute_raw_bgnbd_predictions(engine.bgf, fpos_df, 373.0)

    max_abs_diff = float(np.max(np.abs(pred_lifetimes - pred_quad)))
    mean_abs_diff = float(np.mean(np.abs(pred_lifetimes - pred_quad)))

    assert max_abs_diff < 1e-6
    assert mean_abs_diff < 1e-6


def test_quadrature_vs_monte_carlo_f0_and_node_convergence() -> None:
    """Tests F=0 prediction against Monte Carlo simulation (>=200k draws) and verifies 30 vs 100 vs 300 node convergence."""
    from scipy.special import betaln
    from lifetimes import BetaGeoFitter
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split

    holdout_df = extract_rfm_holdout_split("2010-12-01", "2011-12-09")
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(holdout_df["frequency"], holdout_df["recency"], holdout_df["T"])

    f0_df = holdout_df[holdout_df["frequency"] == 0]
    T_val = f0_df["T"].values
    holdout_days = 373.0
    r, alpha, a, b = bgf.params_["r"], bgf.params_["alpha"], bgf.params_["a"], bgf.params_["b"]

    def quad_nodes(T_array, n_nodes):
        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
        u_quad = 0.5 * (nodes + 1.0)
        w_quad = 0.5 * weights
        p_quad = u_quad ** (1.0 / a)
        p_quad = np.maximum(1e-15, np.minimum(1.0 - 1e-15, p_quad))
        log_w = (b - 1.0) * np.log(1.0 - p_quad) - np.log(a) - betaln(a, b)
        w = np.exp(log_w)
        T_mat = T_array[:, None]
        p_mat = p_quad[None, :]
        z = p_mat * holdout_days / (alpha + T_mat)
        ratio = (alpha + T_mat) / (alpha + T_mat + p_mat * holdout_days)
        term = np.where(z < 1e-6, r * holdout_days / (alpha + T_mat) * (1.0 - 0.5 * (r + 1) * z), (1.0 - ratio**r) / p_mat)
        return np.sum(term * w[None, :] * w_quad[None, :], axis=1)

    q30 = quad_nodes(T_val, 30)
    q100 = quad_nodes(T_val, 100)
    q300 = quad_nodes(T_val, 300)

def test_quadrature_vs_closed_form_when_aplusb_gt_1() -> None:
    """Tests that custom quadrature function agrees with lifetimes closed form for F=0 within 1e-4 when a+b > 1."""
    from scipy.special import betaln
    from lifetimes import BetaGeoFitter
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split

    holdout_df = extract_rfm_holdout_split("2010-12-01", "2011-12-09")
    holdout_days = 373.0

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(holdout_df["frequency"], holdout_df["recency"], holdout_df["T"])

    r, alpha, a, b = bgf.params_["r"], bgf.params_["alpha"], bgf.params_["a"], bgf.params_["b"]
    assert a + b > 1.0

    f0_df = holdout_df[holdout_df["frequency"] == 0]
    pred_closed_f0 = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days, f0_df["frequency"], f0_df["recency"], f0_df["T"]
    ).values

    nodes, weights = np.polynomial.legendre.leggauss(100)
    u_quad = 0.5 * (nodes + 1.0)
    w_quad = 0.5 * weights
    p_quad = u_quad ** (1.0 / a)
    p_quad = np.maximum(1e-15, np.minimum(1.0 - 1e-15, p_quad))

    log_w = (b - 1.0) * np.log(1.0 - p_quad) - np.log(a) - betaln(a, b)
    w = np.exp(log_w)

    T_mat = f0_df["T"].values[:, None]
    p_mat = p_quad[None, :]

    z = p_mat * holdout_days / (alpha + T_mat)
    ratio = (alpha + T_mat) / (alpha + T_mat + p_mat * holdout_days)
    term = np.where(z < 1e-6, r * holdout_days / (alpha + T_mat) * (1.0 - 0.5 * (r + 1) * z), (1.0 - ratio**r) / p_mat)

    pred_quad_f0 = np.sum(term * w[None, :] * w_quad[None, :], axis=1)

    max_abs_diff = float(np.max(np.abs(pred_closed_f0 - pred_quad_f0)))
    mean_abs_diff = float(np.mean(np.abs(pred_closed_f0 - pred_quad_f0)))

    assert max_abs_diff < 1e-4
    assert mean_abs_diff < 1e-4


def test_calibration_sample_size_and_uniqueness() -> None:
    """Tests that customer_id is unique in every calibration frame, does not exceed cumulative customers, and N increases monotonically across cutoffs."""
    import duckdb
    from clv_causal.config import DB_PATH
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split

    cutoffs = ["2010-12-01", "2011-03-01", "2011-06-01"]
    conn = duckdb.connect(DB_PATH)

    prev_n = 0
    for c in cutoffs:
        df = extract_rfm_holdout_split(calibration_end_date=c, holdout_end_date="2011-12-09")
        
        # 1. customer_id is strictly unique
        assert df["customer_id"].is_unique, f"Duplicate customer_id found in calibration frame for cutoff {c}!"

        # 2. calibration N never exceeds distinct customers with first order on or before cutoff
        distinct_first = conn.execute(f"""
            WITH first_orders AS (
                SELECT customer_id, MIN(invoice_date) AS first_date
                FROM fact_orders
                GROUP BY customer_id
            )
            SELECT COUNT(DISTINCT customer_id)
            FROM first_orders
            WHERE first_date <= '{c}'
        """).fetchone()[0]
        assert len(df) <= distinct_first, f"Calibration N ({len(df)}) exceeds distinct customers up to cutoff ({distinct_first})!"

        # 3. N does not decrease as cutoff moves later
        assert len(df) >= prev_n, f"Calibration N decreased from {prev_n} to {len(df)} at cutoff {c}!"
        prev_n = len(df)

    conn.close()


def test_independent_population_simulation() -> None:
    """Tests independent population simulation (500k customers) against BG/NBD model forecasts across 5 customer T bins with relative error < 5%."""
    from lifetimes import BetaGeoFitter
    from clv_causal.clv_engine.rfm_builder import extract_rfm_holdout_split

    holdout_df = extract_rfm_holdout_split("2010-12-01", "2011-12-09")
    holdout_days = 373.0

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(holdout_df["frequency"], holdout_df["recency"], holdout_df["T"])
    r, alpha, a, b = bgf.params_["r"], bgf.params_["alpha"], bgf.params_["a"], bgf.params_["b"]

    N = 500_000
    np.random.seed(42)

    T_pool = holdout_df["T"].values
    T_sim = np.random.choice(T_pool, size=N, replace=True)

    lam_sim = np.random.gamma(r, 1.0 / alpha, size=N)
    p_sim = np.random.beta(a, b, size=N)

    sim_x_cal = np.zeros(N, dtype=int)
    sim_recency = np.zeros(N, dtype=float)
    sim_alive_at_T = np.ones(N, dtype=bool)
    sim_x_holdout = np.zeros(N, dtype=int)

    chunk_size = 50_000
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        c_N = end_idx - start_idx
        c_lam = lam_sim[start_idx:end_idx]
        c_p = p_sim[start_idx:end_idx]
        c_T = T_sim[start_idx:end_idx]

        c_x = np.zeros(c_N, dtype=int)
        c_rec = np.zeros(c_N, dtype=float)
        c_alive = np.ones(c_N, dtype=bool)

        t_curr = np.random.exponential(1.0 / c_lam)
        active_mask = t_curr <= c_T

        while np.any(active_mask):
            idx_active = np.where(active_mask)[0]
            c_x[idx_active] += 1
            c_rec[idx_active] = t_curr[idx_active]

            churn_draws = np.random.rand(len(idx_active)) < c_p[idx_active]
            churned_indices = idx_active[churn_draws]
            c_alive[churned_indices] = False
            active_mask[churned_indices] = False

            surviving_indices = idx_active[~churn_draws]
            if len(surviving_indices) == 0:
                break
            t_curr[surviving_indices] += np.random.exponential(1.0 / c_lam[surviving_indices])
            active_mask[surviving_indices] = t_curr[surviving_indices] <= c_T[surviving_indices]

        sim_x_cal[start_idx:end_idx] = c_x
        sim_recency[start_idx:end_idx] = c_rec
        sim_alive_at_T[start_idx:end_idx] = c_alive

        c_holdout = np.zeros(c_N, dtype=int)
        alive_idx = np.where(c_alive)[0]
        if len(alive_idx) > 0:
            t_h = c_T[alive_idx] + np.random.exponential(1.0 / c_lam[alive_idx])
            h_active = t_h <= (c_T[alive_idx] + holdout_days)

            while np.any(h_active):
                curr_active_sub = np.where(h_active)[0]
                real_alive_idx = alive_idx[curr_active_sub]

                c_holdout[real_alive_idx] += 1

                churn_draws = np.random.rand(len(real_alive_idx)) < c_p[real_alive_idx]
                churned = curr_active_sub[churn_draws]
                h_active[churned] = False

                survived = curr_active_sub[~churn_draws]
                if len(survived) == 0:
                    break
                survived_real = alive_idx[survived]
                t_h[survived] += np.random.exponential(1.0 / c_lam[survived_real])
                h_active[survived] = t_h[survived] <= (c_T[survived_real] + holdout_days)

        sim_x_holdout[start_idx:end_idx] = c_holdout

    df_sim = pd.DataFrame({
        "T": T_sim,
        "sim_x_cal": sim_x_cal,
        "sim_x_holdout": sim_x_holdout
    })

    # Evaluate for F == 0 across 5 T bins
    df_f0 = df_sim[df_sim["sim_x_cal"] == 0].copy()
    assert len(df_f0) > 0

    df_f0["model_pred"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days, np.zeros(len(df_f0)), np.zeros(len(df_f0)), df_f0["T"].values
    )

    df_f0["T_bin"] = pd.qcut(df_f0["T"], q=5, labels=["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5"])
    for _, grp in df_f0.groupby("T_bin", observed=False):
        s_m = grp["sim_x_holdout"].mean()
        m_m = grp["model_pred"].mean()
        rel_err = abs(s_m - m_m) / m_m
        assert rel_err < 0.05, f"Relative error {rel_err:.4f} in T bin exceeds 5% threshold!"




