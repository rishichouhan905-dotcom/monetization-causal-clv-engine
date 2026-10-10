"""Policy Intervention Simulator Wrapper.

Delegates simulation logic to src/causal/simulate_policy.py for randomized
treatment policy interventions with known multiplicative effects.
"""

from typing import Dict, Optional
import pandas as pd

from causal.simulate_policy import simulate_policy_intervention
from clv_causal.config import (
    AOV_CHANGE,
    DB_PATH,
    INTERVENTION_DATE,
    ORDER_RATE_CHANGE,
    RANDOM_SEED,
    TOP_CLV_CHURN_HAZARD,
    TREATMENT_SHARE,
)


def simulate_causal_intervention(
    intervention_date: str = INTERVENTION_DATE,
    treatment_share: float = TREATMENT_SHARE,
    order_rate_change: float = ORDER_RATE_CHANGE,
    aov_change: float = AOV_CHANGE,
    top_clv_churn_hazard: float = TOP_CLV_CHURN_HAZARD,
    random_seed: int = RANDOM_SEED,
    db_path: str = DB_PATH,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper function to trigger randomized policy simulation.

    Args:
        intervention_date: ISO date string for policy rollout date t_0.
        treatment_share: Fraction of customers randomly assigned to treatment.
        order_rate_change: Relative change in order frequency.
        aov_change: Relative change in average order value.
        top_clv_churn_hazard: Hazard rate for top CLV quintile churn.
        random_seed: Seed for random draws.
        db_path: Path to DuckDB database.

    Returns:
        DataFrame containing customer-by-week panel with true counterfactuals and observed values.
    """
    return simulate_policy_intervention(
        db_path=db_path,
        intervention_date=intervention_date,
        treatment_share=treatment_share,
        order_rate_change=order_rate_change,
        aov_change=aov_change,
        top_clv_churn_hazard=top_clv_churn_hazard,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    simulate_causal_intervention()
