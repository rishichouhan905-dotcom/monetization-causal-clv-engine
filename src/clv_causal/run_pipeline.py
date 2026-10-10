"""Master CLI Orchestrator Pipeline for Monetization Causal CLV Engine.

Executes data ingestion, dbt star schema modeling, data quality assertions,
probabilistic CLV modeling, holdout validation, policy simulation, DiD / Synthetic Control estimation,
and executive decision segmentation.
"""

import time


from causal.decision_engine import run_decision_engine
from causal.evaluation import run_full_causal_evaluation
from causal.simulate_policy import simulate_policy_intervention
from clv_causal.clv_engine.clv_pipeline import run_clv_pipeline
from clv_causal.config import INTERVENTION_DATE, POLICY_LIFT_PCT
from clv_causal.data.download_dataset import ingest_data
from clv_causal.warehouse.build_warehouse import build_star_schema


def run_full_pipeline() -> None:
    """Runs end-to-end data engineering, CLV fitting, causal inference, and decision modeling pipeline."""
    start_time: float = time.time()
    print("==========================================================================")
    print("ENTERPRISE PROBABILISTIC CLV & CAUSAL DECISION PLATFORM PIPELINE")
    print("==========================================================================\n")

    # Phase 1: Environment & Warehouse Modeling
    print("--- PHASE 1: Data Ingestion & Warehouse Star Schema Modeling ---")
    ingest_data()
    build_star_schema()

    # Phase 2: Probabilistic CLV Engine & Holdout Validation
    print("\n--- PHASE 2: Probabilistic BG/NBD & Gamma-Gamma CLV Engine & Validation ---")
    run_clv_pipeline()

    # Phase 3: Causal Impact & Quasi-Experimentation
    print("\n--- PHASE 3: Policy Intervention Simulation & Causal Estimators Suite ---")
    simulate_policy_intervention()
    run_full_causal_evaluation()

    # Phase 4: Strategy & Decision Studio Engine
    print("\n--- PHASE 4: Strategic Customer Segmentation & Financial Decision Layer ---")
    run_decision_engine()

    elapsed: float = time.time() - start_time
    print("\n==========================================================================")
    print(f"PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("==========================================================================")
    print("Launch BI Application via: streamlit run app/main.py")


if __name__ == "__main__":
    run_full_pipeline()
