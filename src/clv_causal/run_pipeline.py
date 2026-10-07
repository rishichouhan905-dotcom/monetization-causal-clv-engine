import os
import sys
import time

from clv_causal.data.download_dataset import ingest_data
from clv_causal.warehouse.build_warehouse import build_star_schema
from clv_causal.clv_engine.clv_pipeline import run_clv_pipeline
from clv_causal.causal_engine.intervention_simulator import simulate_causal_intervention
from clv_causal.causal_engine.did_estimator import run_did_analysis
from clv_causal.causal_engine.synthetic_control import run_synthetic_control
from clv_causal.causal_engine.decision_engine import run_decision_engine

def run_full_pipeline():
    start_time = time.time()
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
    print("\n--- PHASE 3: Policy Intervention Simulation & DiD / Multi-Donor Synthetic Control ---")
    simulate_causal_intervention(intervention_date="2010-06-01", policy_lift_pct=0.15)
    run_did_analysis()
    run_synthetic_control()
    
    # Phase 4: Strategy & Decision Studio Engine
    print("\n--- PHASE 4: Strategic Customer Segmentation & Financial Decision Layer ---")
    run_decision_engine()
    
    elapsed = time.time() - start_time
    print("\n==========================================================================")
    print(f"PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("==========================================================================")
    print("Launch BI Application via: streamlit run app/main.py")

if __name__ == "__main__":
    run_full_pipeline()
