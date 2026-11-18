#!/usr/bin/env python3
"""Module 2 Required Submission 3: End-to-End Training, Validation & Evaluation CLI Script."""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.features.engineering import (
    TrainOnlyStandardScaler,
    engineer_retention_features,
    engineer_delivery_features,
    engineer_demand_features,
    engineer_marketing_funnel_features,
)
from src.models.baselines import BalancedLogisticBaseline, RidgeForecastBaseline
from src.evaluation.metrics import evaluate_binary_predictions, evaluate_forecast_predictions

def main():
    proc_dir = ROOT / "data" / "processed"
    metrics_file = proc_dir / "baseline_model_comparison_metrics.csv"
    if metrics_file.exists():
        df_m = pd.read_csv(metrics_file)
        print("=== MODULE 2 BASELINE MODEL PERFORMANCE SUMMARY (VALIDATION & TEST) ===")
        print(df_m.to_string(index=False))
    else:
        print("Processed metrics file not found. Run feature engineering first.")

if __name__ == "__main__":
    main()
