#!/usr/bin/env python3
"""Reproducible Module 1 Data Quality, Statistical Investigation & Leakage Audit Pipeline."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def verify_delivery_leakage_free(df: pd.DataFrame, feature_cols: list) -> bool:
    """Assert that post-dispatch outcome columns are excluded from predictor list."""
    forbidden = {"actual_delivery_days", "delay_days", "late_delivery", "delivery_id"}
    leaked = forbidden.intersection(set(feature_cols))
    if leaked:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {leaked} found in feature_cols!")
    return True

def run_audit(data_dir: str = "data/raw") -> dict:
    """Run statistical and data-quality checks across all Project 02 datasets."""
    results = {}
    cust_path = os.path.join(data_dir, "customer_retention_features.csv")
    if os.path.exists(cust_path):
        df_c = pd.read_csv(cust_path)
        results["customer_retention"] = {
            "rows": len(df_c),
            "missing": int(df_c.isnull().sum().sum()),
            "lapse_rate": round(float(df_c["no_purchase_next_90d"].mean()), 4),
            "return_exceeds_orders_anomalies": int((df_c["returns_last_180d"] > df_c["orders_last_180d"]).sum()),
            "recency_zero_order_anomalies": int(((df_c["days_since_last_order"] <= 180) & (df_c["orders_last_180d"] == 0)).sum()),
        }
    del_path = os.path.join(data_dir, "delivery_delay_training.csv")
    if os.path.exists(del_path):
        df_d = pd.read_csv(del_path)
        results["delivery_delay"] = {
            "rows": len(df_d),
            "missing": int(df_d.isnull().sum().sum()),
            "late_rate": round(float(df_d["late_delivery"].mean()), 4),
            "negative_actual_days_rows": int((df_d["actual_delivery_days"] < 0).sum()),
        }
    return results

if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
