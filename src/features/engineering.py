#!/usr/bin/env python3
"""Module 2: Leakage-Safe Feature Engineering & Multicollinearity (VIF) Pipelines."""
import numpy as np
import pandas as pd

class TrainOnlyStandardScaler:
    """StandardScaler fit strictly on training data to prevent distribution leakage."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.where(np.std(X, axis=0) == 0, 1.0, np.std(X, axis=0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fit on training data before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

def engineer_retention_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct customer-level behavioural and smoothed ratio features."""
    out = pd.DataFrame(index=df.index)
    out["tenure_months"] = df["tenure_months"].astype(float)
    out["orders_last_180d"] = df["orders_last_180d"].astype(float)
    out["log_avg_order_value"] = np.log1p(df["avg_order_value"].astype(float))
    out["days_since_last_order"] = df["days_since_last_order"].astype(float)
    out["discount_order_share"] = df["discount_order_share"].astype(float)
    out["returns_last_180d"] = df["returns_last_180d"].astype(float)
    out["support_tickets_last_180d"] = df["support_tickets_last_180d"].astype(float)
    out["sessions_last_30d"] = df["sessions_last_30d"].astype(float)
    out["smoothed_return_rate"] = out["returns_last_180d"] / (out["orders_last_180d"] + 1.0)
    out["tickets_per_order"] = out["support_tickets_last_180d"] / (out["orders_last_180d"] + 1.0)
    out["flag_recency_zero_order"] = ((df["days_since_last_order"] <= 180) & (df["orders_last_180d"] == 0)).astype(float)
    return out

def engineer_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct dispatch-time delivery features with strict outcome exclusion."""
    forbidden = {"actual_delivery_days", "delay_days"}
    if forbidden.intersection(set(df.columns)):
        df = df.drop(columns=[c for c in forbidden if c in df.columns])
    out = pd.DataFrame(index=df.index)
    out["log_distance_km"] = np.log1p(df["distance_km"].astype(float))
    out["log_package_weight_g"] = np.log1p(df["package_weight_g"].astype(float))
    out["seller_historical_rating"] = df["seller_historical_rating"].astype(float)
    out["carrier_load_index"] = df["carrier_load_index"].astype(float)
    out["promised_delivery_days"] = df["promised_delivery_days"].astype(float)
    out["distance_per_promised_day"] = df["distance_km"].astype(float) / np.maximum(df["promised_delivery_days"].astype(float), 1.0)
    for w in ["heat", "rain", "storm"]:
        out[f"weather_{w}"] = (df["weather_at_dispatch"] == w).astype(float)
    return out

def engineer_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct strictly shifted lag, rolling, and cyclical calendar features per SKU."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    g = df.groupby("product_id")["units_sold"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)
    df["roll_mean_7"] = g.shift(1).groupby(df["product_id"]).transform(lambda s: s.rolling(7).mean())
    df["roll_std_7"] = g.shift(1).groupby(df["product_id"]).transform(lambda s: s.rolling(7).std())
    df["roll_mean_28"] = g.shift(1).groupby(df["product_id"]).transform(lambda s: s.rolling(28).mean())
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df

def engineer_marketing_funnel_features(mql_df: pd.DataFrame, closed_df: pd.DataFrame) -> tuple:
    """Join MQLs with closed_deals strictly for target creation, barring closed_deals predictors."""
    merged = mql_df.merge(closed_df[["mql_id", "won_date"]], on="mql_id", how="left")
    y = merged["won_date"].notnull().astype(int).values
    out = pd.DataFrame(index=merged.index)
    dt = pd.to_datetime(merged["first_contact_date"])
    out["contact_month"] = dt.dt.month.astype(float)
    out["contact_dow"] = dt.dt.dayofweek.astype(float)
    out["origin_clean"] = merged["origin"].fillna("unknown")
    return out, y
