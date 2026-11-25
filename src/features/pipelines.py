"""Leak-free preprocessing, feature engineering, and out-of-fold target encoding pipelines.

Every stateful transformer (`fit` / `transform`) learns statistics strictly from the
training fold passed to `fit()` and never accesses validation or test labels/features.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class OutOfFoldTargetEncoder:
    """M-estimate smoothed target encoder with strict out-of-fold encoding on train.

    Guarantees:
    - During `fit_transform(X_train, y_train, folds)`, row i is encoded ONLY using
      statistics computed from training rows NOT in row i's fold.
    - During `transform(X_val_or_test)`, encoding uses statistics computed strictly
      from `X_train, y_train` (never touching validation or test labels).
    - Unseen categories fall back to the training global prior.
    """

    def __init__(self, cols: list[str], smoothing: float = 20.0, n_splits: int = 5, random_state: int = 42):
        self.cols = list(cols)
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.global_prior_: float = 0.0
        self.mapping_: dict[str, dict] = {}
        self.fitted_train_size_: int = 0

    def _compute_map(self, s: pd.Series, y: np.ndarray, prior: float) -> dict:
        df_tmp = pd.DataFrame({"k": s.fillna("__MISSING__").astype(str), "y": y})
        stats = df_tmp.groupby("k")["y"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + self.smoothing * prior) / (stats["count"] + self.smoothing)
        return smooth.to_dict()

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "OutOfFoldTargetEncoder":
        y = np.asarray(y, dtype=np.float64)
        self.global_prior_ = float(np.mean(y))
        self.fitted_train_size_ = len(y)
        self.mapping_ = {}
        for col in self.cols:
            self.mapping_[col] = self._compute_map(df[col], y, self.global_prior_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.cols:
            mp = self.mapping_[col]
            out[f"{col}_te"] = df[col].fillna("__MISSING__").astype(str).map(mp).fillna(self.global_prior_).astype(np.float64)
        return out

    def fit_transform_oof(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit full training mapping for test-time use, and return out-of-fold encoded train columns."""
        y = np.asarray(y, dtype=np.float64)
        self.fit(df, y)
        n = len(df)
        rng = np.random.default_rng(self.random_state)
        # Stratified fold assignment for OOF encoding
        fold_ids = np.zeros(n, dtype=int)
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            for k, idx in enumerate(cls_idx):
                fold_ids[idx] = k % self.n_splits

        out = pd.DataFrame(index=df.index)
        for col in self.cols:
            encoded = np.zeros(n, dtype=np.float64)
            s = df[col].fillna("__MISSING__").astype(str).reset_index(drop=True)
            for f in range(self.n_splits):
                tr_mask = fold_ids != f
                va_mask = fold_ids == f
                prior_f = float(np.mean(y[tr_mask]))
                mp_f = self._compute_map(s.iloc[tr_mask], y[tr_mask], prior_f)
                encoded[va_mask] = s.iloc[va_mask].map(mp_f).fillna(prior_f).values
            out[f"{col}_te"] = encoded
        return out


class TabularPreprocessor:
    """Leak-free tabular preprocessor: numeric median imputation + z-scaling,
    one-hot encoding for low-cardinality categoricals, frequency encoding &
    out-of-fold target encoding for high-cardinality categoricals.
    """

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str] | None = None,
        te_cols: list[str] | None = None,
        freq_cols: list[str] | None = None,
        te_smoothing: float = 20.0,
        random_state: int = 42,
    ):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols or [])
        self.te_cols = list(te_cols or [])
        self.freq_cols = list(freq_cols or [])
        self.te_smoothing = te_smoothing
        self.random_state = random_state

        self.medians_: dict[str, float] = {}
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.cat_levels_: dict[str, list[str]] = {}
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.te_: OutOfFoldTargetEncoder | None = None
        self.feature_names_: list[str] = []

    def _fit_transform_core(self, df: pd.DataFrame, y: np.ndarray | None, is_train: bool) -> np.ndarray:
        parts = []
        names = []

        # 1. Numeric columns
        num_mat = np.zeros((len(df), len(self.num_cols)), dtype=np.float64)
        for j, col in enumerate(self.num_cols):
            vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
            if is_train:
                med = float(np.nanmedian(vals)) if not np.all(np.isnan(vals)) else 0.0
                self.medians_[col] = med
            else:
                med = self.medians_[col]
            vals = np.where(np.isnan(vals), med, vals)
            num_mat[:, j] = vals
            names.append(col)

        # 2. Frequency-encoded columns (fit strictly on train)
        freq_cols_mat = []
        for col in self.freq_cols:
            s = df[col].fillna("__MISSING__").astype(str)
            if is_train:
                vc = s.value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = vc
            vals = s.map(self.freq_maps_[col]).fillna(0.0).values.astype(np.float64)
            freq_cols_mat.append(vals.reshape(-1, 1))
            names.append(f"{col}_freq")

        # 3. Target-encoded columns (strict OOF on train, train-only map on val/test)
        te_mat = []
        if self.te_cols:
            if is_train:
                assert y is not None, "y is required for training target encoding"
                self.te_ = OutOfFoldTargetEncoder(self.te_cols, smoothing=self.te_smoothing, random_state=self.random_state)
                te_df = self.te_.fit_transform_oof(df, y)
            else:
                te_df = self.te_.transform(df)
            for col in te_df.columns:
                te_mat.append(te_df[col].values.reshape(-1, 1))
                names.append(col)

        # Combine continuous columns and z-scale strictly from train
        cont_blocks = [num_mat] + freq_cols_mat + te_mat
        cont = np.hstack(cont_blocks) if cont_blocks else np.empty((len(df), 0))
        if is_train:
            self.means_ = cont.mean(axis=0)
            self.stds_ = cont.std(axis=0)
            self.stds_ = np.where(self.stds_ < 1e-8, 1.0, self.stds_)
        cont_scaled = (cont - self.means_) / self.stds_
        parts.append(cont_scaled)

        # 4. One-hot encoded categorical columns (drop first? keep all for trees + L2 handles collinearity cleanly)
        for col in self.cat_cols:
            s = df[col].fillna("__MISSING__").astype(str)
            if is_train:
                levels = sorted(s.unique().tolist())
                self.cat_levels_[col] = levels
            else:
                levels = self.cat_levels_[col]
            ohe = np.zeros((len(df), len(levels)), dtype=np.float64)
            for j, lv in enumerate(levels):
                ohe[:, j] = (s.values == lv).astype(np.float64)
                names.append(f"{col}={lv}")
            parts.append(ohe)

        if is_train:
            self.feature_names_ = names
        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        return self._fit_transform_core(df, y, is_train=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._fit_transform_core(df, None, is_train=False)


# ===================== DOMAIN FEATURE ENGINEERING (100% LEAK-FREE) =====================

def engineer_retention_features(df: pd.DataFrame) -> pd.DataFrame:
    """Snapshot-time behavioral ratios for customer retention (excludes customer_id, snapshot_date, target)."""
    out = df.copy()
    out["orders_per_tenure_month"] = out["orders_last_180d"] / np.maximum(out["tenure_months"], 1.0)
    out["return_rate_180d"] = out["returns_last_180d"] / np.maximum(out["orders_last_180d"], 1.0)
    out["tickets_per_order"] = out["support_tickets_last_180d"] / np.maximum(out["orders_last_180d"], 1.0)
    out["recency_to_tenure_ratio"] = out["days_since_last_order"] / np.maximum(out["tenure_months"] * 30.0, 1.0)
    out["sessions_per_recency"] = out["sessions_last_30d"] / (out["days_since_last_order"] + 7.0)
    out["estimated_180d_spend"] = out["orders_last_180d"] * out["avg_order_value"]
    out["log_avg_order_value"] = np.log1p(np.maximum(out["avg_order_value"], 0.0))
    return out


def engineer_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch-time logistics features ONLY. Strictly excludes actual_delivery_days, delay_days, late_delivery."""
    out = df.copy()
    out["distance_per_promised_day"] = out["distance_km"] / np.maximum(out["promised_delivery_days"], 1.0)
    out["weight_kg"] = out["package_weight_g"] / 1000.0
    out["log_weight_g"] = np.log1p(np.maximum(out["package_weight_g"], 0.0))
    out["load_x_distance"] = out["carrier_load_index"] * out["distance_km"]
    out["is_weekend_order"] = out["order_weekday"].isin([5, 6]).astype(int)
    out["is_adverse_weather"] = out["weather_at_dispatch"].isin(["storm", "rain", "heat"]).astype(int)
    out["is_storm"] = (out["weather_at_dispatch"] == "storm").astype(int)
    out["seller_risk_gap"] = 5.0 - out["seller_historical_rating"]
    return out


def engineer_marketing_features(mql_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly pre-conversion MQL features ONLY. Never touches any closed_deals column."""
    out = mql_df.copy()
    dt = pd.to_datetime(out["first_contact_date"])
    out["contact_month"] = dt.dt.month
    out["contact_dayofweek"] = dt.dt.dayofweek
    out["contact_day"] = dt.dt.day
    out["contact_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out["contact_days_since_start"] = (dt - dt.min()).dt.days
    out["origin_clean"] = out["origin"].fillna("missing")
    out["origin_is_missing"] = out["origin"].isna().astype(int)
    return out
