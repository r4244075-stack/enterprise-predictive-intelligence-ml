"""Comprehensive automated unit & leakage-invariant test suite (22 tests) for Project 02."""

from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipelines import (
    OutOfFoldTargetEncoder,
    TabularPreprocessor,
    engineer_retention_features,
    engineer_delivery_features,
    engineer_marketing_features,
)
from src.models.estimators import (
    LogisticRegressionL2,
    CalibratedLinearSVM,
    HistDecisionTreeClassifier,
    HistRandomForestClassifier,
    HistGradientBoostingClassifier,
    RidgeRegressor,
    HistGradientBoostingRegressor,
)
from src.evaluation.metrics import (
    roc_auc_score,
    pr_auc_score,
    classification_metrics,
    regression_metrics,
    optimize_threshold,
)
from src.evaluation.validation import stratified_kfold_indices, group_kfold_indices
from src.evaluation.clustering import kmeans_plus_plus, silhouette_score_sampled
from src.data.leakage_audit import (
    audit_1_sentiment_duplicate_leakage,
    audit_2_target_encoding_proof,
    audit_3_timeseries_lag_leakage_proof,
)


def _find_data_dir() -> Path:
    for c in [
        Path(os.environ.get("PROJECT02_DATA_DIR", "")),
        PROJECT_ROOT / "data" / "raw",
        Path("/working_dir/c_602e9190924555b9/user_uploaded_files/ttl=63d/user_uploaded"),
    ]:
        if c and c.is_dir() and (c / "customer_retention_features.csv").exists():
            return c
    raise FileNotFoundError("Dataset directory not found")


class TestDataAndLeakageInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = _find_data_dir()
        cls.ret_df = pd.read_csv(cls.data_dir / "customer_retention_features.csv")
        cls.del_df = pd.read_csv(cls.data_dir / "delivery_delay_training.csv")
        cls.mql_df = pd.read_csv(cls.data_dir / "olist_marketing_qualified_leads_dataset.csv")
        cls.cd_df = pd.read_csv(cls.data_dir / "olist_closed_deals_dataset.csv")
        cls.rev_df = pd.read_csv(cls.data_dir / "review_sentiment_training.csv")
        cls.dem_df = pd.read_csv(cls.data_dir / "daily_product_demand.csv")

    def test_01_retention_schema_and_no_nulls(self):
        self.assertEqual(len(self.ret_df), 30000)
        self.assertEqual(int(self.ret_df.isna().sum().sum()), 0)
        self.assertIn("no_purchase_next_90d", self.ret_df.columns)

    def test_02_delivery_schema_and_no_nulls(self):
        self.assertEqual(len(self.del_df), 40000)
        self.assertEqual(int(self.del_df.isna().sum().sum()), 0)
        self.assertTrue({"late_delivery", "delay_days", "actual_delivery_days"}.issubset(self.del_df.columns))

    def test_03_marketing_schema_and_join_integrity(self):
        self.assertEqual(len(self.mql_df), 8000)
        self.assertEqual(self.mql_df["mql_id"].nunique(), 8000)
        self.assertEqual(len(self.cd_df), 842)

    def test_04_retention_feature_engineering_no_nans(self):
        eng = engineer_retention_features(self.ret_df.head(500))
        for col in ["orders_per_tenure_month", "return_rate_180d", "tickets_per_order", "estimated_180d_spend"]:
            self.assertIn(col, eng.columns)
            self.assertFalse(eng[col].isna().any())

    def test_05_delivery_engineering_excludes_outcome_columns(self):
        # Drop outcome columns before engineering to verify pipeline works purely at dispatch time
        dispatch_only = self.del_df.head(500).drop(columns=["actual_delivery_days", "delay_days", "late_delivery"])
        eng = engineer_delivery_features(dispatch_only)
        self.assertNotIn("actual_delivery_days", eng.columns)
        self.assertNotIn("delay_days", eng.columns)
        self.assertIn("distance_per_promised_day", eng.columns)

    def test_06_marketing_engineering_pre_conversion_only(self):
        eng = engineer_marketing_features(self.mql_df.head(500))
        for forbidden in ["won_date", "business_segment", "lead_type", "declared_monthly_revenue"]:
            self.assertNotIn(forbidden, eng.columns)
        self.assertIn("origin_clean", eng.columns)

    def test_07_oof_target_encoder_zero_test_leakage(self):
        merged = self.mql_df.merge(self.cd_df[["mql_id", "won_date"]], on="mql_id", how="left")
        merged["converted"] = merged["won_date"].notna().astype(int)
        res = audit_2_target_encoding_proof(merged)
        self.assertEqual(res["test_label_invariance_max_abs_diff"], 0.0)
        self.assertTrue(res["unseen_category_falls_back_to_train_prior"])

    def test_08_oof_target_encoder_unseen_category_fallback(self):
        df_tr = pd.DataFrame({"cat": ["A", "A", "B", "B", "A", "B"]})
        y_tr = np.array([1, 1, 0, 0, 1, 0])
        te = OutOfFoldTargetEncoder(["cat"], smoothing=1.0, n_splits=2).fit(df_tr, y_tr)
        df_te = pd.DataFrame({"cat": ["UNSEEN_CAT"]})
        out = te.transform(df_te)["cat_te"].values[0]
        self.assertAlmostEqual(out, 0.5, places=6)

    def test_09_preprocessor_fits_stats_strictly_on_train(self):
        df_tr = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "c": ["a", "b", "a", "b"]})
        df_te = pd.DataFrame({"x": [100.0, np.nan], "c": ["a", "unseen"]})
        prep = TabularPreprocessor(num_cols=["x"], cat_cols=["c"])
        X_tr = prep.fit_transform(df_tr)
        X_te = prep.transform(df_te)
        # Train mean=2.5, std=1.118; test NaN must be imputed with train median (2.5 -> z=0.0)
        self.assertAlmostEqual(X_te[1, 0], 0.0, places=5)
        self.assertEqual(X_tr.shape[1], X_te.shape[1])

    def test_10_timeseries_lag_and_rolling_zero_future_leakage(self):
        res = audit_3_timeseries_lag_leakage_proof(self.dem_df)
        self.assertTrue(res["zero_future_leakage_verified"])
        self.assertEqual(res["past_features_max_abs_diff_under_future_shock"], 0.0)
        self.assertEqual(res["same_day_features_max_abs_diff_under_same_day_shock"], 0.0)

    def test_11_sentiment_duplicate_leakage_detected_and_grouped_zero_overlap(self):
        res = audit_1_sentiment_duplicate_leakage(self.rev_df)
        self.assertEqual(res["n_unique_review_texts"], 12)
        self.assertEqual(res["random_split_test_text_overlap_pct"], 100.0)
        self.assertEqual(res["grouped_4fold_overlap_pct"], 0.0)
        self.assertLess(res["grouped_4fold_macro_f1_mean"], 0.80)

    def test_12_stratified_kfold_preserves_class_balance(self):
        y = np.array([0] * 800 + [1] * 200)
        folds = stratified_kfold_indices(y, n_splits=5, random_state=42)
        self.assertEqual(len(folds), 5)
        for tr, va in folds:
            self.assertEqual(len(set(tr).intersection(set(va))), 0)
            self.assertAlmostEqual(np.mean(y[va]), 0.2, places=2)

    def test_13_group_kfold_enforces_disjoint_groups(self):
        groups = np.repeat(np.arange(12), 50)
        folds = group_kfold_indices(groups, n_splits=4)
        for tr, va in folds:
            self.assertEqual(len(set(groups[tr]).intersection(set(groups[va]))), 0)

    def test_14_logistic_regression_probabilities_and_determinism(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(400, 5))
        y = (X[:, 0] + 0.8 * X[:, 1] > 0).astype(int)
        m1 = LogisticRegressionL2(C=1.0, random_state=42).fit(X, y)
        m2 = LogisticRegressionL2(C=1.0, random_state=42).fit(X, y)
        p1 = m1.predict_proba(X)[:, 1]
        p2 = m2.predict_proba(X)[:, 1]
        self.assertTrue(np.allclose(p1, p2))
        self.assertGreater(roc_auc_score(y, p1), 0.90)

    def test_15_calibrated_linear_svm_bounds(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] - X[:, 1] > 0).astype(int)
        svm = CalibratedLinearSVM(C=1.0).fit(X, y)
        p = svm.predict_proba(X)[:, 1]
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        self.assertGreater(roc_auc_score(y, p), 0.90)

    def test_16_hist_decision_tree_classifier_learns_nonlinear(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(600, 3))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        dt = HistDecisionTreeClassifier(max_depth=4, min_samples_leaf=15).fit(X, y)
        self.assertGreater(roc_auc_score(y, dt.predict_proba(X)[:, 1]), 0.85)

    def test_17_hist_random_forest_classifier_reproducible(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 4))
        y = (X[:, 0] + X[:, 2] > 0.2).astype(int)
        rf1 = HistRandomForestClassifier(n_estimators=15, max_depth=5, random_state=42).fit(X, y)
        rf2 = HistRandomForestClassifier(n_estimators=15, max_depth=5, random_state=42).fit(X, y)
        self.assertTrue(np.allclose(rf1.predict_proba(X), rf2.predict_proba(X)))

    def test_18_hist_gbdt_classifier_converges(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(600, 4))
        y = (X[:, 0] ** 2 + X[:, 1] > 1.0).astype(int)
        gb = HistGradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=42).fit(X, y)
        self.assertGreater(roc_auc_score(y, gb.predict_proba(X)[:, 1]), 0.90)

    def test_19_regression_estimators_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 4))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * rng.normal(size=500)
        r1 = RidgeRegressor(alpha=1.0).fit(X, y)
        r2 = HistGradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X, y)
        self.assertGreater(regression_metrics(y, r1.predict(X))["r2"], 0.95)
        self.assertGreater(regression_metrics(y, r2.predict(X))["r2"], 0.90)

    def test_20_threshold_optimizer_improves_or_matches_default(self):
        rng = np.random.default_rng(42)
        y = (rng.uniform(size=1000) < 0.15).astype(int)
        p = np.clip(0.1 + 0.35 * y + rng.normal(0, 0.12, size=1000), 0.01, 0.99)
        t_opt, f1_opt = optimize_threshold(y, p, metric="f1")
        f1_05 = classification_metrics(y, p, threshold=0.5)["f1"]
        self.assertGreaterEqual(f1_opt, f1_05)
        self.assertTrue(0.05 <= t_opt <= 0.90)

    def test_21_kmeans_recovers_separated_clusters(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(-5, 0.5, (150, 2)), rng.normal(5, 0.5, (150, 2))])
        lbl, _, _ = kmeans_plus_plus(X, k=2, n_init=5, random_state=42)
        sil = silhouette_score_sampled(X, lbl)
        self.assertGreater(sil, 0.85)

    def test_22_saved_metrics_artifact_exists_and_complete(self):
        p = PROJECT_ROOT / "reports" / "metrics" / "all_results.json"
        self.assertTrue(p.exists(), "Run run_all.py before running full artifact check")
        import json
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key in [
            "part_a_retention",
            "part_a_delivery_classification",
            "part_a_marketing_conversion",
            "part_b_delivery_regression",
            "part_c_customer_segmentation",
            "module_2_leakage_audits",
        ]:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
