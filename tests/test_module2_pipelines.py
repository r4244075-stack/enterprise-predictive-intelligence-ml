import unittest
import numpy as np
import pandas as pd
from src.features.engineering import (
    TrainOnlyStandardScaler,
    engineer_delivery_features,
    engineer_demand_features,
    engineer_marketing_funnel_features,
)

class TestModule2Pipelines(unittest.TestCase):
    def test_scaler_requires_fit(self):
        scaler = TrainOnlyStandardScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_delivery_engineering_drops_leaks(self):
        df = pd.DataFrame({
            "distance_km": [100.0],
            "package_weight_g": [500.0],
            "seller_historical_rating": [4.5],
            "carrier_load_index": [0.8],
            "promised_delivery_days": [5],
            "weather_at_dispatch": ["clear"],
            "actual_delivery_days": [7.0],
            "delay_days": [2.0],
        })
        X = engineer_delivery_features(df)
        self.assertNotIn("actual_delivery_days", X.columns)
        self.assertNotIn("delay_days", X.columns)

    def test_demand_lag_has_no_lookahead(self):
        df = pd.DataFrame({
            "product_id": ["P1", "P1", "P1"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "day_of_week": [0, 1, 2],
            "month": [1, 1, 1],
            "units_sold": [10, 20, 30],
        })
        out = engineer_demand_features(df)
        self.assertTrue(pd.isna(out.loc[0, "lag_1"]))
        self.assertEqual(out.loc[1, "lag_1"], 10)
        self.assertEqual(out.loc[2, "lag_1"], 20)

    def test_marketing_funnel_excludes_closed_deals(self):
        mql = pd.DataFrame({
            "mql_id": ["M1", "M2"],
            "first_contact_date": ["2018-01-01", "2018-01-02"],
            "origin": ["organic_search", None],
        })
        closed = pd.DataFrame({
            "mql_id": ["M1"],
            "won_date": ["2018-01-10"],
            "declared_monthly_revenue": [50000.0],
        })
        X, y = engineer_marketing_funnel_features(mql, closed)
        self.assertListEqual(list(y), [1, 0])
        self.assertNotIn("declared_monthly_revenue", X.columns)
        self.assertNotIn("won_date", X.columns)

if __name__ == "__main__":
    unittest.main()
