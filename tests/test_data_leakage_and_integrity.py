import unittest
import pandas as pd
from src.data.audit_pipeline import verify_delivery_leakage_free

class TestModule1LeakageGuards(unittest.TestCase):
    def test_delivery_leakage_guard_blocks_outcomes(self):
        df = pd.DataFrame()
        with self.assertRaises(ValueError):
            verify_delivery_leakage_free(df, ["distance_km", "actual_delivery_days"])
        with self.assertRaises(ValueError):
            verify_delivery_leakage_free(df, ["distance_km", "delay_days"])

    def test_delivery_leakage_guard_allows_dispatch_features(self):
        df = pd.DataFrame()
        valid = [
            "distance_km", "package_weight_g", "seller_historical_rating",
            "carrier_load_index", "order_weekday", "weather_at_dispatch",
            "region", "promised_delivery_days"
        ]
        self.assertTrue(verify_delivery_leakage_free(df, valid))

if __name__ == "__main__":
    unittest.main()
