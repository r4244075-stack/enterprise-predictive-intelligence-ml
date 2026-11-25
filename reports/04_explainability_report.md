# Module 3 — Model Interpretability & Explainability Report (SHAP, PFI & PDP)

## Customer Retention (Logistic Regression)

### Top 8 Global SHAP Feature Attributions
| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | days_since_last_order | 0.53898 |
| 2 | orders_last_180d | 0.34853 |
| 3 | tenure_months | 0.25923 |
| 4 | support_tickets_last_180d | 0.17398 |
| 5 | returns_last_180d | 0.12971 |
| 6 | discount_order_share | 0.12483 |
| 7 | sessions_last_30d | 0.06796 |
| 8 | avg_order_value | 0.04097 |

### Top 8 Permutation Feature Importance (Test Set)
| Rank | Feature | Metric Drop (Mean ± Std) |
| --- | --- | --- |
| 1 | days_since_last_order | 0.09974 ± 0.00561 |
| 2 | orders_last_180d | 0.05605 ± 0.00321 |
| 3 | tenure_months | 0.02902 ± 0.00286 |
| 4 | support_tickets_last_180d | 0.01048 ± 0.00192 |
| 5 | returns_last_180d | 0.00667 ± 0.00082 |
| 6 | discount_order_share | 0.00499 ± 0.00106 |
| 7 | sessions_last_30d | 0.00281 ± 0.00062 |
| 8 | avg_order_value | 0.00137 ± 0.00055 |

## Delivery Delay Classification (Logistic Regression)

### Top 8 Global SHAP Feature Attributions
| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | carrier_load_index | 1.23074 |
| 2 | is_storm | 0.45127 |
| 3 | is_adverse_weather | 0.37562 |
| 4 | distance_km | 0.35691 |
| 5 | weather_at_dispatch=rain | 0.19639 |
| 6 | weather_at_dispatch=heat | 0.06552 |
| 7 | region=west | 0.03680 |
| 8 | weather_at_dispatch=storm | 0.03522 |

### Top 8 Permutation Feature Importance (Test Set)
| Rank | Feature | Metric Drop (Mean ± Std) |
| --- | --- | --- |
| 1 | carrier_load_index | 0.24198 ± 0.00300 |
| 2 | is_storm | 0.03867 ± 0.00185 |
| 3 | distance_km | 0.02715 ± 0.00186 |
| 4 | is_adverse_weather | 0.01794 ± 0.00081 |
| 5 | weather_at_dispatch=rain | 0.00711 ± 0.00086 |
| 6 | weather_at_dispatch=heat | 0.00223 ± 0.00036 |
| 7 | weather_at_dispatch=clear | 0.00026 ± 0.00012 |
| 8 | distance_per_promised_day | 0.00016 ± 0.00006 |

## Marketing Lead Conversion (Logistic Regression)

### Top 8 Global SHAP Feature Attributions
| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | landing_page_id_te | 0.38315 |
| 2 | contact_month | 0.29761 |
| 3 | contact_days_since_start | 0.26040 |
| 4 | origin_clean=organic_search | 0.14639 |
| 5 | origin_clean=unknown | 0.11269 |
| 6 | contact_day | 0.10579 |
| 7 | origin_clean=paid_search | 0.09396 |
| 8 | contact_dayofweek | 0.06755 |

### Top 8 Permutation Feature Importance (Test Set)
| Rank | Feature | Metric Drop (Mean ± Std) |
| --- | --- | --- |
| 1 | landing_page_id_te | 0.06280 ± 0.00889 |
| 2 | contact_month | 0.03567 ± 0.01085 |
| 3 | contact_days_since_start | 0.01517 ± 0.00838 |
| 4 | origin_clean=unknown | 0.01209 ± 0.00395 |
| 5 | origin_clean=paid_search | 0.00672 ± 0.00203 |
| 6 | landing_page_id_freq | 0.00461 ± 0.00222 |
| 7 | origin_clean=email | 0.00281 ± 0.00396 |
| 8 | contact_is_weekend | 0.00069 ± 0.00120 |

## Delivery Delay Regression (Hist GBDT (Default))

### Top 8 Global SHAP Feature Attributions
| Rank | Feature | Mean |SHAP| |
| --- | --- | --- |
| 1 | carrier_load_index | 0.68662 |
| 2 | load_x_distance | 0.26297 |
| 3 | weather_at_dispatch=rain | 0.18174 |
| 4 | is_storm | 0.15542 |
| 5 | weather_at_dispatch=clear | 0.11417 |
| 6 | weather_at_dispatch=storm | 0.11407 |
| 7 | is_adverse_weather | 0.06938 |
| 8 | package_weight_g | 0.01648 |

### Top 8 Permutation Feature Importance (Test Set)
| Rank | Feature | Metric Drop (Mean ± Std) |
| --- | --- | --- |
| 1 | carrier_load_index | 0.45268 ± 0.00680 |
| 2 | load_x_distance | 0.10046 ± 0.00255 |
| 3 | is_storm | 0.09994 ± 0.00440 |
| 4 | weather_at_dispatch=storm | 0.05433 ± 0.00131 |
| 5 | weather_at_dispatch=rain | 0.04251 ± 0.00226 |
| 6 | weather_at_dispatch=clear | 0.01486 ± 0.00110 |
| 7 | is_adverse_weather | 0.00584 ± 0.00169 |
| 8 | distance_km | 0.00234 ± 0.00024 |

## Local Individual SHAP Explanations (Customer Retention — Logistic Regression)

- **True Positive (High Risk)** (`customer_id=DS_CUST_007498`, Actual=1, Pred Prob=0.8738, Tenure=2m, Orders=0, Recency=150d): Top log-odds SHAP drivers → `support_tickets_last_180d` (+0.748), `days_since_last_order` (+0.720), `orders_last_180d` (+0.560), `tenure_months` (+0.497).
- **False Positive (Over-Predicted Risk)** (`customer_id=DS_CUST_015532`, Actual=0, Pred Prob=0.8215, Tenure=2m, Orders=0, Recency=166d): Top log-odds SHAP drivers → `days_since_last_order` (+0.908), `orders_last_180d` (+0.560), `support_tickets_last_180d` (+0.553), `tenure_months` (+0.497).
- **False Negative (Missed Churn)** (`customer_id=DS_CUST_010028`, Actual=1, Pred Prob=0.0226, Tenure=58m, Orders=11, Recency=20d): Top log-odds SHAP drivers → `orders_last_180d` (-1.077), `days_since_last_order` (-0.809), `tenure_months` (-0.486), `discount_order_share` (-0.248).
