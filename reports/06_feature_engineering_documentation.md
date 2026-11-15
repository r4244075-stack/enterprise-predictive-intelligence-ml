# Module 2 Deliverable: Feature Engineering & Selection Documentation

**Project**: Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02)  
**Module**: Module 2 — Advanced Feature Engineering & Baseline Modelling  
**Date**: November 14, 2026  

## 1. Leakage-Safe Preprocessing Architecture

All transformations are encapsulated inside train-only fit classes (`src/features/engineering.py`):
- **Numerical Scaling & Skewness Mitigation**: Heavily right-skewed positive features (`avg_order_value` skew `2.56`, `package_weight_g` skew `2.83`, `distance_km` skew `1.27`) are transformed via `np.log1p()` prior to Z-score standardization (`TrainOnlyStandardScaler`), where $\mu_{	ext{train}}$ and $\sigma_{	ext{train}}$ are computed strictly on training folds.
- **Missing Value Strategy**: Zero missing values exist in the 6 supplementary CSVs; for Olist Marketing Funnel (`origin` has `60` nulls) and Olist E-Commerce (`review_comment_message` / `product_weight_g`), categorical nulls are imputed as `"unknown"` and numerical nulls via training-fold median imputation.

## 2. Engineered Features, VIF Multicollinearity Audit & Feature Selection

### 2.1 Customer Retention (`customer_retention_features.csv`)
- **Engineered Features**: `log_avg_order_value`, `smoothed_return_rate` (`returns_last_180d / (orders_last_180d + 1)`), `tickets_per_order` (`support_tickets_last_180d / (orders_last_180d + 1)`), and anomaly indicator `flag_recency_zero_order`.
- **Multicollinearity (VIF) & Selection Decisions**: When all candidate ratios were included, `smoothed_return_rate` exhibited `VIF = 8.258` alongside `flag_return_gt_orders` (`VIF = 4.504`). Removing `flag_return_gt_orders`, `order_freq_per_tenure_mo`, `recency_to_tenure_ratio`, and the statistically non-significant `marketing_email_opt_in` ($\chi^2 p = 0.8692$) reduced all remaining feature VIFs to **$\le 3.527$**.

### 2.2 Delivery Delay (`delivery_delay_training.csv`)
- **Engineered Features**: `log_distance_km`, `log_package_weight_g`, `distance_per_promised_day`, and one-hot dispatch weather indicators (`weather_heat`, `weather_rain`, `weather_storm`, reference = `clear`).
- **Multicollinearity (VIF) & Selection Decisions**: Adding the interaction `load_x_log_distance` inflated VIFs (`69.784` on interaction, `60.161` on `carrier_load_index`, `11.727` on `log_distance_km`). Dropping `load_x_log_distance` plus non-informative `region_*` ($\chi^2 p = 0.5857$) and `is_weekend_order` reduced all remaining VIFs to **$\le 2.874$**.

### 2.3 Product Demand (`daily_product_demand.csv`)
- **Time-Dependent Features**: Strictly shifted autoregressive lags (`lag_1`, `lag_7`, `lag_14`, `lag_28`), shifted rolling statistics (`roll_mean_7`, `roll_std_7`, `roll_mean_28` computed on `units_sold.shift(1)`), and cyclical Fourier encodings (`dow_sin`, `dow_cos`, `month_sin`, `month_cos`).

### 2.4 Marketing Lead Conversion (`olist_marketing_qualified_leads_dataset.csv`)
- **Pre-Qualification Features**: `contact_month`, `contact_dow`, `origin_clean`, and out-of-fold target-encoded `landing_page_id`. All 13 `closed_deals` columns are strictly excluded.
