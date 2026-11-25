# Module 1 Deliverable: Feature & Target Identification and Leakage Prevention Matrix

**Project**: Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02)  
**Module**: Module 1 — Data Science Foundations & Problem Framing  
**Date**: November 12, 2026  

## 1. Target Variable & ML Paradigm Specification Across All 6 Investigation Areas

| Investigation Area | Primary Dataset(s) | ML Problem Type | Target Variable(s) | Granularity / Entity Key | Positive Class / Target Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Customer Retention Prediction** | [`customer_retention_features.csv`](https://drive.google.com/file/d/1YTgNMMZlfivpcNLbj3vGpVNtuR0c8gIz/view?usp=drivesdk) & [Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Supervised Binary Classification | `no_purchase_next_90d` (`1` = lapsed 90d, `0` = retained) | `customer_id` / `customer_unique_id` at `snapshot_date` | `34.79%` positive (`10,438` lapsed vs. `19,562` active) |
| **2. Delivery Delay Prediction** | [`delivery_delay_training.csv`](https://drive.google.com/file/d/1D4SLq3uE64kadv29oSU0x6ht8bn73NG3/view?usp=drivesdk) & [Olist Orders](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Binary Classification & Continuous Regression | Classification: `late_delivery` (`0`/`1`)<br>Regression: `delay_days` (`>= 0`) | `delivery_id` / `order_id` at dispatch timestamp | `59.29%` late (`23,717` late); `delay_days` mean `1.825d` (`std = 1.701d`) |
| **3. Product Demand Forecasting** | [`daily_product_demand.csv`](https://drive.google.com/file/d/1UagJBHolBS5olzdLv83Ge2dCeSn8gNMR/view?usp=drivesdk) | Panel Time-Series Forecasting | `units_sold` (daily integer volume per SKU) | `(product_id, date)` across 150 SKUs & 730 days | Mean `18.65` units/day (`median = 16.0`, `min = 0`, `max = 109`) |
| **4. Review Sentiment Classification** | [`review_sentiment_training.csv`](https://drive.google.com/file/d/1Lv6G6pPHgA6tSiL_vdvD90uPljw-P_IP/view?usp=drivesdk) & [`review_sentiment_unlabeled.csv`](https://drive.google.com/file/d/1zGsB2h52_GQJLD5o2nHUJ3yB1mAJDjlz/view?usp=drivesdk) | Multi-Class NLP Text Classification | `sentiment_label` (`positive`, `neutral`, `negative`) | `review_id` | Train: `57.78%` pos (`11,555`), `25.04%` neg (`5,008`), `17.18%` neu (`3,437`) |
| **5. Marketing Lead Conversion** | [Olist Marketing Funnel Dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) (`mql` + `closed_deals`) | Supervised Binary Classification | `converted` (`1` if `won_date` is non-null, `0` otherwise) | `mql_id` at `first_contact_date` | `10.53%` converted (`842` closed deals out of `8,000` MQLs) |
| **6. Customer Behavioural Segmentation** | [`customer_retention_features.csv`](https://drive.google.com/file/d/1YTgNMMZlfivpcNLbj3vGpVNtuR0c8gIz/view?usp=drivesdk) & [Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Unsupervised Clustering | None (unsupervised cluster assignment $k \in \{3, 4\}$) | `customer_id` / `customer_unique_id` | 4 distinct behavioural personas (`k=4` Silhouette = `0.1342`, `k=3` = `0.1519`) |

## 2. Mathematical Proof of Target, Temporal, and Entity Leakage

### 2.1 Deterministic Target Leakage in `delivery_delay_training.csv`
Empirical inspection of all `40,000` rows in [`delivery_delay_training.csv`](https://drive.google.com/file/d/1D4SLq3uE64kadv29oSU0x6ht8bn73NG3/view?usp=drivesdk) proves two exact algebraic identities that hold with **zero exceptions (`40,000 / 40,000` rows)**:
1. `delay_days = max(0.0, actual_delivery_days - promised_delivery_days)`
2. `late_delivery = 1` if and only if `delay_days > 1.0` (`min(delay_days)` when `late_delivery == 1` is `1.1` days; `max(delay_days)` when `late_delivery == 0` is `1.0` day, revealing a 1.0-day operational grace window across `6,867` rows with `0 < delay_days <= 1.0`).

**Mandatory Leakage Guardrail**: Both `actual_delivery_days` ($r = 0.4284$ with `late_delivery`, $r = 0.5199$ with `delay_days`) and `delay_days` ($r = 0.7721$ with `late_delivery`) are post-delivery outcomes unavailable at order dispatch. They must be **strictly excluded** from all feature matrices `X`. Only the 8 dispatch-time predictors (`distance_km`, `package_weight_g`, `seller_historical_rating`, `carrier_load_index`, `order_weekday`, `weather_at_dispatch`, `region`, and `promised_delivery_days`) are permissible inputs.

### 2.2 Post-Conversion Target Leakage in Olist Marketing Funnel
When left-joining `olist_marketing_qualified_leads_dataset.csv` (`8,000` rows: `mql_id`, `first_contact_date`, `landing_page_id`, `origin`) with `olist_closed_deals_dataset.csv` (`842` rows) from the [Marketing Funnel by Olist Dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist), all 13 columns from `closed_deals` (`seller_id`, `sdr_id`, `sr_id`, `won_date`, `business_segment`, `lead_type`, `lead_behaviour_profile`, `has_company`, `has_gtin`, `average_stock`, `business_type`, `declared_product_catalog_size`, `declared_monthly_revenue`) are populated **only if the lead converted (`converted == 1`)** and are `NaN` for all `7,158` non-converted leads. Including any `closed_deals` attribute as a predictor leaks the target with `100%` precision. Only pre-qualification features (`first_contact_date` temporal features, `landing_page_id` historical conversion encoding with out-of-fold smoothing, and `origin` channel) may be used to predict lead conversion; `closed_deals` attributes may only be used for post-conversion seller LTV modeling.

### 2.3 Entity & Temporal Leakage in Customer Retention, Demand Forecasting, and NLP
- **Customer Entity Confusion (Olist)**: In `olist_customers_dataset.csv`, `customer_id` is generated per order (`99,441` unique values), whereas `customer_unique_id` identifies the actual human buyer (`96,096` unique buyers). Aggregating retention or RFM features on `customer_id` fails to link repeat purchases.
- **Look-Ahead Lag Leakage (`daily_product_demand.csv`)**: Rolling demand features (7-day/14-day/28-day moving averages and standard deviations) must be computed on `units_sold.shift(horizon)` grouped by `product_id`. Computing unshifted rolling means leaks future sales into the training matrix.
- **Supervised Target Leakage in Clustering (`customer_retention_features.csv`)**: `no_purchase_next_90d` is a future 90-day outcome (`2024-01-01` to `2024-03-31`). It must be excluded when fitting K-Means/GMM customer segments on `2024-01-01` snapshot features, and used solely as an external post-hoc validation metric to profile segment churn risk.
