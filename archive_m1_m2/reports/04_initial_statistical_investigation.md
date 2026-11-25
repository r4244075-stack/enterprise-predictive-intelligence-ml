# Module 1 Deliverable: Initial Statistical Investigation & Hypothesis Testing Report

**Project**: Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02)  
**Module**: Module 1 — Data Science Foundations & Problem Framing  
**Date**: November 12, 2026  

## 1. Customer Retention Statistical Tests (`customer_retention_features.csv`, $N = 30,000$)

Two-sided Mann-Whitney U tests (non-parametric rank-sum tests resilient to skewness) and Pearson ($r$) / Spearman ($ho$) correlations were conducted against `no_purchase_next_90d` (`0` = Retained [$n=19,562$], `1` = Lapsed [$n=10,438$]):

| Feature Name | Retained Mean (Med) | Lapsed Mean (Med) | Pearson $r$ | Spearman $ho$ | Mann-Whitney $U$ | $p$-Value | Statistical Conclusion |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `orders_last_180d` | 4.40 (4.0) | 2.81 (2.0) | -0.2601 | -0.2660 | 69,372,953 | `< 1e-100` | **Primary protective driver** (higher frequency lowers lapse risk) |
| `tenure_months` | 33.82 (35.0) | 24.48 (22.0) | -0.2568 | -0.2568 | 70,322,294 | `< 1e-100` | **Primary protective driver** (-9.34 months shorter tenure in lapsed) |
| `days_since_last_order` | 80.54 (77.0) | 107.63 (114.0) | +0.2470 | +0.2470 | 132,662,284 | `< 1e-100` | **Primary risk driver** (+27.09 days recency gap in lapsed) |
| `support_tickets_last_180d` | 1.13 (1.0) | 1.33 (1.0) | +0.0859 | +0.0818 | 111,773,836 | `1.57e-45` | **Significant friction driver** (service dissatisfaction increases lapse) |
| `discount_order_share` | 0.317 (0.314) | 0.342 (0.351) | +0.0624 | +0.0623 | 109,808,070 | `3.57e-27` | **Significant promo-sensitivity effect** (deal-chasers lapse more) |
| `returns_last_180d` | 0.328 (0.0) | 0.386 (0.0) | +0.0470 | +0.0427 | 106,300,146 | `1.45e-13` | **Significant return-friction effect** |
| `sessions_last_30d` | 5.07 (5.0) | 4.90 (5.0) | -0.0364 | -0.0361 | 97,667,311 | `3.99e-10` | **Moderate engagement signal** |
| `avg_order_value` | 82.10 (66.36) | 83.17 (67.04) | +0.0085 | +0.0045 | 102,648,293 | `0.4379` | **Statistically non-significant** (uninformative standalone predictor) |
| `marketing_email_opt_in` | 49.7% | 49.6% | -0.0010 | -0.0010 | $\chi^2 = 0.0271$ | `0.8692` | **Pure noise** ($\chi^2$ independence test confirms zero linear association) |

## 2. Delivery Delay Statistical Investigation (`delivery_delay_training.csv`, $N = 40,000$)

1. **Continuous Dispatch Predictors**: `carrier_load_index` is the dominant dispatch-time predictor of both `delay_days` ($r = +0.5401$, $p < 10^{-100}$) and `late_delivery` ($r = +0.4852$, $p < 10^{-100}$). `distance_km` is the second strongest continuous predictor ($r = +0.1917$ with `delay_days`, $r = +0.1482$ with `late_delivery`), followed by `package_weight_g` ($r = +0.0275$ with `delay_days`). By contrast, `order_weekday` ($r = 0.0000$), `promised_delivery_days` ($r = -0.0018$), and `seller_historical_rating` ($r = +0.0077$) show negligible marginal linear correlation.
2. **Categorical Contingency Tests ($\chi^2$)**:
   - **`weather_at_dispatch` ($\chi^2 = 3,197.90, 	ext{df} = 3, p < 10^{-100}$)**: Highly significant determinant of late delivery. Late rates escalate monotonically across weather severity: `clear` (`50.14%` late, mean delay `1.39d`, $n=24,898$), `heat` (`57.29%` late, `1.65d`, $n=3,189$), `rain` (`73.44%` late, `2.33d`, $n=8,776$), and `storm` (`94.39%` late, `4.02d`, $n=3,137$).
   - **`region` ($\chi^2 = 2.8354, 	ext{df} = 4, p = 0.5857$)**: Fails to reject the null hypothesis of independence; late rates are invariant across `central` (`59.79%`), `west` (`59.60%`), `north` (`59.42%`), `south` (`59.00%`), and `east` (`58.65%`).

## 3. Product Demand Seasonality & Promotional Lift (`daily_product_demand.csv`, $N = 109,500$)

- **Promotional Lift Hypothesis Test**: On promotional days (`promotion_flag == 1`, $n = 4,500$, `4.11%` of observations), mean daily demand is **29.42 units/SKU** compared to **18.19 units/SKU** on non-promotional days ($n = 105,000$)—a **+61.72% promotional demand lift** (Mann-Whitney $U = 359,842,110$, $p < 10^{-100}$).
- **Day-of-Week Seasonality**: Kruskal-Wallis test across `day_of_week` ($0=	ext{Mon}$ to $6=	ext{Sun}$) confirms strong weekly cyclicity ($H = 4,061.89, p < 10^{-100}$), peaking mid-week on Wednesday (`22.32` units) and Tuesday (`21.55` units) and troughing on Saturday (`15.01` units) and Sunday (`15.67` units).
- **Monthly Seasonality**: Kruskal-Wallis across `month` ($1	ext{–}12$) shows mild but statistically significant variation ($H = 21.75, p = 0.0264$), with January (`19.02` units) and February (`18.96` units) slightly above August (`18.40` units).

## 4. Customer Behavioural Segmentation Profile (`customer_retention_features.csv`, K-Means++ $k = 4$)

Standardizing the 8 behavioural features (`avg_order_value` log-transformed via $\ln(1+x)$) and profiling $k \in \{2,3,4,5,6\}$ yields optimal interpretability at $k = 4$ (`Silhouette = 0.1342`, `Inertia = 177,947.9`):

| Segment ID & Persona Name | Customer Count (%) | Mean Tenure (mo) | Mean Orders (180d) | Mean Recency (d) | Mean Returns (180d) | Mean Tickets (180d) | Post-Hoc 90d Lapse Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Seg 0: Low-Tenure At-Risk Buyers** | 11,562 (38.5%) | 15.77 | 1.60 | 90.49 | 0.26 | 0.74 | **44.1%** (Highest Risk) |
| **Seg 1: High-Return Friction Buyers** | 1,414 (4.7%) | 30.88 | 3.89 | 90.96 | **2.13** | 1.20 | **43.6%** (Return Churn) |
| **Seg 2: High-Support Escalation Cohort** | 5,901 (19.7%) | 29.48 | 3.42 | 90.56 | 0.26 | **2.75** | **41.4%** (Service Churn) |
| **Seg 3: High-Tenure Loyal Champions** | 11,123 (37.1%) | **46.50** | **6.41** | 88.99 | 0.26 | 0.85 | **20.4%** (Retained Core) |
