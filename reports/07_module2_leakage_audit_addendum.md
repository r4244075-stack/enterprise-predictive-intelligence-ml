# Module 2 Evaluator Remarks — Reproducible Leakage Audit Addendum

This addendum directly answers all four technical validation questions raised in the Module 2 evaluation using reproducible experiments (`src/data/leakage_audit.py`) and automated unit tests (`tests/test_leakage_and_pipelines.py`, 22/22 passing).

## Question 1: Why did the sentiment model achieve a perfect Macro-F1 of 1.0?
- **Root Cause (Duplicate-Text Leakage):** `review_sentiment_training.csv` contains **20,000 rows** built from only **12 unique template sentences** (99.94% exact duplicates, zero label ambiguity).
- Under a standard stratified random 80/20 split, **100.0%** of test rows are verbatim copies of training sentences, yielding an artificial **Macro-F1 = 1.0000** via memorization.
- **After Grouping Duplicate Review Sentences (`GroupKFold`, 4 folds, 0.0% text overlap):** Macro-F1 drops sharply to **0.2906 ± 0.1804** (fold scores: [0.3333, 0.3333, 0.0, 0.4956]), confirming the `DATASET_README.md` warning that the synthetic template corpus does not represent natural-language generalization.

## Question 2: Complete Target-Encoding Implementation & Proof of Zero Val/Test Leakage
- Implemented in `src/features/pipelines.py` (`OutOfFoldTargetEncoder`):
  - **Training rows** use internal 5-fold stratified out-of-fold M-estimate smoothing (`smoothing=20.0`), so row *i*'s label never contributes to row *i*'s encoded value.
  - **Validation/Test rows** are transformed solely via `transform(df_eval)` using the frozen training mapping; unseen categories fall back deterministically to the training global prior (**0.1053**).
- **Programmatic Proof (`test_07_oof_target_encoder_zero_test_leakage`):** Across 6,400 train rows and 1,600 test rows (40 unseen `landing_page_id`s in test), `transform()` never receives `y_test` (label-invariance max abs diff = **0.0**), and all 40 unseen landing pages fall back exactly to the training prior (`True`).

## Question 3: Proof That No Future Demand Values Enter Lag & Rolling Features
- On `daily_product_demand.csv` (109,500 rows, 150 products × 730 days), every lag (`lag_1`, `lag_7`, `lag_14`) and rolling statistic (`roll_mean_7`, `roll_std_7`, `roll_mean_28`) is computed strictly per `product_id` after sorting by `date` and applying `.shift(1)` **before** `.rolling()`.
- **Future-Shock Perturbation Proof (`test_10_timeseries_lag_and_rolling_zero_future_leakage`):** Adding `+10,000` units to all `units_sold` on `date >= 2024-09-01` produces a maximum absolute difference of **0.0** on all features for `date < 2024-09-01`, and adding `+10,000` on `date == 2024-09-01` produces **0.0** difference on that day's features (`zero_future_leakage_verified = True`).

## Question 4: Marketing Conversion Leakage Audit — Does ROC-AUC Remain Above 0.95?
- **No — ROC-AUC does NOT remain above 0.95 once leakage is removed (`remains_above_0_95 = False`).**
- Controlled ablation on the real Olist Marketing Funnel (`8,000` MQLs, `842` closed deals):
  - Including a post-conversion `closed_deals` indicator (`business_segment` presence): **ROC-AUC = 1.0000** (target leakage).
  - Full-dataset target encoding of `landing_page_id` before splitting: **ROC-AUC = 0.7567** (encoding leakage).
  - **Strict Leak-Free OOF Pipeline (Stratified 80/20):** Logistic Regression **ROC-AUC = 0.7058**, HistGBDT Default **ROC-AUC = 0.7129** (Tuned **0.7155**, HistRF **0.7188**).
  - **Strict Leak-Free OOF Pipeline (Chronological 80/20):** HistGBDT **ROC-AUC = 0.6871**.
- **Conclusion:** Any ROC-AUC > 0.95 on Olist MQL conversion is an artifact of post-conversion or full-dataset encoding leakage. The genuine, reproducible pre-conversion predictive ceiling with the 3 available MQL attributes is **~0.71–0.72 ROC-AUC**.
