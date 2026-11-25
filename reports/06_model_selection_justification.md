# Module 3 — Model Selection Justification & Production Trade-Offs

| Problem | Selected Model | 5-Fold CV Metric | Untouched Test Metric | Key Justification |
| --- | --- | --- | --- | --- |
| Customer Retention | Logistic Regression | ROC-AUC 0.7359 ± 0.0062 | ROC-AUC 0.7418, PR-AUC 0.5983, F1@0.28 = 0.5999 | Highest CV ROC-AUC, zero overfitting (train-test gap -0.0056), well-calibrated probabilities, exact LinearSHAP explainability. |
| Delivery Delay Classification | Logistic Regression | ROC-AUC 0.8494 ± 0.0045 | ROC-AUC 0.8532, PR-AUC 0.8941, F1@0.42 = 0.8197 | Highest CV ROC-AUC (marginally ahead of Tuned HistGBDT 0.8481), sub-millisecond dispatch-time inference, transparent risk weights. |
| Marketing Lead Conversion | Logistic Regression (tie w/ Tuned HistGBDT) | ROC-AUC 0.7232 ± 0.0113 | LogReg AUC 0.7050; Tuned HistGBDT AUC 0.7155, F1@0.15 = 0.2983 | Both leak-free models achieve ~0.723 CV AUC; Tuned HistGBDT captures nonlinear origin × landing-page interactions for higher Test F1 (0.2983 vs 0.2777). |
| Delivery Delay Regression | Hist GBDT (Default) | RMSE 1.1703 ± 0.0114 | RMSE 1.1578d, MAE 0.916d, R² 0.5404 | Captures nonlinear storm/carrier-load/distance interactions better than Ridge (RMSE 1.1612d) with unbiased residuals (-0.0013d mean). |
| Customer Segmentation | K-Means++ (k=4) | CH 3,786.5 (full 30k) | 4 balanced personas (22%–31% each) | Cleanly separates New/Low-Order High-Risk (49.9% non-purchase) from Tenured Loyalists (23.4%–28.1%) and High-Return Active Buyers (33.0%); DBSCAN collapsed into 1 cluster in 8D. |

## Data Limitations Disclosed (Per `DATASET_README.md`)
1. `customer_retention_features.csv`, `delivery_delay_training.csv`, `daily_product_demand.csv`, and `review_sentiment_training.csv` are synthetic datasets; `no_purchase_next_90d` is a 90-day operational proxy rather than permanent churn.
2. `olist_marketing_qualified_leads_dataset.csv` and `olist_closed_deals_dataset.csv` are real Olist funnel records with only 3 pre-conversion attributes (`first_contact_date`, `landing_page_id`, `origin`), bounding realistic leak-free ROC-AUC around 0.71–0.72.
