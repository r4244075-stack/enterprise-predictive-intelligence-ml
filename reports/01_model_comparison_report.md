# Module 3 — Model Comparison Report

All models were trained and evaluated strictly on the real provided datasets using 5-fold stratified cross-validation (classification) or 5-fold CV (regression), with preprocessing, scaling, frequency encoding, and out-of-fold target encoding fit exclusively inside each training fold. Test sets (20% untouched holdout) were never used for model selection or threshold tuning.

## Part A.1 — Customer Retention (`no_purchase_next_90d`, n=30,000, 34.79% positive)

Selected by 5-Fold CV ROC-AUC: **Logistic Regression**

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Opt Thr (OOF) | Test F1@Opt | Precision | Recall | Train-Test Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7359 ± 0.0062 | 0.7418 | 0.5983 | 0.28 | 0.5999 | 0.4808 | 0.7974 | -0.0051 |
| Calibrated Linear SVM | 0.7356 ± 0.0064 | 0.7411 | 0.5958 | 0.28 | 0.5965 | 0.4788 | 0.7907 | -0.0047 |
| Hist Decision Tree | 0.7159 ± 0.0076 | 0.7133 | 0.5626 | 0.26 | 0.5816 | 0.4658 | 0.7739 | +0.0251 |
| Hist Random Forest | 0.7312 ± 0.0067 | 0.7338 | 0.5829 | 0.32 | 0.5936 | 0.4963 | 0.7385 | +0.0368 |
| Hist GBDT (Default) | 0.7329 ± 0.0056 | 0.7349 | 0.5878 | 0.32 | 0.5919 | 0.4974 | 0.7308 | +0.0167 |
| HistGBDT (Tuned) | 0.7327 ± 0.0058 | 0.7350 | 0.5878 | 0.29 | 0.5961 | 0.4836 | 0.7768 | +0.0069 |

## Part A.2 — Delivery Delay Classification (`late_delivery`, n=40,000, 59.29% positive, dispatch-time only)

Selected by 5-Fold CV ROC-AUC: **Logistic Regression**

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Opt Thr (OOF) | Test F1@Opt | Precision | Recall | Train-Test Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8494 ± 0.0045 | 0.8532 | 0.8941 | 0.42 | 0.8197 | 0.7746 | 0.8703 | -0.0034 |
| Calibrated Linear SVM | 0.8493 ± 0.0045 | 0.8528 | 0.8936 | 0.40 | 0.8206 | 0.7664 | 0.8832 | -0.0031 |
| Hist Decision Tree | 0.8397 ± 0.0048 | 0.8476 | 0.8877 | 0.41 | 0.8192 | 0.7589 | 0.8899 | +0.0005 |
| Hist Random Forest | 0.8431 ± 0.0046 | 0.8498 | 0.8896 | 0.46 | 0.8227 | 0.7749 | 0.8769 | +0.0079 |
| Hist GBDT (Default) | 0.8481 ± 0.0042 | 0.8524 | 0.8934 | 0.41 | 0.8188 | 0.7646 | 0.8813 | +0.0023 |
| HistGBDT (Tuned) | 0.8481 ± 0.0042 | 0.8527 | 0.8936 | 0.40 | 0.8190 | 0.7591 | 0.8891 | +0.0027 |

## Part A.3 — Marketing Lead Conversion (`converted`, n=8,000 real Olist MQL, 10.52% positive, leak-free)

Selected by 5-Fold CV ROC-AUC: **Logistic Regression**

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Opt Thr (OOF) | Test F1@Opt | Precision | Recall | Train-Test Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7232 ± 0.0113 | 0.7050 | 0.1836 | 0.18 | 0.2777 | 0.2027 | 0.4405 | +0.0233 |
| Calibrated Linear SVM | 0.7213 ± 0.0109 | 0.7044 | 0.1859 | 0.16 | 0.2821 | 0.2037 | 0.4583 | +0.0215 |
| Hist Decision Tree | 0.6991 ± 0.0147 | 0.6901 | 0.1829 | 0.20 | 0.2808 | 0.2074 | 0.4345 | +0.0573 |
| Hist Random Forest | 0.7212 ± 0.0111 | 0.7188 | 0.2065 | 0.13 | 0.2960 | 0.1951 | 0.6131 | +0.0592 |
| Hist GBDT (Default) | 0.7221 ± 0.0098 | 0.7129 | 0.1915 | 0.14 | 0.2861 | 0.1902 | 0.5774 | +0.0434 |
| HistGBDT (Tuned) | 0.7230 ± 0.0096 | 0.7155 | 0.1908 | 0.15 | 0.2983 | 0.2026 | 0.5655 | +0.0465 |

## Part B — Delivery Delay Duration Regression (`delay_days`, n=40,000, dispatch-time only)

Selected by 5-Fold CV RMSE: **Hist GBDT (Default)**

| Model | 5-Fold CV RMSE | Test RMSE | Test MAE | Test R² | Median AE | Within ±1 Day |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge Regression | 1.1762 ± 0.0099 | 1.1612 | 0.9095 | 0.5377 | 0.7533 | 62.55% |
| Hist Decision Tree | 1.1878 ± 0.0105 | 1.1708 | 0.9187 | 0.5301 | 0.7010 | 64.62% |
| Hist Random Forest | 1.1901 ± 0.0091 | 1.1816 | 0.9453 | 0.5214 | 0.7634 | 62.75% |
| Hist GBDT (Default) | 1.1703 ± 0.0114 | 1.1578 | 0.9077 | 0.5404 | 0.7324 | 63.71% |
| HistGBDT Regressor (Tuned) | 1.1708 ± 0.0108 | 1.1599 | 0.9149 | 0.5388 | 0.7456 | 63.46% |
