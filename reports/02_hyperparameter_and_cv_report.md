# Module 3 — Hyperparameter Optimization, Cross-Validation & Imbalance Report

Every hyperparameter grid candidate was evaluated via 5-fold cross-validation strictly on the 80% training split, with all transformers refit inside every fold.

## Customer Retention — HistGBDT Grid Search

| n_estimators | lr | max_depth | min_leaf | l2_lambda | CV ROC-AUC | CV PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 160 | 0.03 | 3 | 40 | 8.0 | 0.7327 ± 0.0058 | 0.5849 |
| 120 | 0.05 | 4 | 35 | 3.0 | 0.7325 ± 0.0052 | 0.5823 |
| 120 | 0.04 | 3 | 50 | 5.0 | 0.7324 ± 0.0056 | 0.5845 |
| 80 | 0.05 | 3 | 40 | 5.0 | 0.7321 ± 0.0058 | 0.5843 |

## Delivery Delay Classification — HistGBDT Grid Search

| n_estimators | lr | max_depth | min_leaf | l2_lambda | CV ROC-AUC | CV PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 160 | 0.05 | 4 | 35 | 5.0 | 0.8481 ± 0.0042 | 0.8896 |
| 140 | 0.06 | 4 | 30 | 3.0 | 0.8480 ± 0.0042 | 0.8894 |
| 100 | 0.05 | 3 | 40 | 3.0 | 0.8478 ± 0.0039 | 0.8894 |
| 140 | 0.07 | 5 | 30 | 5.0 | 0.8466 ± 0.0040 | 0.8884 |

## Marketing Lead Conversion — HistGBDT Grid Search

| n_estimators | lr | max_depth | min_leaf | l2_lambda | CV ROC-AUC | CV PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 80 | 0.06 | 3 | 25 | 3.0 | 0.7230 ± 0.0096 | 0.2106 |
| 60 | 0.04 | 3 | 40 | 5.0 | 0.7221 ± 0.0127 | 0.2078 |
| 90 | 0.05 | 3 | 30 | 5.0 | 0.7220 ± 0.0094 | 0.2077 |
| 100 | 0.04 | 4 | 35 | 8.0 | 0.7218 ± 0.0104 | 0.2109 |

## Delivery Delay Regression — HistGBDT Grid Search

| n_estimators | lr | max_depth | min_leaf | l2_lambda | CV RMSE | CV R² |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.05 | 3 | 30 | 2.0 | 1.1708 ± 0.0108 | 0.5251 |
| 180 | 0.05 | 4 | 35 | 5.0 | 1.1710 ± 0.0108 | 0.5249 |
| 140 | 0.06 | 4 | 25 | 2.0 | 1.1711 ± 0.0111 | 0.5248 |
| 140 | 0.08 | 5 | 30 | 5.0 | 1.1753 ± 0.0104 | 0.5215 |

## Class Imbalance & Decision Threshold Optimization (Untouched Test Set)

| Task | Strategy | Threshold | F1 | Precision | Recall | Balanced Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Retention | default_threshold_0_5 | 0.50 | 0.4880 | 0.6248 | 0.4004 | 0.6360 |
| Retention | balanced_class_weight_0_5 | 0.50 | 0.5941 | 0.5254 | 0.6834 | 0.6770 |
| Retention | oof_threshold_optimized | 0.28 | 0.5999 | 0.4808 | 0.7974 | 0.6689 |
| Delivery Delay | default_threshold_0_5 | 0.50 | 0.8156 | 0.8071 | 0.8242 | 0.7687 |
| Delivery Delay | balanced_class_weight_0_5 | 0.50 | 0.7995 | 0.8407 | 0.7622 | 0.7759 |
| Delivery Delay | oof_threshold_optimized | 0.42 | 0.8197 | 0.7746 | 0.8703 | 0.7508 |
| Marketing Conv. | default_threshold_0_5 | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Marketing Conv. | balanced_class_weight_0_5 | 0.50 | 0.2872 | 0.1821 | 0.6786 | 0.6605 |
| Marketing Conv. | oof_threshold_optimized | 0.18 | 0.2777 | 0.2027 | 0.4405 | 0.6186 |
