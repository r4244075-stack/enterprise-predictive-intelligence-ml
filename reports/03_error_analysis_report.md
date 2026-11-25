# Module 3 — Error Analysis & Subgroup Performance Report

## Customer Retention — Subgroup Audit (Tenure Quartile & Email Opt-In)

| Dimension | Subgroup | n | Pos Rate | ROC-AUC | F1 | Precision | Recall | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenure_quartile | Q1 (1-16m) | 1560 | 0.511 | 0.6766 | 0.6866 | 0.5372 | 0.9511 | 0.8583 | 0.0489 |
| tenure_quartile | Q2 (17-31m) | 1497 | 0.398 | 0.7208 | 0.6267 | 0.4874 | 0.8773 | 0.6086 | 0.1227 |
| tenure_quartile | Q3 (32-46m) | 1547 | 0.290 | 0.6908 | 0.5000 | 0.4137 | 0.6317 | 0.3649 | 0.3683 |
| tenure_quartile | Q4 (47-60m) | 1396 | 0.177 | 0.6920 | 0.3727 | 0.3424 | 0.4089 | 0.1688 | 0.5911 |
| marketing_email_opt_in | 0 | 3035 | 0.352 | 0.7436 | 0.6007 | 0.4814 | 0.7987 | 0.4672 | 0.2013 |
| marketing_email_opt_in | 1 | 2965 | 0.344 | 0.7399 | 0.5990 | 0.4802 | 0.7961 | 0.4519 | 0.2039 |

## Delivery Delay Classification — Subgroup Audit (Weather & Region)

| Dimension | Subgroup | n | Pos Rate | ROC-AUC | F1 | Precision | Recall | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weather_at_dispatch | clear | 4939 | 0.499 | 0.8286 | 0.7580 | 0.7160 | 0.8052 | 0.3180 | 0.1948 |
| weather_at_dispatch | heat | 664 | 0.566 | 0.8551 | 0.8216 | 0.7786 | 0.8697 | 0.3229 | 0.1303 |
| weather_at_dispatch | rain | 1780 | 0.743 | 0.8176 | 0.8695 | 0.8126 | 0.9349 | 0.6223 | 0.0651 |
| weather_at_dispatch | storm | 617 | 0.942 | 0.8328 | 0.9699 | 0.9417 | 1.0000 | 1.0000 | 0.0000 |
| region | central | 1606 | 0.618 | 0.8685 | 0.8370 | 0.8054 | 0.8711 | 0.3409 | 0.1289 |
| region | east | 1581 | 0.592 | 0.8432 | 0.8055 | 0.7677 | 0.8472 | 0.3721 | 0.1528 |
| region | north | 1585 | 0.580 | 0.8545 | 0.8217 | 0.7687 | 0.8825 | 0.3664 | 0.1175 |
| region | south | 1630 | 0.610 | 0.8351 | 0.8135 | 0.7733 | 0.8581 | 0.3931 | 0.1419 |
| region | west | 1598 | 0.564 | 0.8655 | 0.8204 | 0.7575 | 0.8946 | 0.3702 | 0.1054 |

## Marketing Lead Conversion — Subgroup Audit (Acquisition Origin)

| Dimension | Subgroup | n | Pos Rate | ROC-AUC | F1 | Precision | Recall | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| origin_clean | direct_traffic | 99 | 0.101 | 0.5826 | 0.0556 | 0.0385 | 0.1000 | 0.2809 | 0.9000 |
| origin_clean | email | 97 | 0.010 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| origin_clean | organic_search | 463 | 0.106 | 0.7097 | 0.3114 | 0.2203 | 0.5306 | 0.2222 | 0.4694 |
| origin_clean | paid_search | 346 | 0.142 | 0.5803 | 0.2336 | 0.1818 | 0.3265 | 0.2424 | 0.6735 |
| origin_clean | referral | 62 | 0.065 | 0.6466 | 0.1818 | 0.1429 | 0.2500 | 0.1034 | 0.7500 |
| origin_clean | social | 252 | 0.052 | 0.7221 | 0.0000 | 0.0000 | 0.0000 | 0.0084 | 1.0000 |
| origin_clean | unknown | 223 | 0.179 | 0.6431 | 0.3727 | 0.2479 | 0.7500 | 0.4973 | 0.2500 |

## Delivery Delay Regression — Residual & Subgroup Diagnostics

- Residual Mean: **-0.0013 days** (unbiased)
- Residual Std: **1.1578 days**
- Skewness: **0.4063** (mild right tail from extreme storm delays)
- Excess Kurtosis: **0.4232**

| Dimension | Subgroup | n | Mean Actual | Mean Pred | RMSE | MAE | R² | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weather_at_dispatch | clear | 4929 | 1.39 | 1.40 | 1.0786 | 0.8362 | 0.4225 | +0.0084 |
| weather_at_dispatch | heat | 647 | 1.64 | 1.71 | 1.1357 | 0.8937 | 0.4421 | +0.0669 |
| weather_at_dispatch | rain | 1826 | 2.33 | 2.33 | 1.2791 | 1.0351 | 0.4310 | -0.0043 |
| weather_at_dispatch | storm | 598 | 4.13 | 4.02 | 1.3980 | 1.1235 | 0.4293 | -0.1108 |
| region | central | 1622 | 1.85 | 1.82 | 1.1649 | 0.9220 | 0.5278 | -0.0308 |
| region | east | 1606 | 1.83 | 1.84 | 1.1403 | 0.8880 | 0.5606 | +0.0082 |
| region | north | 1641 | 1.84 | 1.83 | 1.1403 | 0.8940 | 0.5523 | -0.0064 |
| region | south | 1562 | 1.79 | 1.80 | 1.1771 | 0.9186 | 0.5049 | +0.0084 |
| region | west | 1569 | 1.84 | 1.87 | 1.1672 | 0.9168 | 0.5529 | +0.0285 |

## Class-Wise Metrics & Confusion Matrices (Untouched Test Sets)

### Customer Retention (Logistic Regression)
**Confusion Matrix (Untouched Test Set):** TN = 2,114 | FP = 1,798 | FN = 423 | TP = 1,665

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Class 0 (Negative) | 0.8333 | 0.5404 | 0.6556 | 3,912 |
| Class 1 (Positive) | 0.4808 | 0.7974 | 0.5999 | 2,088 |

### Delivery Delay Classification (Logistic Regression)
**Confusion Matrix (Untouched Test Set):** TN = 2,056 | FP = 1,201 | FN = 615 | TP = 4,128

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Class 0 (Negative) | 0.7697 | 0.6313 | 0.6937 | 3,257 |
| Class 1 (Positive) | 0.7746 | 0.8703 | 0.8197 | 4,743 |

### Marketing Lead Conversion (HistGBDT Tuned)
**Confusion Matrix (Untouched Test Set):** TN = 1,058 | FP = 374 | FN = 73 | TP = 95

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Class 0 (Negative) | 0.9355 | 0.7388 | 0.8256 | 1,432 |
| Class 1 (Positive) | 0.2026 | 0.5655 | 0.2983 | 168 |


## Representative Error Examples (High-Confidence FP & FN Cases)

### Customer Retention — Top False Positives & False Negatives

| Type | customer_id | Tenure (m) | Orders 180d | Recency (d) | Tickets | Actual | Pred Prob |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | DS_CUST_015532 | 2 | 0 | 166 | 4 | 0 | 0.8215 |
| FP | DS_CUST_007362 | 5 | 0 | 173 | 2 | 0 | 0.8213 |
| FP | DS_CUST_028923 | 1 | 0 | 167 | 2 | 0 | 0.8151 |
| FN | DS_CUST_010028 | 58 | 11 | 20 | 0 | 1 | 0.0226 |
| FN | DS_CUST_008214 | 59 | 10 | 23 | 1 | 1 | 0.0380 |
| FN | DS_CUST_006358 | 58 | 15 | 78 | 0 | 1 | 0.0409 |

### Delivery Delay — Top False Positives & False Negatives

| Type | delivery_id | Distance (km) | Carrier Load | Weather | Region | Actual | Pred Prob |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP | DS_DEL_0017385 | 1521.0 | 1.21 | rain | north | 0 | 0.9934 |
| FP | DS_DEL_0004697 | 184.1 | 1.13 | storm | west | 0 | 0.9896 |
| FP | DS_DEL_0015752 | 909.3 | 0.86 | storm | west | 0 | 0.9891 |
| FN | DS_DEL_0003045 | 136.1 | 0.43 | clear | east | 1 | 0.0523 |
| FN | DS_DEL_0014094 | 112.2 | 0.46 | clear | east | 1 | 0.0587 |
| FN | DS_DEL_0028299 | 183.2 | 0.45 | clear | south | 1 | 0.0657 |
