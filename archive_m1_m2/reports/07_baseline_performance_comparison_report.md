# Module 2 Deliverable: Baseline Performance Comparison Report

**Project**: Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02)  
**Module**: Module 2 — Advanced Feature Engineering & Baseline Modelling  
**Date**: November 14, 2026  

## 1. Classification Baselines Summary

### 1.1 Customer Retention Classification (`no_purchase_next_90d`, Test $n = 5,999$, 34.79% Positive)
| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Justification & Key Takeaway |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Dummy Base-Rate | 0.5000 | 0.2155 | 0.0000 | 0.0000 | 0.0000 | Floor benchmark predicting class 0. |
| Raw Logistic Regression | 0.7421 | 0.5984 | 0.4852 | 0.6318 | 0.3939 | Misses 60.6% of churners due to class imbalance. |
| **Balanced Engineered LogReg (t=0.50)** | **0.7426** | **0.5985** | **0.5961** | 0.5298 | **0.6814** | Class weighting lifts F1 by **+0.111** and Recall by **+28.8 pts**. |
| Balanced Engineered LogReg (t=0.44*) | 0.7426 | 0.5985 | **0.5975** | 0.4909 | **0.7633** | Threshold tuned on training set maximizes F1 and retention capture. |

### 1.2 Delivery Delay Classification (`late_delivery`, Test $n = 7,999$, 59.29% Positive)
| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Justification & Key Takeaway |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Majority Class Baseline | 0.5000 | 0.3920 | 0.7445 | 0.5929 | 1.0000 | Predicts all deliveries late; zero discrimination. |
| Weather-Rule Baseline | 0.6351 | 0.7306 | 0.7445 | 0.5929 | 1.0000 | Captures weather prior (`storm` 94.4% late) but ignores carrier load. |
| **Engineered LogReg (t=0.50)** | **0.8500** | **0.8911** | **0.8129** | 0.7961 | 0.8305 | Driven by `carrier_load_index` ($eta=+1.40$) & `weather_storm` ($eta=+0.98$). |
| Balanced Engineered LogReg | 0.8500 | 0.8911 | 0.7986 | **0.8320** | 0.7677 | Higher precision (83.2%) when minimizing false delay alerts. |

### 1.3 Marketing Lead Conversion Classification (Olist Funnel Benchmark, 10.53% Positive)
- **Baseline Design**: Chronological split by `first_contact_date` using `BalancedLogisticBaseline` on `origin` + out-of-fold `landing_page_id` + contact calendar features, evaluated with **PR-AUC** and **Top-Decile SDR Lift** (with all 13 `closed_deals` columns strictly excluded).

## 2. Product Demand Forecasting Baselines (`daily_product_demand.csv`, Q4 2024 Test $n = 13,650$)

| Forecasting Model | MAE (units) | RMSE (units) | sMAPE (%) | Improvement vs. Seasonal Naive |
| :--- | ---: | ---: | ---: | :--- |
| Naive (`lag_1`) | 5.9759 | 8.9077 | 32.23% | -5.2% worse MAE (ignores weekly cycle) |
| Seasonal Naive (`lag_7`) | 5.6821 | 8.5258 | 30.87% | Reference seasonal benchmark |
| 7-Day Shifted Rolling Mean | 4.8020 | 7.0994 | 26.24% | -15.5% MAE / -16.7% RMSE |
| **Ridge Regression (Lags + Promo + Fourier)** | **4.1533** | **5.9223** | 27.80% | **-26.9% MAE / -30.5% RMSE** |

## 3. Review Sentiment NLP Baseline (`review_sentiment_training.csv`, Test $n = 3,999$)

- **Majority Class Baseline**: `Accuracy = 0.5779`, `Macro-F1 = 0.2442`.
- **TF-IDF (Unigram + Bigram, Vocab = 150 fit on train only) + Multinomial Naive Bayes ($lpha = 1.0$)**: `Accuracy = 1.0000`, `Macro-F1 = 1.0000` across all 3 classes (`negative`, `neutral`, `positive`).
- **Unlabeled Inference (`review_sentiment_unlabeled.csv`, $n = 2,000$)**: Predicted `710` neutral (35.5%), `654` negative (32.7%), and `636` positive (31.8%), confirming the expected prior shift relative to training. **Limitation**: Reflects memorization of the 12 synthetic templates; real-world generalization will be benchmarked on Olist Portuguese/English free-text reviews in Module 4.
