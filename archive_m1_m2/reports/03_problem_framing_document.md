# Module 1 Deliverable: Enterprise Problem-Framing & Business Impact Document

**Project**: Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02)  
**Module**: Module 1 — Data Science Foundations & Problem Framing  
**Date**: November 12, 2026  

## 1. Problem Formulation Across the 6 Mandatory Business Domains

### Problem 1: Customer Retention & 90-Day Lapse Prediction
- **Business Decision Context**: Identifying existing customers at high risk of lapsing (`no_purchase_next_90d = 1`, base rate `34.79%`) on `snapshot_date` (`2024-01-01`) allows lifecycle marketing teams to trigger targeted win-back incentives before customer attrition becomes permanent.
- **Predictive Objective**: Output calibrated probability estimates $\hat{p}_i = P(	ext{no\_purchase\_next\_90d}_i = 1 \mid \mathbf{x}_{i, t_0})$ using strictly point-in-time behavioural, recency, support-ticket, and return-history features.
- **Operational & Financial Trade-off**: False negatives (missed churners) cost lost future gross merchandise value (GMV, where mean `avg_order_value` is `$82.47`), whereas false positives waste promotional margin (especially among high `discount_order_share` buyers). Optimal probability thresholds are selected by maximizing expected net retention value rather than arbitrary `0.50` cutoffs.

### Problem 2: Dispatch-Time Delivery Delay Classification & Duration Regression
- **Business Decision Context**: Late deliveries (`late_delivery = 1`, base rate `59.29%`, defined operational threshold `delay_days > 1.0` beyond `promised_delivery_days`) are the primary driver of 1-star customer reviews and support ticket escalations.
- **Predictive Objective**:
  1. **Classification**: Predict $P(	ext{late\_delivery}_i = 1 \mid \mathbf{x}_{	ext{dispatch}})$ at order dispatch to trigger proactive customer notifications or carrier rerouting.
  2. **Regression**: Predict expected $\widehat{	ext{delay\_days}}_i \in [0, \infty)$ to dynamically adjust SLA buffer promises at checkout.
- **Operational & Financial Trade-off**: Because adverse weather (`storm` = `94.39%` late, `rain` = `73.44%` late) and high `carrier_load_index` ($r = 0.4852$) dominate delay risk, dispatch-time models enable dynamic SLA padding (+2 to +4 days) during high-load/storm regimes.

### Problem 3: Multi-Horizon SKU Demand Forecasting
- **Business Decision Context**: Inventory stockouts lose immediate sales, while overstocking ties up working capital across the `150` products in [`daily_product_demand.csv`](https://drive.google.com/file/d/1UagJBHolBS5olzdLv83Ge2dCeSn8gNMR/view?usp=drivesdk).
- **Predictive Objective**: Generate multi-horizon daily point and quantile forecasts $\hat{y}_{p, t+h}$ (`units_sold` for product $p$ at horizon $h \in \{1, \dots, 30\}$ days) conditioned on calendar seasonality (`day_of_week`, `month`), autoregressive lags, and planned `promotion_flag` schedules.

### Problem 4: Automated Customer Review Sentiment Triage (NLP)
- **Business Decision Context**: Automatically classifying incoming customer feedback (`positive`, `neutral`, `negative`) routes dissatisfied buyers (`negative` = `25.04%` in training, `32.70%` in unlabeled inference) to customer recovery queues.
- **Predictive Objective & Scope Disclosure**: Train a leakage-safe TF-IDF + Linear/Logistic classifier on [`review_sentiment_training.csv`](https://drive.google.com/file/d/1Lv6G6pPHgA6tSiL_vdvD90uPljw-P_IP/view?usp=drivesdk), produce batch predictions on [`review_sentiment_unlabeled.csv`](https://drive.google.com/file/d/1zGsB2h52_GQJLD5o2nHUJ3yB1mAJDjlz/view?usp=drivesdk), and explicitly disclose the 12-template synthetic limitation per [`DATASET_README.md`](https://drive.google.com/file/d/1ahrGoKWvUuOwsaCSsiCOVur9QG09ixPl/view?usp=drivesdk) while benchmarking real-world generalization on [Olist Order Reviews](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

### Problem 5: B2B Seller Marketing Funnel Lead Conversion
- **Business Decision Context**: In the [Olist Marketing Funnel Dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist), Sales Development Representatives (SDRs) and Sales Representatives (SRs) process `8,000` inbound Marketing Qualified Leads (MQLs) with an overall conversion rate of `10.53%` (`842` closed deals).
- **Predictive Objective**: Score inbound MQLs at `first_contact_date` using acquisition `origin` and `landing_page_id` to prioritize high-propensity seller leads and optimize paid vs. organic acquisition spend.

### Problem 6: Customer Behavioural Segmentation
- **Business Decision Context**: Monolithic retention campaigns treat all customers identically despite distinct underlying friction causes (e.g., product returns vs. support escalations vs. natural lifecycle decay).
- **Predictive Objective**: Partition the `30,000` customers in [`customer_retention_features.csv`](https://drive.google.com/file/d/1YTgNMMZlfivpcNLbj3vGpVNtuR0c8gIz/view?usp=drivesdk) into interpretable behavioural personas using standardized RFM, engagement, return, and support features to tailor lifecycle interventions.
