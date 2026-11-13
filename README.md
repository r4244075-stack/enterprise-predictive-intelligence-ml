# Enterprise Predictive Intelligence & Applied Machine Learning System (`project_02`)

**Assessment**: Vulance Data Science Internship — Project 02 (100 Marks, 30 Days)  
**Active Milestone**: Module 1 — Data Science Foundations & Problem Framing (Due November 14, 2026)  

## 1. Repository Architecture

Following the mandatory structure in [`STUDENT_SUBMISSION_GUIDELINES.md`](https://drive.google.com/file/d/1dUQq7xDrNFfcLUes9RPXVS3QWvnfuIMP/view?usp=drivesdk):

```text
project_02/
├── data/
│   ├── raw/                  # Place supplementary CSVs & Olist CSVs here (git-ignored except data_dictionary.csv)
│   └── processed/            # Leakage-safe processed splits (Module 2+)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_statistical_investigation_and_leakage_audit.ipynb
├── reports/
│   ├── Module1_Data_Science_Foundations_Report.pdf
│   ├── 01_data_quality_report.md
│   ├── 02_feature_and_target_identification.md
│   ├── 03_problem_framing_document.md
│   ├── 04_initial_statistical_investigation.md
│   └── 05_model_validation_strategy.md
├── src/
│   ├── data/audit_pipeline.py
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── tests/
│   └── test_data_leakage_and_integrity.py
├── model_cards/
├── deployment/
├── requirements.txt
└── README.md
```

## 2. Quickstart & Reproducibility Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run automated leakage and integrity unit tests
python3 -m unittest discover -s tests

# Run the Module 1 data audit script
python3 src/data/audit_pipeline.py
```

## 3. Data Limitations & Mandatory Disclosures

1. **Synthetic Supplementary Pack Disclosure**: Per [`DATASET_README.md`](https://drive.google.com/file/d/1ahrGoKWvUuOwsaCSsiCOVur9QG09ixPl/view?usp=drivesdk), all six CSVs in the [Supplementary Dataset Pack](https://drive.google.com/drive/folders/1M0tinzW_P4b_VKMs07oWjsOeicoZQQ_L) are synthetic records and are never combined numerically with financial values from the public [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) or the [Marketing Funnel by Olist Dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist).
2. **Operational Proxy Churn Label**: `no_purchase_next_90d` in `customer_retention_features.csv` is a 90-day inactivity proxy measured from `snapshot_date = 2024-01-01`, not permanent contractual churn.
3. **NLP Template Limitation**: `review_sentiment_training.csv` (20,000 rows) and `review_sentiment_unlabeled.csv` (2,000 rows) are generated from 12 exact English sentence templates. Models trained on this synthetic corpus do not generalize to free-form natural language without incorporating real review text (such as Olist's `olist_order_reviews_dataset.csv`).
4. **Strict Leakage Prevention**: `actual_delivery_days` and `delay_days` in `delivery_delay_training.csv` are post-dispatch outcomes and are strictly barred from predictor matrices (`verify_delivery_leakage_free()`).
