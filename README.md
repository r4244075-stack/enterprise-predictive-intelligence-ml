# Enterprise Predictive Intelligence & Applied Machine Learning System (Project 02 — Module 3)

Complete, self-contained, 100% reproducible implementation for **Module 3: Advanced Machine Learning & Statistical Modelling** (plus a reproducible **Module 2 Leakage Audit Addendum**) for the Vulance Data Science Internship.

## Quick Reproduction (Zero External ML Dependency Friction)
```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m unittest discover -s tests -v
```

## Project Structure
- `src/features/pipelines.py` — Leak-free `TabularPreprocessor`, `OutOfFoldTargetEncoder`, and domain feature engineering.
- `src/models/estimators.py` — Vectorized NumPy/SciPy estimators (L2 Logistic Regression, Calibrated Linear SVM, Histogram Decision Tree, Random Forest, Newton-Raphson HistGBDT, Ridge Regression).
- `src/evaluation/` — Exact metrics, stratified/group/k-fold CV, threshold optimization, subgroup error auditing, K-Means++/Ward/DBSCAN clustering, SHAP, PFI, and 1D PDP.
- `src/data/leakage_audit.py` — Reproducible leakage audits directly answering all 4 Module 2 evaluator questions.
- `tests/test_leakage_and_pipelines.py` — 22 automated unit & leakage-invariant tests (all passing).
- `reports/` — 7 detailed Markdown reports, CSV/JSON metrics (`reports/metrics/`), and 8 high-DPI diagnostic figures (`reports/figures/`).
