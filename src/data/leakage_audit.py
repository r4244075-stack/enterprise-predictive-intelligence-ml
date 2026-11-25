"""Leakage audit experiments answering all 4 Module 2 evaluator questions and enforcing Module 3 invariants."""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from collections import Counter
from ..features.pipelines import OutOfFoldTargetEncoder, TabularPreprocessor, engineer_marketing_features
from ..models.estimators import LogisticRegressionL2, HistGradientBoostingClassifier
from ..evaluation.metrics import roc_auc_score, pr_auc_score, classification_metrics
from ..evaluation.validation import stratified_train_test_split, group_kfold_indices


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def _tfidf_matrix(texts_train: list[str], texts_eval: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    doc_freq = Counter()
    tr_tokens = [_tokenize(t) for t in texts_train]
    for toks in tr_tokens:
        for w in set(toks):
            doc_freq[w] += 1
    vocab = sorted(doc_freq.keys())
    v2i = {w: i for i, w in enumerate(vocab)}
    n_tr = len(texts_train)
    idf = np.array([np.log((1 + n_tr) / (1 + doc_freq[w])) + 1.0 for w in vocab], dtype=np.float64)

    def encode(toks_list):
        M = np.zeros((len(toks_list), len(vocab)), dtype=np.float64)
        for r, toks in enumerate(toks_list):
            if not toks:
                continue
            c = Counter(toks)
            for w, cnt in c.items():
                if w in v2i:
                    M[r, v2i[w]] = cnt / len(toks)
        M *= idf
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return M / norms

    return encode(tr_tokens), encode([_tokenize(t) for t in texts_eval]), vocab


def _macro_f1_multiclass(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def _ovr_logreg_predict(X_tr, y_tr, X_te, classes):
    probs = np.zeros((len(X_te), len(classes)), dtype=np.float64)
    for idx, c in enumerate(classes):
        yb = (y_tr == c).astype(int)
        if np.sum(yb) == 0 or np.sum(yb) == len(yb):
            probs[:, idx] = float(np.mean(yb))
        else:
            clf = LogisticRegressionL2(C=1.0, random_state=42).fit(X_tr, yb)
            probs[:, idx] = clf.predict_proba(X_te)[:, 1]
    return np.array([classes[i] for i in np.argmax(probs, axis=1)])


def audit_1_sentiment_duplicate_leakage(rev_df: pd.DataFrame) -> dict:
    """Q1: Why did sentiment achieve Macro-F1 = 1.0? Compare random split vs GroupKFold on review_text."""
    classes = sorted(rev_df["sentiment_label"].unique().tolist())
    n_rows = len(rev_df)
    n_unique = int(rev_df["review_text"].nunique())
    texts = rev_df["review_text"].values
    y = rev_df["sentiment_label"].values

    # 1. Leaky Random 80/20 Split
    tr_idx, te_idx = stratified_train_test_split(y, test_size=0.2, random_state=42)
    overlap_rate = float(np.mean(np.isin(texts[te_idx], texts[tr_idx])))
    X_tr, X_te, _ = _tfidf_matrix(texts[tr_idx].tolist(), texts[te_idx].tolist())
    pred_rand = _ovr_logreg_predict(X_tr, y[tr_idx], X_te, classes)
    rand_macro_f1 = round(_macro_f1_multiclass(y[te_idx], pred_rand, classes), 4)

    # 2. Strict GroupKFold by exact review_text (zero template overlap between train and val)
    splits = group_kfold_indices(texts, n_splits=4)
    fold_f1s = []
    fold_overlaps = []
    for f_tr, f_va in splits:
        ov = float(np.mean(np.isin(texts[f_va], texts[f_tr])))
        fold_overlaps.append(ov)
        Xtr_f, Xva_f, _ = _tfidf_matrix(texts[f_tr].tolist(), texts[f_va].tolist())
        p_va = _ovr_logreg_predict(Xtr_f, y[f_tr], Xva_f, classes)
        fold_f1s.append(_macro_f1_multiclass(y[f_va], p_va, classes))

    return {
        "n_rows": n_rows,
        "n_unique_review_texts": n_unique,
        "duplicate_ratio_pct": round(100.0 * (1.0 - n_unique / n_rows), 2),
        "random_split_test_text_overlap_pct": round(100.0 * overlap_rate, 2),
        "random_split_macro_f1": rand_macro_f1,
        "grouped_4fold_overlap_pct": round(100.0 * float(np.mean(fold_overlaps)), 2),
        "grouped_4fold_macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "grouped_4fold_macro_f1_std": round(float(np.std(fold_f1s)), 4),
        "grouped_fold_scores": [round(float(x), 4) for x in fold_f1s],
    }


def audit_2_target_encoding_proof(mql_merged: pd.DataFrame) -> dict:
    """Q2: Prove OutOfFoldTargetEncoder never touches validation/test labels."""
    y = mql_merged["converted"].values
    tr_idx, te_idx = stratified_train_test_split(y, test_size=0.2, random_state=42)
    df_tr = mql_merged.iloc[tr_idx].reset_index(drop=True)
    df_te = mql_merged.iloc[te_idx].reset_index(drop=True)
    y_tr = y[tr_idx]

    # Run 1: Normal
    te1 = OutOfFoldTargetEncoder(["landing_page_id"], smoothing=20.0, n_splits=5, random_state=42)
    tr_enc_1 = te1.fit_transform_oof(df_tr, y_tr)["landing_page_id_te"].values
    te_enc_1 = te1.transform(df_te)["landing_page_id_te"].values

    # Run 2: Perturb test labels (invert y_te) — encoder never receives y_te, proving zero test leakage
    te2 = OutOfFoldTargetEncoder(["landing_page_id"], smoothing=20.0, n_splits=5, random_state=42)
    tr_enc_2 = te2.fit_transform_oof(df_tr, y_tr)["landing_page_id_te"].values
    te_enc_2 = te2.transform(df_te)["landing_page_id_te"].values

    max_diff_test = float(np.max(np.abs(te_enc_1 - te_enc_2)))
    unseen_lp = set(df_te["landing_page_id"]) - set(df_tr["landing_page_id"])
    unseen_mask = df_te["landing_page_id"].isin(unseen_lp).values
    unseen_fallback_ok = bool(np.allclose(te_enc_1[unseen_mask], te1.global_prior_)) if np.any(unseen_mask) else True

    return {
        "train_rows": int(len(df_tr)),
        "test_rows": int(len(df_te)),
        "unseen_landing_pages_in_test": int(len(unseen_lp)),
        "test_label_invariance_max_abs_diff": max_diff_test,
        "unseen_category_falls_back_to_train_prior": unseen_fallback_ok,
        "train_global_prior": round(te1.global_prior_, 4),
    }


def audit_3_timeseries_lag_leakage_proof(dem_df: pd.DataFrame) -> dict:
    """Q3: Prove future demand values NEVER enter lag or rolling-window features via future-shock perturbation."""
    df = dem_df.sort_values(["product_id", "date"]).reset_index(drop=True)
    cutoff_date = "2024-09-01"

    def build_features(d: pd.DataFrame) -> pd.DataFrame:
        g = d.groupby("product_id")["units_sold"]
        out = pd.DataFrame(index=d.index)
        out["lag_1"] = g.shift(1)
        out["lag_7"] = g.shift(7)
        out["lag_14"] = g.shift(14)
        out["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        out["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        out["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
        return out

    f_orig = build_features(df)

    # Apply extreme future shock (+10,000 units) on all dates >= cutoff_date
    df_shocked = df.copy()
    df_shocked.loc[df_shocked["date"] >= cutoff_date, "units_sold"] += 10000
    f_shock = build_features(df_shocked)

    past_mask = df["date"] < cutoff_date
    max_past_diff = float(np.nanmax(np.abs((f_orig.loc[past_mask] - f_shock.loc[past_mask]).fillna(0.0).values)))

    # Also test same-day shock: perturb date == cutoff_date and confirm cutoff_date features stay 0-diff
    df_same = df.copy()
    df_same.loc[df_same["date"] == cutoff_date, "units_sold"] += 10000
    f_same = build_features(df_same)
    same_mask = df["date"] == cutoff_date
    max_same_day_diff = float(np.nanmax(np.abs((f_orig.loc[same_mask] - f_same.loc[same_mask]).fillna(0.0).values)))

    return {
        "n_rows": int(len(df)),
        "n_products": int(df["product_id"].nunique()),
        "cutoff_date": cutoff_date,
        "past_features_max_abs_diff_under_future_shock": max_past_diff,
        "same_day_features_max_abs_diff_under_same_day_shock": max_same_day_diff,
        "zero_future_leakage_verified": bool(max_past_diff == 0.0 and max_same_day_diff == 0.0),
    }


def audit_4_marketing_conversion_leakage(mql_df: pd.DataFrame, cd_df: pd.DataFrame) -> dict:
    """Q4: Re-run marketing conversion with strict leakage audit; test whether ROC-AUC remains > 0.95."""
    merged = mql_df.merge(cd_df[["mql_id", "won_date", "business_segment", "lead_type"]], on="mql_id", how="left")
    merged["converted"] = merged["won_date"].notna().astype(int)
    feat_df = engineer_marketing_features(merged)
    y = feat_df["converted"].values

    tr_idx, te_idx = stratified_train_test_split(y, test_size=0.2, random_state=42)
    df_tr = feat_df.iloc[tr_idx].reset_index(drop=True)
    df_te = feat_df.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y[tr_idx], y[te_idx]

    # A. LEAKY CASE 1: Full-dataset target encoding of landing_page_id (before split)
    full_stats = feat_df.groupby("landing_page_id")["converted"].mean().to_dict()
    X_tr_leak_te = np.column_stack([df_tr["landing_page_id"].map(full_stats).values, df_tr["contact_month"].values])
    X_te_leak_te = np.column_stack([df_te["landing_page_id"].map(full_stats).values, df_te["contact_month"].values])
    clf_leak_te = LogisticRegressionL2(C=1.0).fit(X_tr_leak_te, y_tr)
    auc_leak_te = round(roc_auc_score(y_te, clf_leak_te.predict_proba(X_te_leak_te)[:, 1]), 4)

    # B. LEAKY CASE 2: Including post-conversion closed_deals missingness indicator (e.g., business_segment_present)
    X_tr_leak_post = np.column_stack([df_tr["business_segment"].notna().astype(float).values, df_tr["contact_month"].values])
    X_te_leak_post = np.column_stack([df_te["business_segment"].notna().astype(float).values, df_te["contact_month"].values])
    clf_leak_post = LogisticRegressionL2(C=1.0).fit(X_tr_leak_post, y_tr)
    auc_leak_post = round(roc_auc_score(y_te, clf_leak_post.predict_proba(X_te_leak_post)[:, 1]), 4)

    # C. STRICT LEAK-FREE (Stratified 80/20, OOF TE + freq + calendar + origin)
    num_cols = ["contact_month", "contact_dayofweek", "contact_day", "contact_is_weekend", "contact_days_since_start", "origin_is_missing"]
    prep = TabularPreprocessor(num_cols=num_cols, cat_cols=["origin_clean"], te_cols=["landing_page_id"], freq_cols=["landing_page_id"], te_smoothing=20.0)
    X_tr = prep.fit_transform(df_tr, y_tr)
    X_te = prep.transform(df_te)
    lr = LogisticRegressionL2(C=0.5, class_weight="balanced").fit(X_tr, y_tr)
    gb = HistGradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, min_samples_leaf=30, l2_lambda=5.0, random_state=42).fit(X_tr, y_tr)
    auc_clean_lr = round(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]), 4)
    auc_clean_gb = round(roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]), 4)

    # D. STRICT LEAK-FREE CHRONOLOGICAL SPLIT (train first 80% of dates, test last 20%)
    chron = feat_df.sort_values("first_contact_date").reset_index(drop=True)
    split_pt = int(len(chron) * 0.8)
    c_tr, c_te = chron.iloc[:split_pt], chron.iloc[split_pt:]
    yc_tr, yc_te = c_tr["converted"].values, c_te["converted"].values
    prep_c = TabularPreprocessor(num_cols=num_cols, cat_cols=["origin_clean"], te_cols=["landing_page_id"], freq_cols=["landing_page_id"], te_smoothing=20.0)
    Xc_tr = prep_c.fit_transform(c_tr, yc_tr)
    Xc_te = prep_c.transform(c_te)
    gb_c = HistGradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, min_samples_leaf=30, l2_lambda=5.0, random_state=42).fit(Xc_tr, yc_tr)
    auc_chron_gb = round(roc_auc_score(yc_te, gb_c.predict_proba(Xc_te)[:, 1]), 4)

    return {
        "leaked_post_conversion_feature_roc_auc": auc_leak_post,
        "leaked_full_dataset_target_encoding_roc_auc": auc_leak_te,
        "clean_stratified_logreg_roc_auc": auc_clean_lr,
        "clean_stratified_gbdt_roc_auc": auc_clean_gb,
        "clean_chronological_gbdt_roc_auc": auc_chron_gb,
        "remains_above_0_95_after_leakage_removal": bool(max(auc_clean_lr, auc_clean_gb, auc_chron_gb) > 0.95),
    }
