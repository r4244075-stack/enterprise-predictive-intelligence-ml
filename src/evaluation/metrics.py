#!/usr/bin/env python3
"""Module 2: Statistical & Business Evaluation Metrics for Classification, Regression, Forecasting & NLP."""
import numpy as np
from scipy import stats

def roc_auc_score_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos, n_neg = int(np.sum(pos)), int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(y_score)
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def pr_auc_score_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_s = y_true[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(np.sum(y_true), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))

def evaluate_binary_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc_score_safe(y_true, y_prob), 4),
        "pr_auc": round(pr_auc_score_safe(y_true, y_prob), 4),
        "brier_score": round(brier, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def evaluate_forecast_predictions(y_true: np.ndarray, y_pred: np.ndarray, seasonal_naive_mae: float = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom > 0, 200.0 * np.abs(y_true - y_pred) / denom, 0.0)))
    res = {"mae": round(mae, 4), "rmse": round(rmse, 4), "smape_pct": round(smape, 2)}
    if seasonal_naive_mae and seasonal_naive_mae > 0:
        res["mase"] = round(mae / seasonal_naive_mae, 4)
    return res
