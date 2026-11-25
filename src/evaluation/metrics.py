"""Exact classification, regression, and forecasting evaluation metrics."""

from __future__ import annotations
import numpy as np
from scipy import stats


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    ranks = stats.rankdata(y_score)
    rank_sum_pos = np.sum(ranks[y_true == 1])
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray, max_points: int = 200):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / max(tps[-1], 1)])
    fpr = np.concatenate([[0.0], fps / max(fps[-1], 1)])
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (step-function PR-AUC)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    prec = tps / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(prec[y_sorted == 1]) / pos)


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray, max_points: int = 200):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    rec = tps / max(tps[-1], 1)
    prec = tps / np.arange(1, len(y_sorted) + 1)
    if len(rec) > max_points:
        idx = np.linspace(0, len(rec) - 1, max_points).astype(int)
        rec, prec = rec[idx], prec[idx]
    return rec, prec


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    bal_acc = 0.5 * (rec + spec)
    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(pr_auc_score(y_true, y_prob), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "threshold": round(float(threshold), 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def optimize_threshold(y_val: np.ndarray, p_val: np.ndarray, metric: str = "f1", fn_cost: float = 1.0, fp_cost: float = 1.0) -> tuple[float, float]:
    """Find optimal decision threshold strictly on validation / OOF predictions."""
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=np.float64)
    grid = np.linspace(0.05, 0.90, 86)
    best_t = 0.5
    best_score = -np.inf
    for t in grid:
        yp = (p_val >= t).astype(int)
        tp = np.sum((y_val == 1) & (yp == 1))
        fp = np.sum((y_val == 0) & (yp == 1))
        fn = np.sum((y_val == 1) & (yp == 0))
        if metric == "f1":
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            sc = 2 * prec * rec / max(prec + rec, 1e-12)
        elif metric == "cost":
            sc = -(fn_cost * fn + fp_cost * fp) / max(len(y_val), 1)
        else:
            tn = np.sum((y_val == 0) & (yp == 0))
            sc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        if sc > best_score:
            best_score = float(sc)
            best_t = float(t)
    return round(best_t, 3), round(best_score, 4)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    med_ae = float(np.median(np.abs(resid)))
    within_1d = float(np.mean(np.abs(resid) <= 1.0))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(float(r2), 4),
        "median_ae": round(med_ae, 4),
        "within_1_day_pct": round(100.0 * within_1d, 2),
        "resid_mean": round(float(np.mean(resid)), 4),
        "resid_std": round(float(np.std(resid)), 4),
        "resid_skew": round(float(stats.skew(resid)), 4),
        "resid_kurtosis": round(float(stats.kurtosis(resid)), 4),
    }
