"""Leak-free splitting, cross-validation, hyperparameter tuning, learning curves, and subgroup error analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import classification_metrics, regression_metrics, roc_auc_score, pr_auc_score


def stratified_train_test_split(y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    tr_list, te_list = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_size)))
        te_list.append(idx[:n_te])
        tr_list.append(idx[n_te:])
    tr = np.concatenate(tr_list)
    te = np.concatenate(te_list)
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


def regression_train_test_split(n: int, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = int(round(n * test_size))
    return idx[n_te:], idx[:n_te]


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % n_splits].append(i)
    all_idx = np.arange(len(y))
    splits = []
    for f in range(n_splits):
        val_idx = np.array(sorted(folds[f]), dtype=int)
        mask = np.ones(len(y), dtype=bool)
        mask[val_idx] = False
        splits.append((all_idx[mask], val_idx))
    return splits


def kfold_indices(n: int, n_splits: int = 5, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    chunks = np.array_split(idx, n_splits)
    splits = []
    for f in range(n_splits):
        val_idx = chunks[f]
        tr_idx = np.concatenate([chunks[k] for k in range(n_splits) if k != f])
        splits.append((tr_idx, val_idx))
    return splits


def group_kfold_indices(groups: np.ndarray, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strict GroupKFold: no group value ever appears in both train and validation."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g, c in zip(uniq[order], counts[order]):
        f = int(np.argmin(fold_sizes))
        fold_groups[f].append(g)
        fold_sizes[f] += c
    splits = []
    for f in range(n_splits):
        val_mask = np.isin(groups, fold_groups[f])
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def subgroup_classification_audit(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    subgroup_col: str,
    min_n: int = 30,
) -> list[dict]:
    rows = []
    s = df_test[subgroup_col].fillna("missing").astype(str).values
    for grp in sorted(np.unique(s)):
        mask = s == grp
        if int(np.sum(mask)) < min_n:
            continue
        m = classification_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        rows.append({
            "subgroup_col": subgroup_col,
            "subgroup": grp,
            "n": int(np.sum(mask)),
            "pos_rate": round(float(np.mean(y_true[mask])), 4),
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "fpr": round(m["fp"] / max(m["fp"] + m["tn"], 1), 4),
            "fnr": round(m["fn"] / max(m["fn"] + m["tp"], 1), 4),
        })
    return rows


def subgroup_regression_audit(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_col: str,
    min_n: int = 30,
) -> list[dict]:
    rows = []
    s = df_test[subgroup_col].fillna("missing").astype(str).values
    for grp in sorted(np.unique(s)):
        mask = s == grp
        if int(np.sum(mask)) < min_n:
            continue
        m = regression_metrics(y_true[mask], y_pred[mask])
        rows.append({
            "subgroup_col": subgroup_col,
            "subgroup": grp,
            "n": int(np.sum(mask)),
            "mean_actual": round(float(np.mean(y_true[mask])), 3),
            "mean_pred": round(float(np.mean(y_pred[mask])), 3),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "bias": round(float(np.mean(y_pred[mask] - y_true[mask])), 4),
        })
    return rows
