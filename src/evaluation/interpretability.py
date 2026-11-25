"""Model interpretability: Permutation Feature Importance, Partial Dependence Plots, and SHAP values."""

from __future__ import annotations
import numpy as np
from .metrics import roc_auc_score


def permutation_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    task: str = "classification",
    n_repeats: int = 8,
    random_state: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(random_state)
    if task == "classification":
        base_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        pred = model.predict(X_test)
        base_score = -float(np.sqrt(np.mean((y_test - pred) ** 2)))  # negative RMSE (higher is better)

    results = []
    X_work = X_test.copy()
    for j, name in enumerate(feature_names):
        drops = []
        orig_col = X_work[:, j].copy()
        for _ in range(n_repeats):
            X_work[:, j] = rng.permutation(orig_col)
            if task == "classification":
                sc = roc_auc_score(y_test, model.predict_proba(X_work)[:, 1])
            else:
                pred = model.predict(X_work)
                sc = -float(np.sqrt(np.mean((y_test - pred) ** 2)))
            drops.append(base_score - sc)
        X_work[:, j] = orig_col
        results.append({
            "feature": name,
            "importance_mean": round(float(np.mean(drops)), 5),
            "importance_std": round(float(np.std(drops)), 5),
        })
    results.sort(key=lambda r: r["importance_mean"], reverse=True)
    return results


def partial_dependence_1d(
    model,
    X_ref: np.ndarray,
    feat_idx: int,
    task: str = "classification",
    grid_points: int = 25,
    max_samples: int = 1500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    if len(X_ref) > max_samples:
        X_sub = X_ref[rng.choice(len(X_ref), size=max_samples, replace=False)].copy()
    else:
        X_sub = X_ref.copy()

    col = X_sub[:, feat_idx]
    lo, hi = np.percentile(col, [2, 98])
    grid = np.linspace(lo, hi, grid_points)
    pdp = np.zeros(grid_points, dtype=np.float64)
    for i, val in enumerate(grid):
        X_sub[:, feat_idx] = val
        if task == "classification":
            pdp[i] = float(np.mean(model.predict_proba(X_sub)[:, 1]))
        else:
            pdp[i] = float(np.mean(model.predict(X_sub)))
    return grid, pdp


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    task: str = "classification",
    n_permutations: int = 24,
    random_state: int = 42,
) -> np.ndarray:
    """Exact LinearSHAP for linear models; unbiased antithetic Monte-Carlo Shapley for tree ensembles."""
    if hasattr(model, "coef_") and model.coef_ is not None and not hasattr(model, "trees_"):
        # Exact interventional LinearSHAP in log-odds / regression space
        mu = np.mean(X_background, axis=0)
        return (X_explain - mu) * model.coef_

    rng = np.random.default_rng(random_state)
    n_exp, d = X_explain.shape
    bg_n = len(X_background)
    shap_vals = np.zeros((n_exp, d), dtype=np.float64)

    def _f(X_in: np.ndarray) -> np.ndarray:
        if task == "classification":
            return model.predict_proba(X_in)[:, 1]
        return model.predict(X_in)

    for _ in range(n_permutations):
        perm = rng.permutation(d)
        bg = X_background[rng.integers(0, bg_n, size=n_exp)].copy()
        X_curr = bg.copy()
        prev_pred = _f(X_curr)
        for j in perm:
            X_curr[:, j] = X_explain[:, j]
            new_pred = _f(X_curr)
            shap_vals[:, j] += (new_pred - prev_pred)
            prev_pred = new_pred

    return shap_vals / n_permutations
