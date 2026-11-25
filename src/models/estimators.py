"""Core vectorized NumPy/SciPy machine learning estimators.

Self-contained, deterministic, reproducible implementations of:
- LogisticRegressionL2 (L-BFGS-B with exact gradient, L2 penalty, class weights)
- CalibratedLinearSVM (L2 squared-hinge SVM + Platt sigmoid calibration)
- HistDecisionTree (vectorized quantile-binned CART for classification & regression)
- HistRandomForest (bagged feature-subsampled ensemble of HistDecisionTree)
- HistGradientBoosting (Newton-Raphson second-order histogram GBDT, XGBoost/LightGBM style)
- RidgeRegression / LassoRegression (exact closed-form & coordinate descent)
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-7, 1.0 - 1e-7)


class LogisticRegressionL2:
    """L2-regularized binary logistic regression optimized with L-BFGS-B."""

    def __init__(self, C: float = 1.0, class_weight: str | None = None, max_iter: int = 300, random_state: int = 42):
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight == "balanced":
            n = len(y)
            n_pos = max(np.sum(y == 1), 1)
            n_neg = max(np.sum(y == 0), 1)
            w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
            return w
        return np.ones_like(y, dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionL2":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        sw = self._sample_weights(y)
        sw_sum = np.sum(sw)
        reg = 1.0 / (self.C * sw_sum)

        def loss_and_grad(params: np.ndarray):
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            p = expit(z)
            # numerically stable weighted log-loss
            loss = np.sum(sw * (np.logaddexp(0.0, z) - y * z)) / sw_sum + 0.5 * reg * np.dot(w, w)
            diff = sw * (p - y) / sw_sum
            gw = X.T @ diff + reg * w
            gb = np.sum(diff)
            return loss, np.append(gw, gb)

        p0 = np.zeros(d + 1, dtype=np.float64)
        p_pos = np.clip(np.average(y, weights=sw), 1e-5, 1.0 - 1e-5)
        p0[-1] = np.log(p_pos / (1.0 - p_pos))
        res = minimize(loss_and_grad, p0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x[:-1].copy()
        self.intercept_ = float(res.x[-1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class CalibratedLinearSVM:
    """L2-regularized squared-hinge Linear SVM with Platt sigmoid probability calibration."""

    def __init__(self, C: float = 1.0, class_weight: str | None = None, max_iter: int = 300, random_state: int = 42):
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.platt_a_: float = 1.0
        self.platt_b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedLinearSVM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_signed = np.where(y == 1, 1.0, -1.0)
        n, d = X.shape
        if self.class_weight == "balanced":
            n_pos = max(np.sum(y == 1), 1)
            n_neg = max(np.sum(y == 0), 1)
            sw = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
        else:
            sw = np.ones(n, dtype=np.float64)
        sw_sum = np.sum(sw)
        reg = 1.0 / (self.C * sw_sum)

        def loss_and_grad(params: np.ndarray):
            w = params[:-1]
            b = params[-1]
            margin = 1.0 - y_signed * (X @ w + b)
            hinge = np.maximum(0.0, margin)
            loss = np.sum(sw * (hinge ** 2)) / sw_sum + 0.5 * reg * np.dot(w, w)
            coeff = -2.0 * sw * hinge * y_signed / sw_sum
            gw = X.T @ coeff + reg * w
            gb = np.sum(coeff)
            return loss, np.append(gw, gb)

        res = minimize(loss_and_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x[:-1].copy()
        self.intercept_ = float(res.x[-1])

        # Platt scaling on decision scores
        scores = X @ self.coef_ + self.intercept_

        def platt_loss(ab: np.ndarray):
            z = ab[0] * scores + ab[1]
            p = expit(z)
            l = np.mean(np.logaddexp(0.0, z) - y * z) + 1e-4 * (ab[0] ** 2 + ab[1] ** 2)
            diff = (p - y) / n
            return l, np.array([np.dot(diff, scores) + 2e-4 * ab[0], np.sum(diff) + 2e-4 * ab[1]])

        pres = minimize(platt_loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        self.platt_a_, self.platt_b_ = float(pres.x[0]), float(pres.x[1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(self.platt_a_ * self.decision_function(X) + self.platt_b_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class _BinMapper:
    """Fast quantile bin mapper (max_bins <= 64) fit strictly on training features."""

    def __init__(self, max_bins: int = 32):
        self.max_bins = max_bins
        self.edges_: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "_BinMapper":
        X = np.asarray(X, dtype=np.float64)
        self.edges_ = []
        qs = np.linspace(0, 100, self.max_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= self.max_bins:
                edges = 0.5 * (uniq[:-1] + uniq[1:]) if len(uniq) > 1 else np.array([], dtype=np.float64)
            else:
                edges = np.unique(np.percentile(col, qs))
            self.edges_.append(edges)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        B = np.zeros(X.shape, dtype=np.int32)
        for j, edges in enumerate(self.edges_):
            if len(edges) > 0:
                B[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return B


class _HistTree:
    """Vectorized histogram decision tree supporting Newton-Raphson (GBDT) or CART (DT/RF)."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        l2_lambda: float = 1.0,
        min_gain: float = 1e-6,
        max_features: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_lambda = l2_lambda
        self.min_gain = min_gain
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(42)
        # Flat node arrays for fast vectorized prediction
        self.feature: list[int] = []
        self.threshold_bin: list[int] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.value: list[float] = []
        self.feature_gains_: np.ndarray | None = None

    def fit_grad_hess(self, B: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins_per_feat: np.ndarray, idx: np.ndarray):
        n_features = B.shape[1]
        self.feature_gains_ = np.zeros(n_features, dtype=np.float64)
        self._build_node(B, g, h, n_bins_per_feat, idx, depth=0)
        return self

    def _add_leaf(self, val: float) -> int:
        nid = len(self.value)
        self.feature.append(-1)
        self.threshold_bin.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(val)
        return nid

    def _build_node(self, B: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins_per_feat: np.ndarray, idx: np.ndarray, depth: int) -> int:
        G_tot = float(np.sum(g[idx]))
        H_tot = float(np.sum(h[idx]))
        leaf_val = -G_tot / (H_tot + self.l2_lambda)

        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return self._add_leaf(leaf_val)

        n_features = B.shape[1]
        if self.max_features is not None and self.max_features < n_features:
            feat_candidates = self.rng.choice(n_features, size=self.max_features, replace=False)
        else:
            feat_candidates = range(n_features)

        parent_score = (G_tot * G_tot) / (H_tot + self.l2_lambda)
        best_gain = self.min_gain
        best_feat = -1
        best_bin = -1

        g_sub = g[idx]
        h_sub = h[idx]

        for j in feat_candidates:
            nb = int(n_bins_per_feat[j])
            if nb <= 1:
                continue
            b_col = B[idx, j]
            cnt_hist = np.bincount(b_col, minlength=nb)
            g_hist = np.bincount(b_col, weights=g_sub, minlength=nb)
            h_hist = np.bincount(b_col, weights=h_sub, minlength=nb)

            cnt_L = np.cumsum(cnt_hist)[:-1]
            cnt_R = len(idx) - cnt_L
            valid = (cnt_L >= self.min_samples_leaf) & (cnt_R >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            G_L = np.cumsum(g_hist)[:-1]
            H_L = np.cumsum(h_hist)[:-1]
            G_R = G_tot - G_L
            H_R = H_tot - H_L

            gains = 0.5 * (
                (G_L * G_L) / (H_L + self.l2_lambda)
                + (G_R * G_R) / (H_R + self.l2_lambda)
                - parent_score
            )
            gains = np.where(valid, gains, -np.inf)
            b_idx = int(np.argmax(gains))
            if gains[b_idx] > best_gain:
                best_gain = float(gains[b_idx])
                best_feat = int(j)
                best_bin = b_idx

        if best_feat == -1:
            return self._add_leaf(leaf_val)

        self.feature_gains_[best_feat] += best_gain
        nid = len(self.value)
        self.feature.append(best_feat)
        self.threshold_bin.append(best_bin)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(leaf_val)

        mask_L = B[idx, best_feat] <= best_bin
        left_id = self._build_node(B, g, h, n_bins_per_feat, idx[mask_L], depth + 1)
        right_id = self._build_node(B, g, h, n_bins_per_feat, idx[~mask_L], depth + 1)
        self.left[nid] = left_id
        self.right[nid] = right_id
        return nid

    def predict_binned(self, B: np.ndarray) -> np.ndarray:
        n = B.shape[0]
        nodes = np.zeros(n, dtype=np.int32)
        feat = np.array(self.feature, dtype=np.int32)
        thr = np.array(self.threshold_bin, dtype=np.int32)
        left = np.array(self.left, dtype=np.int32)
        right = np.array(self.right, dtype=np.int32)
        val = np.array(self.value, dtype=np.float64)

        while True:
            f = feat[nodes]
            active = f >= 0
            if not np.any(active):
                break
            act_idx = np.nonzero(active)[0]
            act_nodes = nodes[act_idx]
            go_left = B[act_idx, f[act_idx]] <= thr[act_nodes]
            nodes[act_idx] = np.where(go_left, left[act_nodes], right[act_nodes])
        return val[nodes]


class HistDecisionTreeClassifier:
    """Single histogram CART decision tree for binary classification (probability output)."""

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 25, max_bins: int = 32, class_weight: str | None = None, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.tree_: _HistTree | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistDecisionTreeClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)
        if self.class_weight == "balanced":
            n = len(y)
            sw = np.where(y == 1, n / (2.0 * max(np.sum(y == 1), 1)), n / (2.0 * max(np.sum(y == 0), 1)))
        else:
            sw = np.ones_like(y)
        # For CART MSE on 0/1 labels (equivalent to Gini impurity), g = -y*sw, h = sw -> leaf = -G/H = weighted mean(y)
        self.tree_ = _HistTree(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_lambda=1e-6,
            rng=np.random.default_rng(self.random_state),
        ).fit_grad_hess(B, -y * sw, sw, n_bins, np.arange(len(y)))
        gains = self.tree_.feature_gains_
        self.feature_importances_ = gains / max(np.sum(gains), 1e-12)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        p1 = np.clip(self.tree_.predict_binned(B), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class HistRandomForestClassifier:
    """Bagged, feature-subsampled Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 8,
        min_samples_leaf: int = 15,
        max_features: str | int = "sqrt",
        max_bins: int = 32,
        class_weight: str | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.trees_: list[_HistTree] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistRandomForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)
        if self.class_weight == "balanced":
            sw = np.where(y == 1, n / (2.0 * max(np.sum(y == 1), 1)), n / (2.0 * max(np.sum(y == 0), 1)))
        else:
            sw = np.ones(n, dtype=np.float64)

        mf = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else (d if self.max_features is None else int(self.max_features))
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        total_gains = np.zeros(d, dtype=np.float64)
        g = -y * sw
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n)
            t = _HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_lambda=1e-6,
                max_features=mf,
                rng=rng,
            ).fit_grad_hess(B, g, sw, n_bins, boot_idx)
            self.trees_.append(t)
            total_gains += t.feature_gains_
        self.feature_importances_ = total_gains / max(np.sum(total_gains), 1e-12)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        preds = np.mean([t.predict_binned(B) for t in self.trees_], axis=0)
        p1 = np.clip(preds, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class HistGradientBoostingClassifier:
    """Second-order Newton-Raphson Histogram GBDT (XGBoost/LightGBM style) for binary classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        min_samples_leaf: int = 25,
        l2_lambda: float = 2.0,
        subsample: float = 0.85,
        colsample: float = 0.85,
        max_bins: int = 32,
        class_weight: str | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_lambda = l2_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.trees_: list[_HistTree] = []
        self.init_score_: float = 0.0
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)

        if self.class_weight == "balanced":
            sw = np.where(y == 1, n / (2.0 * max(np.sum(y == 1), 1)), n / (2.0 * max(np.sum(y == 0), 1)))
        else:
            sw = np.ones(n, dtype=np.float64)

        p0 = np.clip(np.average(y, weights=sw), 1e-5, 1.0 - 1e-5)
        self.init_score_ = float(np.log(p0 / (1.0 - p0)))
        F = np.full(n, self.init_score_, dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        sub_n = max(1, int(n * self.subsample))
        mf = max(1, int(np.ceil(d * self.colsample)))
        self.trees_ = []
        total_gains = np.zeros(d, dtype=np.float64)

        for _ in range(self.n_estimators):
            p = expit(F)
            g = sw * (p - y)
            h = sw * np.maximum(p * (1.0 - p), 1e-6)
            idx = rng.choice(n, size=sub_n, replace=False) if self.subsample < 1.0 else np.arange(n)
            t = _HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_lambda=self.l2_lambda,
                max_features=mf,
                rng=rng,
            ).fit_grad_hess(B, g, h, n_bins, idx)
            F += self.learning_rate * t.predict_binned(B)
            self.trees_.append(t)
            total_gains += t.feature_gains_

        self.feature_importances_ = total_gains / max(np.sum(total_gains), 1e-12)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        F = np.full(B.shape[0], self.init_score_, dtype=np.float64)
        for t in self.trees_:
            F += self.learning_rate * t.predict_binned(B)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ===================== REGRESSION ESTIMATORS =====================

class RidgeRegressor:
    """Closed-form L2-regularized linear regression."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xm = X.mean(axis=0)
        ym = y.mean()
        Xc = X - xm
        yc = y - ym
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = float(ym - xm @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class HistDecisionTreeRegressor:
    """Single histogram CART decision tree regressor."""

    def __init__(self, max_depth: int = 7, min_samples_leaf: int = 25, max_bins: int = 32, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.tree_: _HistTree | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistDecisionTreeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)
        self.tree_ = _HistTree(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_lambda=1e-6,
            rng=np.random.default_rng(self.random_state),
        ).fit_grad_hess(B, -y, np.ones_like(y), n_bins, np.arange(len(y)))
        gains = self.tree_.feature_gains_
        self.feature_importances_ = gains / max(np.sum(gains), 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        return self.tree_.predict_binned(B)


class HistRandomForestRegressor:
    """Bagged, feature-subsampled Random Forest regressor."""

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 9,
        min_samples_leaf: int = 15,
        max_features: str | int = "sqrt",
        max_bins: int = 32,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.trees_: list[_HistTree] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistRandomForestRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)
        mf = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else (d if self.max_features is None else int(self.max_features))
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        total_gains = np.zeros(d, dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n)
            t = _HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_lambda=1e-6,
                max_features=mf,
                rng=rng,
            ).fit_grad_hess(B, -y, ones, n_bins, boot_idx)
            self.trees_.append(t)
            total_gains += t.feature_gains_
        self.feature_importances_ = total_gains / max(np.sum(total_gains), 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        return np.mean([t.predict_binned(B) for t in self.trees_], axis=0)


class HistGradientBoostingRegressor:
    """Histogram Gradient Boosted Decision Trees for MSE regression."""

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.07,
        max_depth: int = 4,
        min_samples_leaf: int = 25,
        l2_lambda: float = 2.0,
        subsample: float = 0.85,
        colsample: float = 0.85,
        max_bins: int = 32,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_lambda = l2_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.random_state = random_state
        self.mapper_: _BinMapper | None = None
        self.trees_: list[_HistTree] = []
        self.init_score_: float = 0.0
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.mapper_ = _BinMapper(self.max_bins).fit(X)
        B = self.mapper_.transform(X)
        n_bins = np.array([len(e) + 1 for e in self.mapper_.edges_], dtype=np.int32)
        self.init_score_ = float(np.mean(y))
        F = np.full(n, self.init_score_, dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        sub_n = max(1, int(n * self.subsample))
        mf = max(1, int(np.ceil(d * self.colsample)))
        self.trees_ = []
        total_gains = np.zeros(d, dtype=np.float64)

        for _ in range(self.n_estimators):
            g = F - y
            idx = rng.choice(n, size=sub_n, replace=False) if self.subsample < 1.0 else np.arange(n)
            t = _HistTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_lambda=self.l2_lambda,
                max_features=mf,
                rng=rng,
            ).fit_grad_hess(B, g, ones, n_bins, idx)
            F += self.learning_rate * t.predict_binned(B)
            self.trees_.append(t)
            total_gains += t.feature_gains_

        self.feature_importances_ = total_gains / max(np.sum(total_gains), 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        F = np.full(B.shape[0], self.init_score_, dtype=np.float64)
        for t in self.trees_:
            F += self.learning_rate * t.predict_binned(B)
        return F
