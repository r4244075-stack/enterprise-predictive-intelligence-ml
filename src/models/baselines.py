#!/usr/bin/env python3
"""Module 2: Baseline Model Implementations (Classification, Forecasting & NLP)."""
import numpy as np
from scipy import optimize

class BalancedLogisticBaseline:
    """L2-regularized Logistic Regression with optional inverse-frequency class balancing."""
    def __init__(self, C: float = 1.0, balanced: bool = True):
        self.C = C
        self.balanced = balanced
        self.theta_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        if self.balanced:
            n1 = max(np.sum(y == 1), 1)
            n0 = max(np.sum(y == 0), 1)
            weights = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
        else:
            weights = np.ones(n)
        reg = 1.0 / (self.C * n)

        def loss_grad(theta):
            z = np.clip(Xb @ theta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            loss = -np.mean(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            loss += 0.5 * reg * np.sum(theta[1:] ** 2)
            grad = (Xb.T @ (weights * (p - y))) / n
            grad[1:] += reg * theta[1:]
            return loss, grad

        res = optimize.minimize(loss_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B")
        self.theta_ = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return 1.0 / (1.0 + np.exp(-np.clip(Xb @ self.theta_, -30, 30)))

class RidgeForecastBaseline:
    """L2-regularized Ridge Regression for panel demand forecasting."""
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.beta_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        reg = self.alpha * np.eye(Xb.shape[1])
        reg[0, 0] = 0.0
        self.beta_ = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.maximum(Xb @ self.beta_, 0.0)
