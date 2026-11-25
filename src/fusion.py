"""Reliability-gated fusion (the proposed component).

p_i = a_i * p_node_i + (1 - a_i) * p_nbr_i,   a_i = sigmoid(w0 + w^T r_i)

r_i are label-free reliability signals (expert disagreement, expert entropies, degrees,
neighbourhood dispersion, feature-space novelty). The gate is fitted by penalised
log-likelihood on out-of-time (forward cross-fitted) expert predictions only, so it learns
which view transfers forward in time rather than which view fits the training period.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def _entropy(p):
    p = np.clip(p, EPS, 1 - EPS); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def reliability_features(p_node, p_nbr, side):
    """side: array with columns [log_deg_in, log_deg_out, nbr_dispersion, novelty]."""
    return np.column_stack([
        np.abs(_logit(p_node) - _logit(p_nbr)),   # disagreement
        _entropy(p_node), _entropy(p_nbr), side])


class ReliabilityGate:
    def __init__(self, l2=1e-2, use_features=True, class_weight=None):
        self.l2, self.use_features, self.cw = l2, use_features, class_weight

    def _design(self, R):
        if not self.use_features:
            return np.ones((len(R), 1))
        Z = (R - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, p_node, p_nbr, R, y):
        self.mu_, self.sd_ = R.mean(0), R.std(0) + 1e-9
        D = self._design(R)
        w_s = np.ones_like(y, dtype=float)
        if self.cw == "balanced":
            pos = y.mean(); w_s = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))

        def obj(w):
            a = expit(D @ w)
            p = np.clip(a * p_node + (1 - a) * p_nbr, EPS, 1 - EPS)
            ll = w_s * (y * np.log(p) + (1 - y) * np.log(1 - p))
            # gradient: dL/da = w_s*(y/p - (1-y)/(1-p)) * (p_node - p_nbr)
            g_p = w_s * (y / p - (1 - y) / (1 - p))
            g_w = D.T @ (g_p * (p_node - p_nbr) * a * (1 - a))
            reg = self.l2 * np.sum(w[1:] ** 2)
            greg = np.r_[0.0, 2 * self.l2 * w[1:]]
            n = len(y)
            return -ll.sum() / n + reg, -g_w / n + greg

        w0 = np.zeros(D.shape[1])
        self.res_ = minimize(obj, w0, jac=True, method="L-BFGS-B")
        assert self.res_.success, self.res_.message
        self.w_ = self.res_.x
        return self

    def alpha(self, R):
        return expit(self._design(R) @ self.w_)

    def predict(self, p_node, p_nbr, R):
        a = self.alpha(R); return a * p_node + (1 - a) * p_nbr, a
