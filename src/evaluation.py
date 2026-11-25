"""Metrics, calibration, thresholds."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score, average_precision_score,
                             roc_auc_score, matthews_corrcoef, brier_score_loss)
from sklearn.linear_model import LogisticRegression


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 400)))
    f = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def metrics(y, p, thr, t=None):
    yh = (p >= thr).astype(int)
    out = dict(f1=f1_score(y, yh, zero_division=0), precision=precision_score(y, yh, zero_division=0),
               recall=recall_score(y, yh, zero_division=0), mcc=matthews_corrcoef(y, yh),
               pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
               brier=brier_score_loss(y, p), ece=ece(y, p), threshold=thr)
    if t is not None:
        pre, post = t < 43, t >= 43
        out["f1_pre43"] = f1_score(y[pre], yh[pre], zero_division=0)
        out["f1_post43"] = f1_score(y[post], yh[post], zero_division=0)
    return out


class Platt:
    """Platt scaling on the logit, fitted on a disjoint calibration block."""
    def fit(self, p, y):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        self.lr = LogisticRegression(C=1e4).fit(z[:, None], y); return self

    def transform(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        return self.lr.predict_proba(z[:, None])[:, 1]


def per_step_f1(y, p, thr, t):
    return {int(s): f1_score(y[t == s], p[t == s] >= thr, zero_division=0) for s in np.unique(t)}
