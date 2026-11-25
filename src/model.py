"""Expert and baseline learners (feature-level models)."""
from __future__ import annotations
import numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


def lgbm(seed, num_leaves=31, min_child_samples=20, n_estimators=400):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=num_leaves,
                              min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, reg_lambda=1.0, random_state=seed, n_jobs=2,
                              verbose=-1, deterministic=True, force_col_wise=True)


def tune_lgbm(Xtr, ytr, Xva, yva, seed=42):
    """Small grid on a temporal validation block; returns params incl. early-stopped size."""
    best = None
    for nl in (15, 31, 63):
        for mcs in (20, 50):
            m = lgbm(seed, nl, mcs, 2000)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            n = max(50, int(m.best_iteration_ or 2000))
            ap = average_precision_score(yva, m.predict_proba(Xva, num_iteration=n)[:, 1])
            if best is None or ap > best[0]:
                best = (ap, dict(num_leaves=nl, min_child_samples=mcs, n_estimators=n))
    return best[1], best[0]


def make(name, seed, params=None):
    params = params or {}
    if name.startswith("LGBM"):
        return lgbm(seed, **params)
    if name == "RF-Weber":   # configuration reported by Weber et al. (2019)
        return RandomForestClassifier(n_estimators=50, max_features=50, random_state=seed, n_jobs=2)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_features=params.get("max_features", "sqrt"),
                                      min_samples_leaf=1, random_state=seed, n_jobs=2)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=params.get("n_estimators", 400), learning_rate=0.05,
                                 max_depth=params.get("max_depth", 6), subsample=0.8, colsample_bytree=0.5,
                                 tree_method="hist", random_state=seed, n_jobs=2, eval_metric="aucpr")
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    raise KeyError(name)
