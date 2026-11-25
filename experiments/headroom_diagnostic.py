"""Headroom analysis: splits the Elliptic test period at the dark-market closure and reports, per
period, ranking quality (ROC-AUC, PR-AUC), illicit prevalence, F1 at the validation-selected
threshold and an ORACLE F1 (best threshold chosen on the test labels of that period). The oracle
is a diagnostic upper bound for any re-thresholding or prior-shift correction; it is never used
for model selection."""
import os, sys, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from src.evaluation import Platt, best_f1_threshold
R = f"{ROOT}/results"; m = np.load(f"{R}/preds/_meta.npz"); y = np.nan_to_num(m["y"]).astype(int); t = m["t"]; test = m["test"]
cal = (~np.isnan(m["y"])) & (t >= 30) & (t <= 34); rows = []
for meth in ["TRUST-GRAF", "LGBM-AF", "LGBM-AF+G", "XGB", "RF"]:
    for f in sorted(glob.glob(f"{R}/preds/{meth}_seed*.npz")):
        d = np.load(f); pl = Platt().fit(d["oot"][cal], y[cal]); thr = best_f1_threshold(y[cal], pl.transform(d["oot"][cal]))
        for per, (a, b) in {"pre": (35, 42), "post": (43, 49)}.items():
            mk = test & (t >= a) & (t <= b); yy = y[mk]; qq = pl.transform(d["test"][mk])
            pr, rc, _ = precision_recall_curve(yy, qq)
            rows.append(dict(method=meth, seed=int(f.split("_seed")[1][:-4]), period=per, roc_auc=roc_auc_score(yy, qq), pr_auc=average_precision_score(yy, qq),
                             prevalence=yy.mean(), f1_val_threshold=f1_score(yy, qq >= thr), f1_oracle=float(np.max(2 * pr * rc / np.maximum(pr + rc, 1e-12)))))
df = pd.DataFrame(rows); df.to_csv(f"{R}/headroom_per_seed.csv", index=False)
s = df.groupby(["method", "period"])[["roc_auc", "pr_auc", "prevalence", "f1_val_threshold", "f1_oracle"]].mean(); s.to_csv(f"{R}/headroom_summary.csv"); print(s.round(4).to_string())
