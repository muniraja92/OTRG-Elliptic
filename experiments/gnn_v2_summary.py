"""Test PR-AUC/F1 of the GNN baselines, run ONCE after all npz files are frozen.
Platt scaling + F1-optimal threshold fitted on OOT block 30-34 (src.evaluation), as for every method."""
import sys, os, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import metrics, best_f1_threshold, Platt
from sklearn.metrics import average_precision_score
R = f"{ROOT}/results"; meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34)
rows = []
for name in ["GCN", "GraphSAGE", "Mowst", "GAT"]:
    for f in sorted(glob.glob(f"{R}/preds/{name}_seed*.npz")):
        d = np.load(f); po, pt = d["oot"][cal], d["test"][test]
        pl = Platt().fit(po, yi[cal]); thr = best_f1_threshold(yi[cal], pl.transform(po))
        m = metrics(yi[test], pl.transform(pt), thr, t[test])
        rows.append(dict(model=name, seed=int(f.split("_seed")[1][:-4]), val_pr_auc=average_precision_score(yi[cal], po), **m))
det = pd.DataFrame(rows); det.to_csv(f"{R}/gnn_v2_per_seed.csv", index=False)
cols = ["val_pr_auc", "pr_auc", "f1", "precision", "recall", "f1_pre43", "f1_post43", "roc_auc"]
s = det.groupby("model")[cols].agg(["mean", "std"]); s.columns = [f"{a}_{b}" for a, b in s.columns]
s["n_seeds"] = det.groupby("model").size(); s["seeds"] = det.groupby("model").seed.apply(lambda v: " ".join(map(str, sorted(v))))
s.to_csv(f"{R}/gnn_v2_summary.csv"); print(s.round(4).T.to_string())
