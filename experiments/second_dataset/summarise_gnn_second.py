"""Summarise GNN baselines (run_gnn_second.py): mean +- sd over split seeds and per-split paired differences
versus LGBM-G and OTRG taken from results/second_dataset/metrics_per_split.csv. Light work, no lock needed.
Usage: python3 -I summarise_gnn_second.py
"""
import os
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); R = f"{ROOT}/results/second_dataset"
G = pd.read_csv(f"{R}/gnn_metrics_per_split.csv"); T = pd.read_csv(f"{R}/metrics_per_split.csv")
K = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "ece"]
S = G.groupby(["dataset", "method"])[K].agg(["mean", "std"]); S.columns = [f"{a}_{b}" for a, b in S.columns]
S.reset_index().to_csv(f"{R}/gnn_summary.csv", index=False, float_format="%.5f")
rows, per = [], []
for (ds, gm), g in G.groupby(["dataset", "method"]):
    g = g.set_index("split_seed").sort_index()
    for ref in ["LGBM-G", "OTRG"]:
        t = T[(T.dataset == ds) & (T.method == ref)].set_index("split_seed").sort_index()
        assert list(t.index) == list(g.index)
        for met in ["pr_auc", "roc_auc", "f1"]:
            d = g[met].values - t[met].values
            for s, v in zip(g.index, d): per.append(dict(dataset=ds, gnn=gm, ref=ref, metric=met, split_seed=s, diff=v))
            rows.append(dict(dataset=ds, gnn=gm, ref=ref, metric=met, gnn_mean=g[met].mean(), ref_mean=t[met].mean(),
                             mean_diff=d.mean(), sd_diff=d.std(ddof=1), min_diff=d.min(), max_diff=d.max(),
                             splits_gnn_better=int((d > 0).sum()), splits_ref_better=int((d < 0).sum()),
                             paired_t_p=stats.ttest_rel(g[met].values, t[met].values).pvalue))
P = pd.DataFrame(rows)
for (ds, met), ix in P.groupby(["dataset", "metric"]).groups.items():   # Holm within dataset x metric (4 comparisons)
    p = P.loc[ix, "paired_t_p"].values; o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    P.loc[ix, "paired_t_p_holm"] = adj
P.to_csv(f"{R}/gnn_paired.csv", index=False, float_format="%.5g")
pd.DataFrame(per).to_csv(f"{R}/gnn_paired_per_split.csv", index=False, float_format="%.5f")
print(S.round(4).to_string()); print(P.round(4).to_string())
