"""Evaluate second-dataset predictions: Platt + F1-optimal threshold on dev for every method,
test metrics per split seed, mean+-sd, split-seed paired differences, stratified bootstrap on split 42.
Usage: python3 -I evaluate_second.py yelp amazon
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from src.evaluation import metrics, best_f1_threshold, Platt

R = f"{ROOT}/results/second_dataset"; SEEDS = [42, 1, 7, 123, 2024]
METHODS = ["LGBM-node", "LGBM-G", "LGBM-Grew", "OTRG", "A-constant-gate", "A-average", "Stack-LR",
           "A-structural-only-gate", "A-rewired-gate"]
if os.path.exists(f"{R}/preds/{sys.argv[1]}_LGBM-deg_split42.npz"): METHODS.append("LGBM-deg")  # supplementary control
METHODS += ["LGBM-Grew-onesided", "A-rewired-gate-onesided"]   # discarded first rewiring (index leakage), reported for transparency
COMP = [("LGBM-G", "LGBM-node"), ("LGBM-G", "LGBM-Grew"), ("LGBM-Grew", "LGBM-node"),
        ("OTRG", "A-constant-gate"), ("OTRG", "A-average"), ("OTRG", "Stack-LR"), ("OTRG", "LGBM-node"),
        ("OTRG", "LGBM-G"), ("OTRG", "A-structural-only-gate"), ("A-rewired-gate", "LGBM-node")]
if "LGBM-deg" in METHODS: COMP += [("LGBM-deg", "LGBM-node"), ("LGBM-G", "LGBM-deg"), ("LGBM-Grew", "LGBM-deg")]
COMP += [("LGBM-Grew-onesided", "LGBM-Grew")]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


rows, srows, boot_store = [], [], {}
for ds in sys.argv[1:]:
    for s in SEEDS:
        I = np.load(f"{R}/preds/{ds}_split{s}_idx.npz"); y, dv, te = I["y"], I["dev"], I["test"]
        for mth in METHODS:
            d = np.load(f"{R}/preds/{ds}_{mth}_split{s}.npz"); pdv, pte = d["dev"][dv], d["test"][te]
            pl = Platt().fit(pdv, y[dv]); cd, ct = pl.transform(pdv), pl.transform(pte)
            thr = best_f1_threshold(y[dv], cd); m = metrics(y[te], ct, thr)
            rows.append(dict(dataset=ds, split_seed=s, method=mth, **m))
            if s == 42: boot_store[(ds, mth)] = (ct, thr)
        if s == 42: boot_store[(ds, "_y")] = y[te]
M = pd.DataFrame(rows); M.to_csv(f"{R}/metrics_per_split.csv", index=False)
keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "ece"]
summ = M.groupby(["dataset", "method"])[keys].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
summ["order"] = summ.method.map({m: i for i, m in enumerate(METHODS)}); summ = summ.sort_values(["dataset", "order"]).drop(columns="order")
summ.to_csv(f"{R}/summary.csv", index=False, float_format="%.5f")

B = 1000; rng0 = 12345
for ds in sys.argv[1:]:
    y = boot_store[(ds, "_y")]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(rng0)
    idxs = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]
    for met in ["pr_auc", "roc_auc", "f1"]:
        part = []
        for a, b in COMP:
            xa = M[(M.dataset == ds) & (M.method == a)].sort_values("split_seed")[met].values
            xb = M[(M.dataset == ds) & (M.method == b)].sort_values("split_seed")[met].values
            dlt = xa - xb
            t_p = stats.ttest_rel(xa, xb).pvalue if np.any(dlt != 0) else 1.0
            w_p = stats.wilcoxon(xa, xb).pvalue if np.any(dlt != 0) else 1.0
            (pa, ta), (pb, tb) = boot_store[(ds, a)], boot_store[(ds, b)]
            def f(p, t, ix):
                if met == "pr_auc": return average_precision_score(y[ix], p[ix])
                if met == "roc_auc": return roc_auc_score(y[ix], p[ix])
                return f1_score(y[ix], p[ix] >= t, zero_division=0)
            full = np.arange(len(y)); d0 = f(pa, ta, full) - f(pb, tb, full)
            bd = np.array([f(pa, ta, ix) - f(pb, tb, ix) for ix in idxs])
            bp = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))
            part.append(dict(dataset=ds, metric=met, a=a, b=b, seed_mean_diff=dlt.mean(), seed_sd_diff=dlt.std(ddof=1),
                             seeds_a_better=int((dlt > 0).sum()), paired_t_p=t_p, wilcoxon_p=w_p,
                             boot42_diff=d0, boot42_lo=np.quantile(bd, 0.025), boot42_hi=np.quantile(bd, 0.975), boot42_p=bp))
        adj_t = holm([r["paired_t_p"] for r in part]); adj_b = holm([r["boot42_p"] for r in part])
        for r, at, ab in zip(part, adj_t, adj_b): r["paired_t_p_holm"] = at; r["boot42_p_holm"] = ab
        srows += part
pd.DataFrame(srows).to_csv(f"{R}/stats.csv", index=False, float_format="%.5g")
print("wrote summary.csv, stats.csv, metrics_per_split.csv")
