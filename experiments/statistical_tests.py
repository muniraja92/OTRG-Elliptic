"""Evaluation, calibration, ablation tables and paired statistics from saved predictions.

Primary operating point (pre-registered in run_experiments.py docstring): Platt scaling and the
F1-optimal threshold are both fitted on the out-of-time block 30..34; the test set is never
used for any choice. Secondary: raw probabilities at threshold 0.5 (Weber et al. convention).
Paired inference: (i) stratified paired bootstrap over test transactions (resampled within time
step), applied to the seed-averaged metric; (ii) Wilcoxon signed-rank over the 15 test time
steps on seed-averaged per-step F1. Holm correction within each family.
"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import metrics, best_f1_threshold, Platt, per_step_f1
R = f"{ROOT}/results"; meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34)
yt, tt = yi[test], t[test]
PROPOSED = "TRUST-GRAF"
files = sorted(glob.glob(f"{R}/preds/*_seed*.npz")); methods = sorted({os.path.basename(f).split("_seed")[0] for f in files})
rows, step_rows, store = [], [], {}
for mname in methods:
    for f in sorted(glob.glob(f"{R}/preds/{mname}_seed*.npz")):
        seed = int(f.split("_seed")[1].split(".")[0]); d = np.load(f)
        po, pt = d["oot"][cal], d["test"][test]
        assert not np.isnan(po).any() and not np.isnan(pt).any(), f
        pl = Platt().fit(po, yi[cal]); pc_o, pc_t = pl.transform(po), pl.transform(pt)
        thr = best_f1_threshold(yi[cal], pc_o)
        m = metrics(yt, pc_t, thr, tt); raw = metrics(yt, pt, 0.5, tt)
        m.update(method=mname, seed=seed, f1_raw05=raw["f1"], f1_post43_raw05=raw["f1_post43"],
                 brier_raw=raw["brier"], ece_raw=raw["ece"])
        rows.append(m); store[(mname, seed)] = (pc_t, thr)
        for s, v in per_step_f1(yt, pc_t, thr, tt).items(): step_rows.append(dict(method=mname, seed=seed, time_step=s, f1=v))
met = pd.DataFrame(rows); met.to_csv(f"{R}/metrics.csv", index=False)
steps = pd.DataFrame(step_rows); steps.to_csv(f"{R}/per_timestep_f1.csv", index=False)
cols = ["f1", "precision", "recall", "mcc", "pr_auc", "roc_auc", "brier", "ece", "f1_pre43", "f1_post43", "f1_raw05", "f1_post43_raw05", "brier_raw", "ece_raw"]
summ = met.groupby("method")[cols].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ["n_seeds"] = met.groupby("method").size(); summ.sort_values("pr_auc_mean", ascending=False).to_csv(f"{R}/metrics_summary.csv")
print(summ[["f1_mean", "f1_std", "pr_auc_mean", "pr_auc_std", "f1_post43_mean", "brier_mean", "ece_mean", "f1_raw05_mean", "n_seeds"]].sort_values("pr_auc_mean", ascending=False).round(4).to_string())

# ---------- paired statistics: proposed vs every other method ----------
rng = np.random.default_rng(12345); B = 1000
strata = [np.where(tt == s)[0] for s in np.unique(tt)]
boot_idx = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(B)]
def ap_fast(a, p):
    """Average precision identical to sklearn's step-wise definition (ties grouped)."""
    o = np.argsort(-p, kind="mergesort"); a, p = a[o], p[o]
    tp = np.cumsum(a); last = np.r_[np.diff(p) != 0, True]
    tp_l, k = tp[last], np.arange(1, len(a) + 1)[last]; prec = tp_l / k
    rec_inc = np.diff(np.r_[0, tp_l]) / max(1, a.sum()); return float((prec * rec_inc).sum())
def f1_fast(a, p, thr):
    h = p >= thr; tp = (a & h).sum(); d = a.sum() + h.sum(); return 2 * tp / d if d else 0.0
assert abs(ap_fast(yt, store[(PROPOSED, 42)][0]) - average_precision_score(yt, store[(PROPOSED, 42)][0])) < 1e-9
yb = yt.astype(bool)
FN = {"pr_auc": lambda a, p, thr: ap_fast(a, p), "f1": lambda a, p, thr: f1_fast(a, p, thr)}
cache = {}
for mname in methods:                       # bootstrap vector of the seed-averaged metric, per method
    for k, fn in FN.items():
        runs = [(p, thr) for (mm, sd), (p, thr) in store.items() if mm == mname]
        cache[(mname, k, "obs")] = np.mean([fn(yb, p, thr) for p, thr in runs])
        cache[(mname, k)] = np.array([np.mean([fn(yb[ix], p[ix], thr) for p, thr in runs]) for ix in boot_idx])
srows = []
ps = steps.groupby(["method", "time_step"]).f1.mean().unstack()
for other in methods:
    if other == PROPOSED: continue
    r = dict(comparison=f"{PROPOSED} vs {other}")
    for k, fn in FN.items():
        obs = cache[(PROPOSED, k, "obs")] - cache[(other, k, "obs")]
        d = cache[(PROPOSED, k)] - cache[(other, k)]
        r[f"delta_{k}"] = obs; r[f"ci_lo_{k}"], r[f"ci_hi_{k}"] = np.percentile(d, [2.5, 97.5])
        r[f"p_boot_{k}"] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / B)
    diff = (ps.loc[PROPOSED] - ps.loc[other]).values
    if np.allclose(diff, 0): r["p_wilcoxon_step_f1"], r["rank_biserial"] = 1.0, 0.0
    else:
        w = wilcoxon(diff, zero_method="wilcox"); nz = diff[diff != 0]
        rk = pd.Series(np.abs(nz)).rank().values; r["p_wilcoxon_step_f1"] = w.pvalue
        r["rank_biserial"] = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    r["steps_won"] = int((diff > 0).sum()); r["steps_lost"] = int((diff < 0).sum()); srows.append(r)
st = pd.DataFrame(srows)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
for c in ["p_boot_pr_auc", "p_boot_f1", "p_wilcoxon_step_f1"]: st[c + "_holm"] = holm(st[c])
st.to_csv(f"{R}/statistical_results.csv", index=False)
print(st[["comparison", "delta_pr_auc", "ci_lo_pr_auc", "ci_hi_pr_auc", "p_boot_pr_auc_holm", "delta_f1", "ci_lo_f1", "ci_hi_f1", "p_boot_f1_holm", "p_wilcoxon_step_f1_holm", "steps_won", "steps_lost"]].round(4).to_string())
abl = ["LGBM-AF", "LGBM-AF+G", "A-average", "Stack-LR", "A-static-gate", "A-structural-only-gate", "A-shuffled-graph", PROPOSED]
summ.loc[[a for a in abl if a in summ.index]].to_csv(f"{R}/ablation_results.csv")

# ---------- secondary family: does the real graph view carry information? (H2) ----------
PAIRS = [("LGBM-AF+G", "LGBM-AF"), ("LGBM-AF+G", "LGBM-AF+Gshuf"), ("LGBM-AF+Gshuf", "LGBM-AF"), ("RF", "LGBM-AF"), ("XGB", "LGBM-AF")]
if "LGBM-AF+Gdp" in methods: PAIRS += [("LGBM-AF+G", "LGBM-AF+Gdp"), ("LGBM-AF+Gdp", "LGBM-AF")]
if "GCN" in methods: PAIRS += [("GCN", "LGBM-AF"), ("GCN", "RF")]
srow2 = []
for a, b in PAIRS:
    r = dict(comparison=f"{a} vs {b}")
    for k in FN:
        d = cache[(a, k)] - cache[(b, k)]; r[f"delta_{k}"] = cache[(a, k, "obs")] - cache[(b, k, "obs")]
        r[f"ci_lo_{k}"], r[f"ci_hi_{k}"] = np.percentile(d, [2.5, 97.5]); r[f"p_boot_{k}"] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / B)
    srow2.append(r)
s2 = pd.DataFrame(srow2)
for c in ["p_boot_pr_auc", "p_boot_f1"]: s2[c + "_holm"] = holm(s2[c])
s2.to_csv(f"{R}/statistical_results_secondary.csv", index=False); print(s2.round(4).to_string())

# ---------- gate diagnostics ----------
al, ws = [], []
for f in sorted(glob.glob(f"{R}/preds/{PROPOSED}_seed*.npz")):
    d = np.load(f); al.append(d["alpha"][test]); ws.append(d["w"])
al = np.array(al); ws = np.array(ws); am = al.mean(0)
from scipy.stats import spearmanr
g = dict(GateAlphaMean=am.mean(), GateAlphaPTen=np.percentile(am, 10), GateAlphaPNinety=np.percentile(am, 90),
         GateAlphaPre=am[tt < 43].mean(), GateAlphaPost=am[tt >= 43].mean(), GateMaxAbsW=np.abs(ws[:, 1:]).max(),
         GateInterceptMean=ws[:, 0].mean(), GateRhoTime=spearmanr(am, tt).correlation)
pd.DataFrame(g, index=["value"]).to_csv(f"{R}/gate_summary.csv"); print(g)
pd.DataFrame(ws, columns=["intercept", "disagreement", "entropy_node", "entropy_graph", "log_deg_in", "log_deg_out", "dispersion", "novelty"]).to_csv(f"{R}/gate_weights.csv", index=False)

# ---------- lambda sensitivity (test-set evaluation of every grid value; not used for selection) ----------
sens = []
for f in sorted(glob.glob(f"{R}/preds_sensitivity/*.npz")):
    b = os.path.basename(f); v, rest = b.split("__l2_"); lam, seed = rest.replace(".npz", "").split("_seed")
    d = np.load(f); po, pt = d["oot"][cal], d["test"][test]; pl = Platt().fit(po, yi[cal]); thr = best_f1_threshold(yi[cal], pl.transform(po))
    m = metrics(yt, pl.transform(pt), thr, tt); sens.append(dict(variant=v, l2=float(lam), seed=int(seed), pr_auc=m["pr_auc"], f1=m["f1"], f1_post43=m["f1_post43"]))
sens = pd.DataFrame(sens).groupby(["variant", "l2"])[["pr_auc", "f1", "f1_post43"]].mean().reset_index(); sens.to_csv(f"{R}/sensitivity_summary.csv", index=False); print(sens.round(4).to_string())
