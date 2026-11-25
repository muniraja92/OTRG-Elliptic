"""Revised paired inference for every method in results/preds (statistical revision v2).

Calibration is IDENTICAL to experiments/statistical_tests.py: for every (method, seed) a Platt map
and the F1-optimal threshold are fitted on the labelled out-of-time block 30..34 (src.evaluation),
then applied once to the test period (labelled nodes of steps 35..49).

Resampling schemes (all paired: the same replicate weights are applied to every method):
  txn   : transactions resampled with replacement within time step (the v1 scheme, re-run here
          only as a reference point).
  comm  : PRIMARY. Cluster bootstrap within time step. The specification asked for weakly
          connected components of the test-period transaction graph, but in Elliptic every time
          step is ONE weakly connected component (49 components for 49 steps; verified below),
          so resampling components within step is degenerate (zero variance). We therefore use
          the finest graph-respecting partition that still lets us stratify by step: Louvain
          communities (networkx, seed 12345, resolution 1) of the undirected step graph built on
          ALL nodes (labelled and unlabelled). Communities are resampled with replacement within
          each step. This decision was fixed before any v2 p-value was computed.
  step  : Cluster bootstrap whose clusters ARE the weakly connected components, i.e. the 15 test
          time steps, resampled with replacement (no stratification possible). Conservative
          sensitivity analysis; changes the pre/post-43 mix across replicates.
  two-stage (comm x seed): each replicate also resamples the 5 seeds with replacement (paired by
          seed value when both methods have the same seed set, otherwise independently per
          method), so training-seed variance enters the interval.
p-values: two-sided CI-inversion p = min(1, 2 * (min(#{d<=0}, #{d>=0}) + 1) / (B + 1)).
B = 5000 for methods entering a pre-specified key comparison, 2000 for the others (comparisons
involving an "other" method use the first 2000 replicates; common random numbers).
Only labelled test nodes enter the metrics. AP is computed with replicate weights (exactly
sklearn's average_precision_score with sample_weight; asserted).
Outputs: results/statistical_v2_{primary,key,tost,seed,components}.csv
"""
import sys, os, glob, time, numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import average_precision_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import metrics, best_f1_threshold, Platt, per_step_f1
T0 = time.time(); R = f"{ROOT}/results"; OUT = f"{R}/statistical_v2_"
PROPOSED = "TRUST-GRAF"; B_KEY, B_REST = 5000, 2000; RNG_SEED = 20261123
KEY = [("TRUST-GRAF", "LGBM-AF"), ("TRUST-GRAF", "LGBM-AF+G"), ("TRUST-GRAF", "A-static-gate"),
       ("TRUST-GRAF", "Stack-LR"), ("TRUST-GRAF", "A-average"), ("LGBM-AF+G", "LGBM-AF"),
       ("LGBM-AF+G", "LGBM-AF+Gshuf"), ("LGBM-AF+G", "LGBM-AF+Gdp"), ("A-dp-graph-gate", "TRUST-GRAF")]
SECONDARY = [("LGBM-AF+G", "LGBM-AF"), ("LGBM-AF+G", "LGBM-AF+Gshuf"), ("LGBM-AF+Gshuf", "LGBM-AF"),
             ("LGBM-AF+G", "LGBM-AF+Gdp"), ("LGBM-AF+Gdp", "LGBM-AF"), ("RF", "LGBM-AF"), ("XGB", "LGBM-AF"),
             ("GCN", "LGBM-AF"), ("GCN", "RF"), ("GraphSAGE", "LGBM-AF"), ("Mowst", "LGBM-AF"),
             ("A-dp-graph-gate", "TRUST-GRAF")]
TOST = [("TRUST-GRAF", "A-static-gate"), ("TRUST-GRAF", "Stack-LR"), ("TRUST-GRAF", "A-average")]
MARGIN = {"pr_auc": 0.005, "f1": 0.01}
METRICS = ["pr_auc", "f1"]

# ---------------- data: labels, steps, graph ----------------
meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34)
yt, tt = yi[test], t[test]; yb = yt.astype(bool); nT = len(yt)
feat = pd.read_parquet(f"{ROOT}/data/features.parquet", columns=["txId", "time_step"])
assert (feat.time_step.values == t).all(), "node order of features.parquet differs from _meta.npz"
edg = pd.read_parquet(f"{ROOT}/data/edges.parquet")
pos = pd.Series(np.arange(len(feat)), index=feat.txId.values)
src, dst = pos.loc[edg.txId1.values].values, pos.loc[edg.txId2.values].values; del feat, edg, pos
n = len(t); A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
ncc, wcc = connected_components(A, directed=True, connection="weak")
tstep = np.arange(35, 50)
print(f"[graph] {ncc} weakly connected components over {len(np.unique(t))} steps; test-step components:",
      {int(s): len(np.unique(wcc[t == s])) for s in tstep})

import networkx as nx
comm = np.full(n, -1, dtype=np.int64); cid = 0; comp_rows = []
for s in tstep:
    nodes = np.where(t == s)[0]; m = (t[src] == s)
    G = nx.Graph(); G.add_nodes_from(nodes.tolist()); G.add_edges_from(zip(src[m].tolist(), dst[m].tolist()))
    for c in nx.community.louvain_communities(G, seed=12345, resolution=1.0):
        comm[list(c)] = cid; cid += 1
    cut = (comm[src[m]] != comm[dst[m]]).mean()
    lab = test & (t == s); cl_sizes = np.bincount(comm[nodes] - comm[nodes].min())
    lab_per = np.bincount(comm[lab] - comm[nodes].min(), minlength=len(cl_sizes)); lab_per = lab_per[lab_per > 0]
    comp_rows.append(dict(time_step=int(s), n_nodes=len(nodes), n_edges=int(m.sum()), n_wcc=len(np.unique(wcc[nodes])),
                          largest_wcc_share=1.0 if len(np.unique(wcc[nodes])) == 1 else np.bincount(wcc[nodes]).max() / len(nodes),
                          n_labelled=int(lab.sum()), n_illicit=int(yi[lab].sum()), n_louvain=len(cl_sizes),
                          n_louvain_with_labelled=len(lab_per), louvain_size_median=float(np.median(cl_sizes)),
                          louvain_size_p90=float(np.percentile(cl_sizes, 90)), louvain_size_max=int(cl_sizes.max()),
                          labelled_per_cluster_median=float(np.median(lab_per)), labelled_per_cluster_max=int(lab_per.max()),
                          edge_cut_fraction=float(cut)))
comp = pd.DataFrame(comp_rows); comp.to_csv(f"{OUT}components.csv", index=False)
print(comp.to_string()); print(f"[graph] done {time.time()-T0:.0f}s")

def make_weights(cluster, strata, B, rng):
    """Replicate weights, TRANSPOSED (nT x B, float32): node weight = times its cluster is drawn."""
    WT = np.zeros((nT, B), dtype=np.float32)
    for s in np.unique(strata):
        ix = np.where(strata == s)[0]; u, inv = np.unique(cluster[ix], return_inverse=True)
        cnt = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(np.float32)
        WT[ix] = cnt.T[inv]
    return WT
ctest = comm[test]; assert (ctest >= 0).all()
rng = np.random.default_rng(RNG_SEED)
SCHEME_DEF = {"txn": (np.arange(nT), tt), "comm": (ctest, tt), "step": (tt, np.zeros(nT))}
SCHEMES = list(SCHEME_DEF)

def ap_w(a, p, WT):
    """Weighted AP for every column of WT (sklearn step definition with sample_weight, ties grouped).
    Only tie groups that contain a positive contribute, so the ranking is cut into segments ending at
    those groups; segment weight sums are obtained with one sparse product (no per-replicate sort)."""
    o = np.argsort(-p, kind="mergesort"); ps, a_s = p[o], a[o]
    grp = np.r_[0, np.cumsum(np.diff(ps) != 0)]                 # tie-group id in rank order
    gpos = np.unique(grp[a_s])                                     # groups containing a positive
    gend = np.searchsorted(grp, gpos, side="right") - 1            # last rank of each such group
    seg = np.searchsorted(gend, np.arange(len(ps)), side="left")   # segment id per rank (len(gend) = beyond)
    keep = seg < len(gend); rows = seg[keep]; cols = o[keep]; ns = len(gend)
    S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ns, len(p)))
    Sp = sp.csr_matrix((np.ones(int(a_s[keep].sum()), np.float32), (rows[a_s[keep]], cols[a_s[keep]])), shape=(ns, len(p)))
    k = np.cumsum((S @ WT).astype(np.float64), axis=0); tps = (Sp @ WT).astype(np.float64); tp = np.cumsum(tps, axis=0)
    prec = np.divide(tp, k, out=np.zeros_like(tp), where=k > 0)
    return (prec * tps).sum(0) / np.maximum(tp[-1], 1e-12)
def f1_w(a, p, thr, WT):
    h = p >= thr; tp = (a & h).astype(np.float32) @ WT; d = a.astype(np.float32) @ WT + h.astype(np.float32) @ WT
    return np.divide(2 * tp.astype(np.float64), d, out=np.zeros(WT.shape[1]), where=d > 0)

# ---------------- predictions + calibration (identical to v1) ----------------
files = sorted(glob.glob(f"{R}/preds/*_seed*.npz")); store, rows, skipped = {}, [], []
for f in files:
    mname = os.path.basename(f).split("_seed")[0]; seed = int(f.split("_seed")[1].split(".")[0])
    try:
        d = np.load(f); po, pt = d["oot"][cal], d["test"][test]
        assert not np.isnan(po).any() and not np.isnan(pt).any()
    except Exception as e:
        skipped.append((os.path.basename(f), repr(e)[:80])); continue
    pl = Platt().fit(po, yi[cal]); pc = pl.transform(pt); thr = best_f1_threshold(yi[cal], pl.transform(po))
    store[(mname, seed)] = (pc, thr)
if skipped: print("[warn] skipped unreadable/incomplete prediction files:", skipped)
methods = sorted({m for m, _ in store}); seeds_of = {m: sorted(s for mm, s in store if mm == m) for m in methods}
key_pairs = [(a, b) for a, b in KEY if a in methods and b in methods]
key_methods = {x for pr in key_pairs for x in pr}
print("[methods]", {m: len(seeds_of[m]) for m in methods}); print("[key]", key_pairs)
obs, rep, step_f1 = {}, {}, {}
for (mname, seed), (pc, thr) in store.items():
    obs[(mname, seed, "pr_auc")] = average_precision_score(yt, pc); obs[(mname, seed, "f1")] = f1_w(yb, pc, thr, np.ones((nT, 1), np.float32))[0]
    step_f1[(mname, seed)] = per_step_f1(yt, pc, thr, tt)
pc0 = store[(PROPOSED, 42)][0]
assert abs(ap_w(yb, pc0, np.ones((nT, 1), np.float32))[0] - average_precision_score(yt, pc0)) < 1e-9
for sch in SCHEMES:                     # one scheme in memory at a time (nT x 5000 float32 = 0.33 GB)
    WT = make_weights(*SCHEME_DEF[sch], B_KEY, rng)
    for r in range(3):
        assert abs(ap_w(yb, pc0, WT[:, r:r + 1])[0] - average_precision_score(yt, pc0, sample_weight=WT[:, r])) < 1e-9
    for (mname, seed), (pc, thr) in store.items():
        Bm = B_KEY if mname in key_methods else B_REST; Wm = WT[:, :Bm]
        rep[(mname, seed, sch, "pr_auc")] = ap_w(yb, pc, Wm); rep[(mname, seed, sch, "f1")] = f1_w(yb, pc, thr, Wm)
    del WT; print(f"[boot] scheme {sch} done {time.time()-T0:.0f}s", flush=True)
print(f"[boot] replicate metrics done {time.time()-T0:.0f}s")

def pval(d):
    return min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
def seedavg(m, k):
    return np.mean([obs[(m, s, k)] for s in seeds_of[m]])
def boot_seedavg(m, sch, k, B):
    return np.mean([rep[(m, s, sch, k)][:B] for s in seeds_of[m]], axis=0)
srng = np.random.default_rng(RNG_SEED + 1)
def two_stage(a, b, k, B):
    """Clusters (comm) x seeds: for replicate r, draw seeds with replacement (paired by seed value if
    both methods share the seed set) and average the replicate-r metric over the drawn seeds."""
    Ma = np.array([rep[(a, s, "comm", k)][:B] for s in seeds_of[a]]); Mb = np.array([rep[(b, s, "comm", k)][:B] for s in seeds_of[b]])
    if seeds_of[a] == seeds_of[b]:
        ia = srng.integers(0, len(seeds_of[a]), size=(B, len(seeds_of[a]))); ib = ia
    else:
        ia = srng.integers(0, len(seeds_of[a]), size=(B, len(seeds_of[a]))); ib = srng.integers(0, len(seeds_of[b]), size=(B, len(seeds_of[b])))
    cols = np.arange(B)
    return Ma[ia, cols[:, None]].mean(1) - Mb[ib, cols[:, None]].mean(1)

def compare(a, b, with_two_stage=False):
    B = B_KEY if (a in key_methods and b in key_methods) else B_REST
    r = dict(comparison=f"{a} vs {b}", method_a=a, method_b=b, seeds_a=len(seeds_of[a]), seeds_b=len(seeds_of[b]), B=B)
    for k in METRICS:
        r[f"delta_{k}"] = seedavg(a, k) - seedavg(b, k)
        for sch in SCHEMES:
            d = boot_seedavg(a, sch, k, B) - boot_seedavg(b, sch, k, B)
            r[f"ci_lo_{k}_{sch}"], r[f"ci_hi_{k}_{sch}"] = np.percentile(d, [2.5, 97.5]); r[f"p_{k}_{sch}"] = pval(d)
            r[f"se_{k}_{sch}"] = d.std(ddof=1)
        if with_two_stage:
            d = two_stage(a, b, k, B); r[f"ci_lo_{k}_2stage"], r[f"ci_hi_{k}_2stage"] = np.percentile(d, [2.5, 97.5])
            r[f"p_{k}_2stage"] = pval(d); r[f"se_{k}_2stage"] = d.std(ddof=1)
    sa = pd.DataFrame([step_f1[(a, s)] for s in seeds_of[a]]).mean(); sb = pd.DataFrame([step_f1[(b, s)] for s in seeds_of[b]]).mean()
    diff = (sa - sb).values
    r["p_wilcoxon_step_f1"] = 1.0 if np.allclose(diff, 0) else wilcoxon(diff, zero_method="wilcox").pvalue
    r["steps_won"], r["steps_lost"] = int((diff > 0).sum()), int((diff < 0).sum())
    return r

# ---------------- primary table: proposed vs every method + secondary family ----------------
prim = [dict(compare(PROPOSED, o), family="primary") for o in methods if o != PROPOSED]
prim += [dict(compare(a, b), family="secondary") for a, b in SECONDARY if a in methods and b in methods]
P = pd.DataFrame(prim)
for fam in ["primary", "secondary"]:
    ix = P.family == fam
    for k in METRICS:
        for sch in SCHEMES: P.loc[ix, f"p_{k}_{sch}_holm"] = holm(P.loc[ix, f"p_{k}_{sch}"])
    P.loc[ix, "p_wilcoxon_step_f1_holm"] = holm(P.loc[ix, "p_wilcoxon_step_f1"])
    # cross-metric Holm within family: comparisons x {pr_auc, f1} on the primary (comm) scheme
    pp = np.r_[P.loc[ix, "p_pr_auc_comm"].values, P.loc[ix, "p_f1_comm"].values]; hh = holm(pp); m = ix.sum()
    P.loc[ix, "p_pr_auc_comm_holm_xmetric"], P.loc[ix, "p_f1_comm_holm_xmetric"] = hh[:m], hh[m:]
P.to_csv(f"{OUT}primary.csv", index=False)

# ---------------- key family ----------------
K = pd.DataFrame([compare(a, b, with_two_stage=True) for a, b in key_pairs])
for sch in list(SCHEMES) + ["2stage"]:
    for k in METRICS: K[f"p_{k}_{sch}_holm_permetric"] = holm(K[f"p_{k}_{sch}"])
    pp = np.r_[K[f"p_pr_auc_{sch}"].values, K[f"p_f1_{sch}"].values]; hh = holm(pp); m = len(K)
    K[f"p_pr_auc_{sch}_holm_family"], K[f"p_f1_{sch}_holm_family"] = hh[:m], hh[m:]
# sensitivity: family also containing the step-level Wilcoxon (3 tests per comparison)
pp = np.r_[K["p_pr_auc_comm"].values, K["p_f1_comm"].values, K["p_wilcoxon_step_f1"].values]; hh = holm(pp); m = len(K)
K["p_pr_auc_comm_holm_family3"], K["p_f1_comm_holm_family3"], K["p_wilcoxon_holm_family3"] = hh[:m], hh[m:2 * m], hh[2 * m:]
K["min_attainable_p_raw"] = 2 / (K.B + 1); K["min_attainable_p_holm_family"] = 2 * 2 * len(K) / (K.B + 1)
K.to_csv(f"{OUT}key.csv", index=False)

# ---------------- seed-level paired analysis ----------------
srows = []
for a, b in key_pairs:
    common = sorted(set(seeds_of[a]) & set(seeds_of[b]))
    for k in METRICS:
        d = np.array([obs[(a, s, k)] - obs[(b, s, k)] for s in common])
        r = dict(comparison=f"{a} vs {b}", metric=k, n_seeds_paired=len(common), seeds=" ".join(map(str, common)),
                 mean_diff=d.mean(), sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan, min_diff=d.min(), max_diff=d.max(),
                 n_positive=int((d > 0).sum()), n_negative=int((d < 0).sum()),
                 sd_seed_a=np.std([obs[(a, s, k)] for s in seeds_of[a]], ddof=1), sd_seed_b=np.std([obs[(b, s, k)] for s in seeds_of[b]], ddof=1))
        r["per_seed_diffs"] = " ".join(f"{x:+.5f}" for x in d)
        nz = d[d != 0]
        r["p_sign_exact"] = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
        r["p_wilcoxon_exact"] = wilcoxon(d, method="exact").pvalue if len(nz) == len(d) and len(d) >= 2 else np.nan
        r["min_attainable_p_two_sided"] = 2 / 2 ** len(common) if len(common) else np.nan
        srows.append(r)
S = pd.DataFrame(srows); S.to_csv(f"{OUT}seed.csv", index=False)

# ---------------- TOST equivalence ----------------
trows = []
for a, b in TOST:
    if a not in methods or b not in methods: continue
    B = B_KEY
    for k in METRICS:
        mg = MARGIN[k]
        for sch, d in [("comm", boot_seedavg(a, "comm", k, B) - boot_seedavg(b, "comm", k, B)),
                       ("step", boot_seedavg(a, "step", k, B) - boot_seedavg(b, "step", k, B)),
                       ("2stage", two_stage(a, b, k, B))]:
            p_lo = ((d <= -mg).sum() + 1) / (B + 1); p_hi = ((d >= mg).sum() + 1) / (B + 1)
            lo90, hi90 = np.percentile(d, [5, 95])
            trows.append(dict(comparison=f"{a} vs {b}", metric=k, scheme=sch, margin=mg, delta=seedavg(a, k) - seedavg(b, k),
                              ci90_lo=lo90, ci90_hi=hi90, p_tost_lower=p_lo, p_tost_upper=p_hi, p_tost=max(p_lo, p_hi),
                              equivalent_at_0_05=bool(max(p_lo, p_hi) < 0.05 and lo90 > -mg and hi90 < mg), B=B))
T = pd.DataFrame(trows); T["p_tost_holm"] = np.nan
for sch in T.scheme.unique():
    ix = T.scheme == sch; T.loc[ix, "p_tost_holm"] = holm(T.loc[ix, "p_tost"])
T.to_csv(f"{OUT}tost.csv", index=False)

pd.set_option("display.width", 250)
print(K[["comparison", "delta_pr_auc", "p_pr_auc_txn", "p_pr_auc_comm", "p_pr_auc_step", "p_pr_auc_2stage", "delta_f1", "p_f1_txn", "p_f1_comm", "p_f1_step", "p_f1_2stage", "p_pr_auc_comm_holm_family", "p_f1_comm_holm_family", "p_f1_2stage_holm_family"]].round(4).to_string())
print(S[["comparison", "metric", "mean_diff", "sd_diff", "n_positive", "p_sign_exact", "p_wilcoxon_exact"]].round(5).to_string())
import resource
print(T.round(5).to_string()); print(f"[done] {time.time()-T0:.0f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
