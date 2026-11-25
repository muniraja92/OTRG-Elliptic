"""Degree-preserving graph control (revision fix for the edge-shuffle control).

The original control (src.preprocessing.shuffle_edges_within_step) re-draws both endpoints
of every edge uniformly within a time step, so it keeps only per-step edge counts and
changes every node's degree (and hence the log-degree features). Here we build a
configuration-model-style control that keeps every node's in-degree AND out-degree:
within each time step the `dst` endpoints are permuted across that step's edges (sources
fixed), and self-loops / duplicate directed edges are then repaired by degree-preserving
dst-swaps between pairs of edges of the same step. Rewiring rng seed 2024.

Stages (run each under `flock /tmp/cpu.lock`):
  train    rewire + verify, tune LGBM on AF+Gdp (train 1-29 / val 30-34, same grid as
           run_experiments.py), cross-fitted OOT + final 1-34 fit -> test for seeds
           42 1 7 123 2024 (method LGBM-AF+Gdp), and the reliability gate over
           LGBM-AF and LGBM-AF+Gdp exactly as TRUST-GRAF in refit_fusion.py (method
           A-dp-graph-gate).
  valcheck validation-only sanity check on calibration block 30-34 (no test data read).
  test     ONLY after all predictions are frozen: Platt + F1-optimal threshold on 30-34
           (as statistical_tests.py), test PR-AUC/F1, paired stratified bootstrap.
"""
from __future__ import annotations
import sys, os, json, time, logging, argparse, hashlib
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.makedirs(f"{ROOT}/logs", exist_ok=True)
# configure logging BEFORE importing run_experiments so its basicConfig (logs/run.log) is a no-op
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(f"{ROOT}/logs/dp_control.log"), logging.StreamHandler()])
log = logging.info
R = f"{ROOT}/results"; SEEDS = [42, 1, 7, 123, 2024]; METHOD = "LGBM-AF+Gdp"; GATE = "A-dp-graph-gate"


# ----------------------------------------------------------------------------------------
def degree_preserving_rewire(src, dst, time_step, rng, max_passes=200):
    """Within-step directed rewiring preserving every node's in- and out-degree.

    1. per step: permute dst endpoints across that step's edges (sources untouched);
    2. repair: each edge that is a self-loop or a duplicate of another directed edge is
       swapped (dst_i <-> dst_j) with a random edge j of the same step, accepted only if
       neither resulting edge is a self-loop or an existing directed edge. Swaps of dst
       between two edges never change any node's in- or out-degree.
    Returns new (src, dst) and a stats dict."""
    s2, d2 = src.copy(), dst.copy(); et = time_step[src]
    stats = dict(passes_used=0, swaps_accepted=0, initial_selfloops=0, initial_dup_edges=0)
    for t in np.unique(et):
        E = np.where(et == t)[0]
        d2[E] = d2[E][rng.permutation(len(E))]
        pairs = {}
        for i in E:
            k = (s2[i], d2[i]); pairs[k] = pairs.get(k, 0) + 1
        def is_bad(i):
            return s2[i] == d2[i] or pairs[(s2[i], d2[i])] > 1
        bad = [i for i in E if is_bad(i)]
        stats["initial_selfloops"] += int(sum(s2[i] == d2[i] for i in bad))
        stats["initial_dup_edges"] += int(sum(pairs[(s2[i], d2[i])] > 1 for i in bad if s2[i] != d2[i]))
        for p in range(max_passes):
            if not bad: break
            stats["passes_used"] = max(stats["passes_used"], p + 1)
            for i in bad:
                if not is_bad(i): continue
                for _ in range(20):                      # a few attempts per bad edge per pass
                    j = E[rng.integers(len(E))]
                    if j == i or d2[i] == d2[j]: continue
                    a, b = (s2[i], d2[j]), (s2[j], d2[i])
                    if a[0] == a[1] or b[0] == b[1] or pairs.get(a, 0) > 0 or pairs.get(b, 0) > 0: continue
                    pairs[(s2[i], d2[i])] -= 1; pairs[(s2[j], d2[j])] -= 1
                    d2[i], d2[j] = d2[j], d2[i]
                    pairs[a] = pairs.get(a, 0) + 1; pairs[b] = pairs.get(b, 0) + 1
                    stats["swaps_accepted"] += 1; break
            bad = [i for i in E if is_bad(i)]
    return s2, d2, stats


def graph_stats(src, dst, n):
    key = src.astype(np.int64) * n + dst
    u, c = np.unique(key, return_counts=True)
    return dict(n_edges=int(len(src)), self_loops=int((src == dst).sum()),
                multi_edge_excess=int((c - 1).sum()), distinct_directed_pairs=int(len(u)))


def verify(src, dst, s2, d2, t, n):
    din0, dout0 = np.bincount(dst, minlength=n), np.bincount(src, minlength=n)
    din1, dout1 = np.bincount(d2, minlength=n), np.bincount(s2, minlength=n)
    assert np.array_equal(din0, din1), "in-degree changed"
    assert np.array_equal(dout0, dout1), "out-degree changed"
    assert (t[s2] == t[d2]).all(), "cross-step edge created"
    k0 = set((src.astype(np.int64) * n + dst).tolist()); k1 = (s2.astype(np.int64) * n + d2)
    retained = float(np.mean([k in k0 for k in k1.tolist()]))
    und0 = set((np.minimum(src, dst).astype(np.int64) * n + np.maximum(src, dst)).tolist())
    und1 = np.minimum(s2, d2).astype(np.int64) * n + np.maximum(s2, d2)
    retained_und = float(np.mean([k in und0 for k in und1.tolist()]))
    # reciprocal pairs (u->v and v->u), which collapse in the undirected dispersion view
    def recip(s, d):
        ks = set((s.astype(np.int64) * n + d).tolist()); return int(sum((b * n + a) in ks for a, b in zip(s.tolist(), d.tolist())))
    return dict(in_degree_identical=True, out_degree_identical=True, same_step=True,
                frac_original_directed_edges_retained=retained, frac_original_undirected_edges_retained=retained_und,
                reciprocal_edges_original=recip(src, dst), reciprocal_edges_rewired=recip(s2, d2),
                original=graph_stats(src, dst, n), rewired=graph_stats(s2, d2, n))


# ----------------------------------------------------------------------------------------
def stage_train():
    from src import preprocessing as pp
    from src.model import make, tune_lgbm
    from src.fusion import ReliabilityGate, reliability_features
    from run_experiments import novelty, BLOCKS
    T0 = time.time()
    df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values
    lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int); M = lambda a, b: lab & (t >= a) & (t <= b); test = M(35, 49)
    XL, XAgg = pp.feature_blocks(df)

    # ---- rewiring + verification ----
    rng = np.random.default_rng(2024); t1 = time.time()
    s2, d2, rstats = degree_preserving_rewire(src, dst, t, rng)
    ver = verify(src, dst, s2, d2, t, n); ver.update(rstats, rewire_seconds=time.time() - t1, rng_seed=2024)
    # compare with the original uniform shuffle (same rng seed as run_experiments.py)
    us, ud = pp.shuffle_edges_within_step(src, dst, t, np.random.default_rng(2024))
    ver["uniform_shuffle"] = dict(graph_stats(us, ud, n),
        frac_nodes_in_degree_changed=float((np.bincount(ud, minlength=n) != np.bincount(dst, minlength=n)).mean()),
        frac_nodes_out_degree_changed=float((np.bincount(us, minlength=n) != np.bincount(src, minlength=n)).mean()))
    ver["edges_sha256"] = hashlib.sha256(np.stack([s2, d2]).astype(np.int64).tobytes()).hexdigest()
    json.dump(ver, open(f"{R}/dp_control_rewiring.json", "w"), indent=2); log(f"rewiring verified: {json.dumps(ver)}")
    del us, ud

    G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n); struct = G[:, -3:].copy()   # real-graph struct, as in TRUST-GRAF
    Gdp, _, _, _ = pp.neighbourhood_view(XL, s2, d2, n)
    assert np.allclose(G[:, -3:-1], Gdp[:, -3:-1]), "log-degree features must be identical"
    del G
    X = np.hstack([XL, XAgg, Gdp]); del Gdp, XAgg
    log(f"AF+Gdp view built: {X.shape} ({time.time()-T0:.0f}s)")

    # ---- tuning, identical to run_experiments.py ----
    tr, va = M(1, 29), M(30, 34)
    hp, sval = tune_lgbm(X[tr], yi[tr], X[va], yi[va]); log(f"tuned {METHOD} {hp} valAP={sval:.4f}")
    json.dump({METHOD: hp, "valAP_tuning_seed42": sval}, open(f"{R}/dp_control_hyperparameters.json", "w"), indent=2)

    side = {k: novelty(XL, t, t < k[0]) for k in BLOCKS}; side["test"] = novelty(XL, t, t <= 34)
    l2 = json.load(open(f"{R}/gate_lambda_choice.json"))["TRUST-GRAF"]; fit_m, b4 = M(15, 29), M(30, 34)
    runtime = []
    for seed in SEEDS:
        oot = np.full(n, np.nan)
        for (a, b) in BLOCKS:
            trm, bm = lab & (t < a), M(a, b)
            mdl = make(METHOD, seed, hp).fit(X[trm], yi[trm]); oot[bm] = mdl.predict_proba(X[bm])[:, 1]
        t1 = time.time(); trm = M(1, 34); mdl = make(METHOD, seed, hp).fit(X[trm], yi[trm]); fs = time.time() - t1
        pt = np.full(n, np.nan); pt[test] = mdl.predict_proba(X[test])[:, 1]
        np.savez_compressed(f"{R}/preds/{METHOD}_seed{seed}.npz", oot=oot, test=pt)
        runtime.append(dict(model=METHOD, seed=seed, fit_s=fs)); log(f"seed {seed} {METHOD} saved (final fit {fs:.1f}s)")

        # ---- reliability gate, exactly as TRUST-GRAF in refit_fusion.py ----
        pa = np.load(f"{R}/preds/LGBM-AF_seed{seed}.npz"); pb = dict(oot=oot, test=pt)
        Ro = np.full((n, 7), np.nan)
        for (a, b) in BLOCKS:
            bm = M(a, b); Ro[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], np.column_stack([struct, side[(a, b)]])[bm])
        Rt = np.full((n, 7), np.nan); Rt[test] = reliability_features(pa["test"][test], pb["test"][test], np.column_stack([struct, side["test"]])[test])
        g = ReliabilityGate(l2=l2, use_features=True).fit(pa["oot"][fit_m], pb["oot"][fit_m], Ro[fit_m], yi[fit_m])
        go = np.full(n, np.nan); go[b4], _ = g.predict(pa["oot"][b4], pb["oot"][b4], Ro[b4])
        gt = np.full(n, np.nan); al = np.full(n, np.nan); gt[test], al[test] = g.predict(pa["test"][test], pb["test"][test], Rt[test])
        np.savez_compressed(f"{R}/preds/{GATE}_seed{seed}.npz", oot=go, test=gt, alpha=al, w=g.w_)
        log(f"seed {seed} {GATE} saved l2={l2} w={np.round(g.w_, 4).tolist()}")
    pd.DataFrame(runtime).to_csv(f"{R}/dp_control_runtime.csv", index=False)
    # freeze: record digests of every prediction file written by this script
    dig = {}
    for m in (METHOD, GATE):
        for s in SEEDS:
            f = f"{R}/preds/{m}_seed{s}.npz"; dig[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    json.dump(dig, open(f"{R}/dp_control_frozen_preds.json", "w"), indent=2); log(f"ALL DONE {time.time()-T0:.0f}s")


# ----------------------------------------------------------------------------------------
def stage_valcheck():
    """Validation-only: PR-AUC on calibration block 30-34 from OOT predictions. Never reads `test`."""
    from sklearn.metrics import average_precision_score, log_loss
    meta = np.load(f"{R}/preds/_meta.npz"); y, t = meta["y"], meta["t"]
    cal = (~np.isnan(y)) & (t >= 30) & (t <= 34); yc = np.nan_to_num(y).astype(int)[cal]
    rows = []
    for m in ["LGBM-AF", "LGBM-AF+G", "LGBM-AF+Gshuf", METHOD, "TRUST-GRAF", "A-shuffled-graph", GATE]:
        for s in SEEDS:
            p = np.load(f"{R}/preds/{m}_seed{s}.npz")["oot"][cal]; assert not np.isnan(p).any()
            rows.append(dict(method=m, seed=s, val_pr_auc=average_precision_score(yc, p), val_logloss=log_loss(yc, np.clip(p, 1e-6, 1 - 1e-6))))
    v = pd.DataFrame(rows); v.to_csv(f"{R}/dp_control_valcheck.csv", index=False)
    print(v.groupby("method")[["val_pr_auc", "val_logloss"]].agg(["mean", "std"]).round(4).to_string())


# ----------------------------------------------------------------------------------------
def ap_fast(a, p):
    """Average precision identical to sklearn's step-wise definition (same as statistical_tests.py)."""
    o = np.argsort(-p, kind="mergesort"); a, p = a[o], p[o]
    tp = np.cumsum(a); last = np.r_[np.diff(p) != 0, True]
    tp_l, k = tp[last], np.arange(1, len(a) + 1)[last]; prec = tp_l / k
    rec_inc = np.diff(np.r_[0, tp_l]) / max(1, a.sum()); return float((prec * rec_inc).sum())


def f1_fast(a, p, thr):
    h = p >= thr; tp = (a & h).sum(); d = a.sum() + h.sum(); return 2 * tp / d if d else 0.0


def stage_test():
    from sklearn.metrics import average_precision_score
    from src.evaluation import metrics, best_f1_threshold, Platt
    dig = json.load(open(f"{R}/dp_control_frozen_preds.json"))
    for f, h in dig.items():
        assert hashlib.sha256(open(f"{R}/preds/{f}", "rb").read()).hexdigest() == h, f"prediction file changed: {f}"
    meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
    yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34); yt, tt = yi[test], t[test]
    METHODS = ["LGBM-AF", "LGBM-AF+G", "LGBM-AF+Gshuf", METHOD, "TRUST-GRAF", "A-shuffled-graph", GATE]
    rows, store = [], {}
    for m in METHODS:
        for s in SEEDS:
            d = np.load(f"{R}/preds/{m}_seed{s}.npz"); po, pt = d["oot"][cal], d["test"][test]
            assert not np.isnan(po).any() and not np.isnan(pt).any()
            pl = Platt().fit(po, yi[cal]); pco, pct = pl.transform(po), pl.transform(pt); thr = best_f1_threshold(yi[cal], pco)
            mm = metrics(yt, pct, thr, tt); mm.update(method=m, seed=s); rows.append(mm); store[(m, s)] = (pct, thr)
    met = pd.DataFrame(rows); met.to_csv(f"{R}/dp_control_metrics_per_seed.csv", index=False)
    summ = met.groupby("method")[["pr_auc", "f1", "f1_post43", "roc_auc"]].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.loc[METHODS]
    rng = np.random.default_rng(12345); B = 1000
    strata = [np.where(tt == s)[0] for s in np.unique(tt)]
    boot_idx = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(B)]
    yb = yt.astype(bool); assert abs(ap_fast(yb, store[(METHOD, 42)][0]) - average_precision_score(yt, store[(METHOD, 42)][0])) < 1e-9
    FN = {"pr_auc": lambda a, p, thr: ap_fast(a, p), "f1": f1_fast}
    cache = {}
    need = ["LGBM-AF", "LGBM-AF+G", METHOD, "LGBM-AF+Gshuf", "TRUST-GRAF", GATE]
    for m in need:
        runs = [store[(m, s)] for s in SEEDS]
        for k, fn in FN.items():
            cache[(m, k, "obs")] = np.mean([fn(yb, p, thr) for p, thr in runs])
            cache[(m, k)] = np.array([np.mean([fn(yb[ix], p[ix], thr) for p, thr in runs]) for ix in boot_idx])
    PAIRS = [("LGBM-AF+G", METHOD), (METHOD, "LGBM-AF"), (METHOD, "LGBM-AF+Gshuf"), ("TRUST-GRAF", GATE)]
    out = []
    for m in METHODS:
        r = dict(row_type="method", name=m, **summ.loc[m].to_dict()); out.append(r)
    for a, b in PAIRS:
        r = dict(row_type="comparison", name=f"{a} vs {b}", primary=(a, b) in [("LGBM-AF+G", METHOD), (METHOD, "LGBM-AF")])
        for k in FN:
            dd = cache[(a, k)] - cache[(b, k)]; r[f"delta_{k}"] = cache[(a, k, "obs")] - cache[(b, k, "obs")]
            r[f"ci_lo_{k}"], r[f"ci_hi_{k}"] = np.percentile(dd, [2.5, 97.5]); r[f"p_boot_{k}"] = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()) + 1 / B)
        out.append(r)
    res = pd.DataFrame(out)
    # Holm within the two pre-specified primary comparisons, per metric
    prim = res.row_type.eq("comparison") & res.primary.eq(True)
    for k in FN:
        p = res.loc[prim, f"p_boot_{k}"].values; o = np.argsort(p); adj = np.empty(len(p)); run = 0
        for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
        res.loc[prim, f"p_boot_{k}_holm"] = adj
    res.to_csv(f"{R}/dp_control_results.csv", index=False)
    pd.set_option("display.width", 250); print(res.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["train", "valcheck", "test", "selftest"]); a = ap.parse_args()
    if a.stage == "selftest":   # synthetic graph, unit test only
        rng = np.random.default_rng(0); n = 300; ts = np.repeat([1, 2, 3], 100)
        s = np.concatenate([rng.choice(np.arange(k * 100, k * 100 + 100), 400) for k in range(3)])
        d = np.concatenate([rng.choice(np.arange(k * 100, k * 100 + 100), 400) for k in range(3)])
        ok = s != d; s, d = s[ok], d[ok]; key = np.unique(s * n + d, return_index=True)[1]; s, d = s[key], d[key]
        s2, d2, st = degree_preserving_rewire(s, d, ts, np.random.default_rng(2024)); print(st, verify(s, d, s2, d2, ts, n))
    else:
        {"train": stage_train, "valcheck": stage_valcheck, "test": stage_test}[a.stage]()
