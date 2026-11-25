"""Tuning sensitivity (revision Issue B): disjoint tuning and calibration blocks, expanded grid.

The main study tuned LightGBM on train 1..29 / validate 30..34 and then used block 30..34 again
for Platt scaling and the F1-optimal threshold. Here the two experts are re-tuned on
train 1..24 / validate 25..29 (early stopping and model choice on 25..29 only) over an expanded grid
  num_leaves {7,15,31,63,127} x min_child_samples {10,20,50,100} x colsample_bytree {0.3,0.5,0.8},
  learning_rate 0.05 (other settings as src/model.lgbm), selection by validation average precision,
so block 30..34 is used only for calibration and threshold. With the new hyperparameters the
cross-fitting (run_experiments.BLOCKS) and final 1..34 refit are rerun for seeds 42,1,7,123,2024,
followed by the TRUST-GRAF gate (l2 from results/gate_lambda_choice.json, as in refit_fusion.py),
A-static-gate and A-average. Predictions go to results/preds_tuning_v2/ under new method names
(suffix -T2) and never to results/preds. No test-period label or prediction enters any choice.

Stages: python experiments/tuning_sensitivity.py tune | fit | eval   (default: all three)
Outputs: results/tuning_v2_grid.csv, results/tuning_v2_hyperparameters.json, results/preds_tuning_v2/*.npz,
         results/tuning_v2_metrics.csv, results/tuning_v2_summary.csv, results/tuning_v2_paired.csv
"""
import sys, os, json, time, glob, itertools, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import average_precision_score
R = f"{ROOT}/results"; P2 = f"{R}/preds_tuning_v2"; os.makedirs(P2, exist_ok=True)
SEEDS = [42, 1, 7, 123, 2024]; SUF = "-T2"
GRID = list(itertools.product((7, 15, 31, 63, 127), (10, 20, 50, 100), (0.3, 0.5, 0.8)))
stages = sys.argv[1:] or ["tune", "fit", "eval"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def lgbm(seed, num_leaves, min_child_samples, colsample_bytree, n_estimators):
    """Identical to src.model.lgbm except that colsample_bytree is a tuned parameter."""
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=num_leaves,
                              min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                              colsample_bytree=colsample_bytree, reg_lambda=1.0, random_state=seed, n_jobs=2,
                              verbose=-1, deterministic=True, force_col_wise=True)


def load_views():
    from src import preprocessing as pp
    df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values
    XL, XA = pp.feature_blocks(df); G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n)
    AFG = np.hstack([XL, XA, G]); struct = G[:, -3:].copy(); del G, XA
    return df, XL, AFG, struct, n, t, y


if "tune" in stages or "fit" in stages:
    df, XL, AFG, struct, n, t, y = load_views(); NAF = 165
    lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int); M = lambda a, b: lab & (t >= a) & (t <= b)
    VIEW = {"LGBM-AF": slice(0, NAF), "LGBM-AF+G": slice(0, AFG.shape[1])}
    log(f"loaded n={n} AF={NAF} AF+G={AFG.shape[1]}")

if "tune" in stages:
    tr, va = M(1, 24), M(25, 29); grid_rows = []; hp = {}
    for name, cols in VIEW.items():
        Xtr, Xva = AFG[tr][:, cols], AFG[va][:, cols]; best = None
        for (nl, mcs, cs) in GRID:
            t1 = time.time(); m = lgbm(42, nl, mcs, cs, 2000)
            m.fit(Xtr, yi[tr], eval_set=[(Xva, yi[va])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            ne = max(50, int(m.best_iteration_ or 2000))
            ap = average_precision_score(yi[va], m.predict_proba(Xva, num_iteration=ne)[:, 1])
            grid_rows.append(dict(model=name, num_leaves=nl, min_child_samples=mcs, colsample_bytree=cs, n_estimators=ne, val_ap_25_29=ap, fit_s=time.time() - t1))
            if best is None or ap > best[0]: best = (ap, dict(num_leaves=nl, min_child_samples=mcs, colsample_bytree=cs, n_estimators=ne))
            pd.DataFrame(grid_rows).to_csv(f"{R}/tuning_v2_grid.csv", index=False)
        hp[name] = dict(**best[1], val_ap_25_29=best[0]); log(f"tuned {name} {hp[name]}"); del Xtr, Xva
    hp["_protocol"] = "train labelled steps 1..24, validate 25..29 (early stopping and selection by average precision), seed 42; block 30..34 reserved for Platt + threshold"
    json.dump(hp, open(f"{R}/tuning_v2_hyperparameters.json", "w"), indent=2)

if "fit" in stages:
    from src.fusion import ReliabilityGate, reliability_features
    from run_experiments import novelty, BLOCKS
    hp = json.load(open(f"{R}/tuning_v2_hyperparameters.json")); lam = json.load(open(f"{R}/gate_lambda_choice.json"))["TRUST-GRAF"]
    test = M(35, 49); fit_m, b4 = M(15, 29), M(30, 34)
    side = {k: novelty(XL, t, t < k[0]) for k in BLOCKS}; side["test"] = novelty(XL, t, t <= 34)
    gate_rows = []
    for seed in SEEDS:
        P = {}
        for name, cols in VIEW.items():
            fpath = f"{P2}/{name}{SUF}_seed{seed}.npz"
            if os.path.exists(fpath): P[name] = dict(np.load(fpath)); continue
            h = {k: hp[name][k] for k in ("num_leaves", "min_child_samples", "colsample_bytree", "n_estimators")}
            oot = np.full(n, np.nan)
            for (a, b) in BLOCKS:
                trm, bm = lab & (t < a), M(a, b)
                mdl = lgbm(seed, **h).fit(AFG[trm][:, cols], yi[trm]); oot[bm] = mdl.predict_proba(AFG[bm][:, cols])[:, 1]
            trm = M(1, 34); mdl = lgbm(seed, **h).fit(AFG[trm][:, cols], yi[trm])
            pt = np.full(n, np.nan); pt[test] = mdl.predict_proba(AFG[test][:, cols])[:, 1]
            P[name] = dict(oot=oot, test=pt); np.savez_compressed(fpath, **P[name]); log(f"seed {seed} {name}{SUF} done")
        pa, pb = P["LGBM-AF"], P["LGBM-AF+G"]
        Ro = np.full((n, 7), np.nan)
        for (a, b) in BLOCKS:
            bm = M(a, b); Ro[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], np.column_stack([struct, side[(a, b)]])[bm])
        Rt = np.full((n, 7), np.nan); Rt[test] = reliability_features(pa["test"][test], pb["test"][test], np.column_stack([struct, side["test"]])[test])
        for vname, l2, feat in (("TRUST-GRAF", lam, True), ("A-static-gate", 1e-2, False)):
            g = ReliabilityGate(l2=l2, use_features=feat).fit(pa["oot"][fit_m], pb["oot"][fit_m], Ro[fit_m], yi[fit_m])
            oot = np.full(n, np.nan); oot[b4], ab4 = g.predict(pa["oot"][b4], pb["oot"][b4], Ro[b4])
            pt = np.full(n, np.nan); al = np.full(n, np.nan); pt[test], al[test] = g.predict(pa["test"][test], pb["test"][test], Rt[test])
            np.savez_compressed(f"{P2}/{vname}{SUF}_seed{seed}.npz", oot=oot, test=pt, alpha=al, w=g.w_)
            gate_rows.append(dict(method=vname + SUF, seed=seed, w=json.dumps(np.round(g.w_, 4).tolist()), alpha_mean_b4=float(ab4.mean()),
                                  alpha_p10_b4=float(np.percentile(ab4, 10)), alpha_p90_b4=float(np.percentile(ab4, 90)),
                                  max_abs_w=float(np.abs(g.w_[1:]).max()) if feat else 0.0))
            log(f"seed {seed} {vname}{SUF} w={np.round(g.w_, 3).tolist()}")
        np.savez_compressed(f"{P2}/A-average{SUF}_seed{seed}.npz", oot=0.5 * (pa["oot"] + pb["oot"]), test=0.5 * (pa["test"] + pb["test"]))
    pd.DataFrame(gate_rows).to_csv(f"{R}/tuning_v2_gate.csv", index=False)
    np.savez_compressed(f"{P2}/_meta.npz", y=y, t=t, test=test)

if "eval" in stages:
    from src.evaluation import metrics, best_f1_threshold, Platt
    meta = np.load(f"{P2}/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
    yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34); yt, tt = yi[test], t[test]
    rows, store = [], {}
    for f in sorted(glob.glob(f"{P2}/*{SUF}_seed*.npz")):
        mname = os.path.basename(f).split("_seed")[0]; seed = int(f.split("_seed")[1].split(".")[0]); d = np.load(f)
        po, pt = d["oot"][cal], d["test"][test]; assert not np.isnan(po).any() and not np.isnan(pt).any(), f
        pl = Platt().fit(po, yi[cal]); pc_o, pc_t = pl.transform(po), pl.transform(pt); thr = best_f1_threshold(yi[cal], pc_o)
        m = metrics(yt, pc_t, thr, tt); m.update(method=mname, seed=seed); rows.append(m); store[(mname, seed)] = (pc_t, thr)
    met = pd.DataFrame(rows); met.to_csv(f"{R}/tuning_v2_metrics.csv", index=False)
    cols = ["pr_auc", "f1", "f1_pre43", "f1_post43", "precision", "recall", "brier", "ece", "roc_auc", "threshold"]
    summ = met.groupby("method")[cols].agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_seeds"] = met.groupby("method").size(); summ.to_csv(f"{R}/tuning_v2_summary.csv"); print(summ.round(4).to_string())
    # paired comparisons: seed-level differences and stratified paired bootstrap (within time step) of the seed-averaged metric
    rng = np.random.default_rng(12345); B = 2000; yb = yt.astype(bool)
    strata = [np.where(tt == s)[0] for s in np.unique(tt)]
    def ap_fast(a, p):
        o = np.argsort(-p, kind="mergesort"); a, p = a[o], p[o]; tp = np.cumsum(a); last = np.r_[np.diff(p) != 0, True]
        tp_l, k = tp[last], np.arange(1, len(a) + 1)[last]; return float(((tp_l / k) * np.diff(np.r_[0, tp_l]) / max(1, a.sum())).sum())
    def f1_fast(a, p, thr):
        h = p >= thr; tp = (a & h).sum(); dd = a.sum() + h.sum(); return 2 * tp / dd if dd else 0.0
    FN = {"pr_auc": lambda a, p, thr: ap_fast(a, p), "f1": f1_fast}
    PAIRS = [("TRUST-GRAF", "A-average"), ("TRUST-GRAF", "A-static-gate"), ("A-static-gate", "A-average"),
             ("LGBM-AF+G", "LGBM-AF"), ("TRUST-GRAF", "LGBM-AF"), ("TRUST-GRAF", "LGBM-AF+G")]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(B)]
    BV = {}
    for (mm, s), (p, thr) in store.items():
        for k, fnc in FN.items(): BV[(mm, s, k)] = np.array([fnc(yb[ix], p[ix], thr) for ix in boots])
    prow = []
    for a, b in PAIRS:
        A, Bm = a + SUF, b + SUF; r = dict(comparison=f"{A} vs {Bm}")
        for k, fnc in FN.items():
            da = np.array([fnc(yb, *store[(A, s)]) - fnc(yb, *store[(Bm, s)]) for s in SEEDS])
            r[f"delta_{k}"] = da.mean(); r[f"seed_sd_{k}"] = da.std(ddof=1); r[f"seeds_pos_{k}"] = int((da > 0).sum())
            bd = np.mean([BV[(A, s, k)] - BV[(Bm, s, k)] for s in SEEDS], axis=0)
            r[f"ci_lo_{k}"], r[f"ci_hi_{k}"] = np.percentile(bd, [2.5, 97.5]); r[f"p_boot_{k}"] = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()) + 1 / B)
        prow.append(r)
    pr = pd.DataFrame(prow); pr.to_csv(f"{R}/tuning_v2_paired.csv", index=False); print(pr.round(4).to_string())
