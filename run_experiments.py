"""End-to-end experiment driver.

Protocol (fixed before any test-set evaluation):
  test        : labelled nodes of time steps 35..49 (standard Elliptic temporal split)
  development : labelled nodes of steps 1..34 only
    tuning     train 1..29 / validate 30..34 (LightGBM, XGBoost, RF grids)
    cross-fit  out-of-time (OOT) predictions on blocks 15-19, 20-24, 25-29, 30-34,
               each from models trained on all labelled steps before the block
    gate fit   OOT blocks 15..29
    calib+thr  OOT block 30..34 (Platt scaling and F1-optimal threshold), disjoint from gate fit
  final       every learner refit on 1..34, scored once on 35..49
Usage: python run_experiments.py [--seeds 42 1 7 123 2024] [--skip-gcn]
"""
from __future__ import annotations
import argparse, json, time, os, sys, logging, platform
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import preprocessing as pp
from src.model import make, tune_lgbm
from src.fusion import ReliabilityGate, reliability_features, _logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import xgboost as xgb

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCKS = [(15, 19), (20, 24), (25, 29), (30, 34)]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(f"{ROOT}/logs/run.log"), logging.StreamHandler()])
log = logging.info


def novelty(XL, t, fit_mask, n_comp=20):
    sc = StandardScaler().fit(XL[fit_mask]); Z = sc.transform(XL)
    pca = PCA(n_comp, random_state=0).fit(Z[fit_mask])
    R = Z - pca.inverse_transform(pca.transform(Z))
    return np.log1p((R ** 2).mean(1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 123, 2024])
    ap.add_argument("--skip-gcn", action="store_true", help="kept for backward compatibility; GCN always runs via experiments/run_gcn.py"); args = ap.parse_args()
    T0 = time.time(); os.makedirs(f"{ROOT}/results/preds", exist_ok=True)
    df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values
    XL, XAgg = pp.feature_blocks(df); AF = np.hstack([XL, XAgg])
    G, gnames, _, _ = pp.neighbourhood_view(XL, src, dst, n)
    rng = np.random.default_rng(2024); s_src, s_dst = pp.shuffle_edges_within_step(src, dst, t, rng)
    Gs, _, _, _ = pp.neighbourhood_view(XL, s_src, s_dst, n)
    views = {"LF": XL, "AF": AF, "AF+G": np.hstack([AF, G]), "AF+Gshuf": np.hstack([AF, Gs])}
    lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int)
    M = lambda a, b: lab & (t >= a) & (t <= b)
    test = M(35, 49); log(f"loaded n={n} test={test.sum()} G dims={G.shape[1]} ({time.time()-T0:.0f}s)")

    # ---------- hyperparameter tuning on train<=29 / val 30..34 (seed 42) ----------
    tr, va = M(1, 29), M(30, 34); hp = {}
    for v in ["LF", "AF", "AF+G", "AF+Gshuf"]:
        hp[f"LGBM-{v}"], s = tune_lgbm(views[v][tr], yi[tr], views[v][va], yi[va]); log(f"tuned LGBM-{v} {hp[f'LGBM-{v}']} valAP={s:.4f}")
    best = None
    for md in (4, 6, 8):
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=md, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", random_state=42, n_jobs=2, eval_metric="aucpr", early_stopping_rounds=100)
        m.fit(AF[tr], yi[tr], eval_set=[(AF[va], yi[va])], verbose=False)
        s = average_precision_score(yi[va], m.predict_proba(AF[va], iteration_range=(0, m.best_iteration + 1))[:, 1])
        if best is None or s > best[0]: best = (s, dict(max_depth=md, n_estimators=max(50, m.best_iteration + 1)))
    hp["XGB"] = best[1]; log(f"tuned XGB {hp['XGB']} valAP={best[0]:.4f}")
    best = None
    for mf in ("sqrt", 50, 0.3):
        m = make("RF", 42, {"max_features": mf}).fit(AF[tr], yi[tr]); s = average_precision_score(yi[va], m.predict_proba(AF[va])[:, 1])
        if best is None or s > best[0]: best = (s, {"max_features": mf})
    hp["RF"] = best[1]; log(f"tuned RF {hp['RF']} valAP={best[0]:.4f}")
    json.dump(hp, open(f"{ROOT}/results/hyperparameters.json", "w"), indent=2, default=str)

    # ---------- novelty side-signal: fit per fold on nodes (any label) of earlier steps ----------
    side = {}
    for (a, b) in BLOCKS:
        side[(a, b)] = novelty(XL, t, t < a)
    side["test"] = novelty(XL, t, t <= 34)
    struct = G[:, -3:]

    feat_models = {"LR": "AF", "RF-Weber": "AF", "RF": "AF", "XGB": "AF", "LGBM-LF": "LF",
                   "LGBM-AF": "AF", "LGBM-AF+G": "AF+G", "LGBM-AF+Gshuf": "AF+Gshuf"}
    experts = {"LGBM-AF", "LGBM-AF+G", "LGBM-AF+Gshuf"}
    runtime = []
    for seed in args.seeds:
        P = {}  # name -> dict(oot=array n (nan where absent), test=array n)
        for name, v in feat_models.items():
            if name == "LR" and seed != args.seeds[0]:
                P[name] = dict(np.load(f"{ROOT}/results/preds/LR_seed{args.seeds[0]}.npz")); continue
            X = views[v]; oot = np.full(n, np.nan)
            blocks = BLOCKS if name in experts else BLOCKS[-1:]
            for (a, b) in blocks:
                trm, bm = lab & (t < a), M(a, b)
                mdl = make(name, seed, hp.get(name)).fit(X[trm], yi[trm]); oot[bm] = mdl.predict_proba(X[bm])[:, 1]
            t1 = time.time(); trm = M(1, 34); mdl = make(name, seed, hp.get(name)).fit(X[trm], yi[trm]); fit_s = time.time() - t1
            t1 = time.time(); pt = np.full(n, np.nan); pt[test] = mdl.predict_proba(X[test])[:, 1]; inf_s = time.time() - t1
            P[name] = dict(oot=oot, test=pt); runtime.append(dict(model=name, seed=seed, fit_s=fit_s, infer_s=inf_s, n_features=X.shape[1]))
            np.savez_compressed(f"{ROOT}/results/preds/{name}_seed{seed}.npz", **P[name]); log(f"seed {seed} {name} done (fit {fit_s:.1f}s)")

        # ---------- fusion variants (proposed + ablations) ----------
        def side_for(mask_key):
            return np.column_stack([struct, side[mask_key]])
        def assemble(pa, pb, use):
            """Reliability matrix for OOT (blockwise novelty) and test."""
            R_oot = np.full((n, 7), np.nan)
            for (a, b) in BLOCKS:
                bm = M(a, b); R_oot[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], side_for((a, b))[bm])
            R_te = np.full((n, 7), np.nan); R_te[test] = reliability_features(pa["test"][test], pb["test"][test], side_for("test")[test])
            return R_oot[:, use], R_te[:, use]
        fit_m = M(15, 29)
        variants = {"TRUST-GRAF": ("LGBM-AF", "LGBM-AF+G", list(range(7)), True),
                    "A-static-gate": ("LGBM-AF", "LGBM-AF+G", list(range(7)), False),
                    "A-structural-only-gate": ("LGBM-AF", "LGBM-AF+G", [3, 4, 5, 6], True),
                    "A-shuffled-graph": ("LGBM-AF", "LGBM-AF+Gshuf", list(range(7)), True)}
        for vname, (ea, eb, use, feat) in variants.items():
            pa, pb = P[ea], P[eb]; R_oot, R_te = assemble(pa, pb, use)
            # NOTE: pre-specified l2=1e-2; experiments/refit_fusion.py later re-selects l2 on block 30-34 (post hoc, disclosed)
            t1 = time.time(); gate = ReliabilityGate(l2=1e-2, use_features=feat).fit(pa["oot"][fit_m], pb["oot"][fit_m], R_oot[fit_m], yi[fit_m]); gs = time.time() - t1
            oot = np.full(n, np.nan); bm = M(30, 34); oot[bm], _ = gate.predict(pa["oot"][bm], pb["oot"][bm], R_oot[bm])
            pt = np.full(n, np.nan); al = np.full(n, np.nan); pt[test], al[test] = gate.predict(pa["test"][test], pb["test"][test], R_te[test])
            P[vname] = dict(oot=oot, test=pt, alpha=al, w=gate.w_)
            np.savez_compressed(f"{ROOT}/results/preds/{vname}_seed{seed}.npz", **P[vname]); runtime.append(dict(model=vname, seed=seed, fit_s=gs, infer_s=np.nan, n_features=len(use)))
            log(f"seed {seed} {vname} w={np.round(gate.w_, 3).tolist()} mean_alpha_test={np.nanmean(al):.3f}")
        # equal-weight average and logistic stacking
        pa, pb = P["LGBM-AF"], P["LGBM-AF+G"]
        P["A-average"] = dict(oot=0.5 * (pa["oot"] + pb["oot"]), test=0.5 * (pa["test"] + pb["test"]))
        Z = lambda p, m: np.column_stack([_logit(pa[p][m]), _logit(pb[p][m])])
        st = LogisticRegression(C=1.0).fit(Z("oot", fit_m), yi[fit_m]); bm = M(30, 34)
        oot = np.full(n, np.nan); oot[bm] = st.predict_proba(Z("oot", bm))[:, 1]; pt = np.full(n, np.nan); pt[test] = st.predict_proba(Z("test", test))[:, 1]
        P["Stack-LR"] = dict(oot=oot, test=pt)
        for k in ("A-average", "Stack-LR"): np.savez_compressed(f"{ROOT}/results/preds/{k}_seed{seed}.npz", **P[k])

        # GCN baseline: run separately with experiments/run_gcn.py (single code path; see README)
        pd.DataFrame(runtime).to_csv(f"{ROOT}/results/runtime_profile.csv", index=False)
    np.savez_compressed(f"{ROOT}/results/preds/_meta.npz", y=y, t=t, test=test)
    env = dict(python=platform.python_version(), platform=platform.platform(), seeds=args.seeds, total_s=time.time() - T0)
    import sklearn, lightgbm, scipy
    env.update(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, xgboost=xgb.__version__, scipy=scipy.__version__)
    try:
        import torch; env["torch"] = torch.__version__
    except Exception: pass
    json.dump(env, open(f"{ROOT}/results/environment.json", "w"), indent=2); log(f"ALL DONE in {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
