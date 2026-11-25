"""Documented Phase-11 revision: select the gate penalty lambda on validation only.

The first run used the pre-specified lambda = 1e-2 and produced near-constant gates. Here the
gate is refitted from the saved out-of-time expert predictions (no expert is retrained) for
lambda in a grid; lambda is chosen by mean out-of-time log-loss on block 30..34 across seeds.
Test predictions for every lambda are written to results/preds_sensitivity/ and evaluated
separately as a sensitivity analysis; they play no role in the choice.
"""
import sys, os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import preprocessing as pp
from src.fusion import ReliabilityGate, reliability_features
from run_experiments import novelty, BLOCKS
from sklearn.metrics import log_loss
R = f"{ROOT}/results"; os.makedirs(f"{R}/preds_sensitivity", exist_ok=True)
df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values; lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int)
XL, _ = pp.feature_blocks(df); G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n); struct = G[:, -3:]
M = lambda a, b: lab & (t >= a) & (t <= b); test = M(35, 49); fit_m, b4 = M(15, 29), M(30, 34)
side = {k: novelty(XL, t, t < k[0]) for k in BLOCKS}; side["test"] = novelty(XL, t, t <= 34)
SEEDS = [42, 1, 7, 123, 2024]; GRID = [0.0, 1e-4, 1e-3, 1e-2]
VARIANTS = {"TRUST-GRAF": ("LGBM-AF", "LGBM-AF+G", list(range(7)), True),
            "A-structural-only-gate": ("LGBM-AF", "LGBM-AF+G", [3, 4, 5, 6], True),
            "A-shuffled-graph": ("LGBM-AF", "LGBM-AF+Gshuf", list(range(7)), True)}
rows, fitted = [], {}
for seed in SEEDS:
    E = {k: np.load(f"{R}/preds/{k}_seed{seed}.npz") for k in ["LGBM-AF", "LGBM-AF+G", "LGBM-AF+Gshuf"]}
    for vname, (ea, eb, use, feat) in VARIANTS.items():
        pa, pb = E[ea], E[eb]; Ro = np.full((n, 7), np.nan)
        for (a, b) in BLOCKS:
            bm = M(a, b); Ro[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], np.column_stack([struct, side[(a, b)]])[bm])
        Rt = np.full((n, 7), np.nan); Rt[test] = reliability_features(pa["test"][test], pb["test"][test], np.column_stack([struct, side["test"]])[test])
        Ro, Rt = Ro[:, use], Rt[:, use]
        for lam in GRID:
            g = ReliabilityGate(l2=lam, use_features=feat).fit(pa["oot"][fit_m], pb["oot"][fit_m], Ro[fit_m], yi[fit_m])
            pv, _ = g.predict(pa["oot"][b4], pb["oot"][b4], Ro[b4])
            rows.append(dict(variant=vname, seed=seed, l2=lam, b4_logloss=log_loss(yi[b4], np.clip(pv, 1e-6, 1 - 1e-6)), w=json.dumps(np.round(g.w_, 4).tolist())))
            oot = np.full(n, np.nan); oot[b4] = pv; pt = np.full(n, np.nan); al = np.full(n, np.nan); pt[test], al[test] = g.predict(pa["test"][test], pb["test"][test], Rt[test])
            fitted[(vname, seed, lam)] = dict(oot=oot, test=pt, alpha=al, w=g.w_)
            np.savez_compressed(f"{R}/preds_sensitivity/{vname}__l2_{lam:g}_seed{seed}.npz", oot=oot, test=pt, alpha=al, w=g.w_)
sel = pd.DataFrame(rows); sel.to_csv(f"{R}/gate_lambda_selection.csv", index=False)
agg = sel.groupby(["variant", "l2"]).b4_logloss.mean().unstack(); print(agg.round(5).to_string())
choice = {v: float(agg.loc[v].idxmin()) for v in VARIANTS}; json.dump(choice, open(f"{R}/gate_lambda_choice.json", "w"), indent=2); print("chosen", choice)
for (vname, seed, lam), d in fitted.items():          # promote validation-selected fits to the main results
    if lam == choice[vname]: np.savez_compressed(f"{R}/preds/{vname}_seed{seed}.npz", **d)
