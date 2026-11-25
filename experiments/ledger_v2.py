"""Revision ledger: appends macros and writes tables for the revision experiments
(degree-preserving control, disjoint-block tuning, robust-scale robustness, GNN baselines,
cluster/seed statistics, second dataset). Run after experiments/ledger.py."""
import os, re, json, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{ROOT}/results"; T = f"{ROOT}/manuscript/tables"
out = []
def mac(n, v): out.append(f"\\newcommand{{\\{n}}}{{{v}}}")
def fp(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
# ---- degree-preserving control ----
dp = pd.read_csv(f"{R}/dp_control_results.csv"); rw = json.load(open(f"{R}/dp_control_rewiring.json"))
m = dp[dp.row_type == "method"].set_index("name")
mac("DPPrAuc", f"{m.loc['LGBM-AF+Gdp','pr_auc_mean']:.3f}"); mac("DPFOne", f"{m.loc['LGBM-AF+Gdp','f1_mean']:.3f}")
c = dp[dp.row_type == "comparison"].set_index("name")
for nm, k in [("LGBM-AF+G vs LGBM-AF+Gdp", "DPRealVsDp"), ("LGBM-AF+Gdp vs LGBM-AF", "DPDpVsNode")]:
    r = c.loc[nm]
    for x in ["pr_auc", "f1"]:
        kk = "PrAuc" if x == "pr_auc" else "FOne"
        mac(f"{k}D{kk}", f"{r['delta_'+x]:+.4f}"); mac(f"{k}Lo{kk}", f"{r['ci_lo_'+x]:+.4f}"); mac(f"{k}Hi{kk}", f"{r['ci_hi_'+x]:+.4f}")
keys = {k.lower(): v for k, v in rw.items() if not isinstance(v, (dict, list))}
for cand in ["frac_original_directed_edges_retained"]:
    if cand in keys: mac("DPRetainedPct", f"{100*float(keys[cand]):.2f}"); break
# ---- disjoint-block tuning ----
tv = pd.read_csv(f"{R}/tuning_v2_summary.csv").set_index("method")
NM2 = {"TRUST-GRAF-T2": "OTRG", "A-static-gate-T2": "Constant gate", "A-average-T2": "Equal-weight average", "LGBM-AF+G-T2": "LightGBM, all features + graph view", "LGBM-AF-T2": "LightGBM, all features"}
L = ["\\begin{tabular}{lcccc}", "\\toprule", "Model (tuned on steps 25--29) & PR-AUC & F1 & F1 (steps 43--49) & ECE \\\\", "\\midrule"]
for mm in ["TRUST-GRAF-T2", "A-static-gate-T2", "A-average-T2", "LGBM-AF+G-T2", "LGBM-AF-T2"]:
    r = tv.loc[mm]; L.append(f"{NM2[mm]} & {r.pr_auc_mean:.3f} $\\pm$ {r.pr_auc_std:.3f} & {r.f1_mean:.3f} $\\pm$ {r.f1_std:.3f} & {r.f1_post43_mean:.3f} & {r.ece_mean:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/tuning_v2.tex", "w").write("\n".join(L) + "\n")
for mm, k in [("TRUST-GRAF-T2", "TTwoOTRG"), ("A-average-T2", "TTwoAvg"), ("A-static-gate-T2", "TTwoStatic"), ("LGBM-AF+G-T2", "TTwoGraph"), ("LGBM-AF-T2", "TTwoNode")]:
    mac(k + "PrAuc", f"{tv.loc[mm,'pr_auc_mean']:.3f}"); mac(k + "FOne", f"{tv.loc[mm,'f1_mean']:.3f}")
# ---- robust-scale robustness ----
rb = pd.read_csv(f"{R}/robustness_v2_summary.csv"); sp = json.load(open(f"{R}/robustness_v2_spread.json"))
mac("RobZeroSpread", sp["n_unperturbed_zero_robust_spread"]); mac("RobSdOverSpread", f"{sp['median_sd_over_robust_spread']:.1f}")
nm = {"clean": "None", "robust_noise": "Gaussian noise ($\\times$ robust spread)", "quantile_noise": "Quantile-space noise (s.d.)", "feature_dropout": "Features set to training median (fraction)"}
L = ["\\begin{tabular}{llccc}", "\\toprule", "Perturbation & Level & OTRG & LightGBM, all features & LightGBM, all + graph \\\\", "\\midrule"]
for (s, l), g in rb.groupby(["setting", "level"], sort=False):
    if s == "quantile_noise" and l == 0: continue
    v = g.set_index("model").pr_auc_mean; L.append(f"{nm.get(s, s)} & {l:g} & {v['TRUST-GRAF']:.3f} & {v['LGBM-AF']:.3f} & {v['LGBM-AF+G']:.3f} \\\\")
    if s in ("robust_noise", "quantile_noise", "feature_dropout"):
        key = {"robust_noise": "RobN", "quantile_noise": "RobQ", "feature_dropout": "RobD"}[s] + {0.01: "a", 0.02: "b", 0.05: "c", 0.1: "d", 0.25: "e", 0.5: "f"}.get(l, "z")
        for mo, kk in [("TRUST-GRAF", "OTRG"), ("LGBM-AF", "Node"), ("LGBM-AF+G", "Graph")]: mac(key + kk, f"{v[mo]:.3f}")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/robustness.tex", "w").write("\n".join(L) + "\n")
# ---- optional blocks filled by later scripts (GNN, stats v2, second dataset) ----
if os.path.exists(f"{ROOT}/experiments/ledger_v2_extra.py"):
    exec(open(f"{ROOT}/experiments/ledger_v2_extra.py").read())
with open(f"{ROOT}/manuscript/numbers.tex", "a") as f: f.write("% ---- revision macros (experiments/ledger_v2.py) ----\n" + "\n".join(out) + "\n")
print(len(out), "revision macros appended")
