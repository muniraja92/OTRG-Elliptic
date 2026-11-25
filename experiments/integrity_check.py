"""Automated integrity checks for the research package (Agent 10 role). Exit code 1 on any failure."""
import os, re, sys, glob, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import Platt, best_f1_threshold, metrics
from sklearn.metrics import average_precision_score
R, MS = f"{ROOT}/results", f"{ROOT}/manuscript"; ok, fails = [], []
def check(name, cond, detail=""):
    (ok if cond else fails).append(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
tex = open(f"{MS}/main.tex").read() + open(f"{MS}/body.tex").read() + open(f"{MS}/main_peerj.tex").read(); nums = open(f"{MS}/numbers.tex").read()
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums))
std = set("documentclass usepackage input journal begin end title author cortext address abstract sep keyword section subsection label caption centering includegraphics textwidth hfill citep item emph ref url bibliographystyle bibliography mathbf mathcal mathrm top log sigma alpha delta ell nu lambda dots quad qquad big Big frac sum in cup le ge pm times tilde top Delta ldots corref newcommand textbf hline toprule midrule bottomrule infty mid otimes approx left right cdot langle rangle mathbb operatorname".split())
used = set(re.findall(r"\\([A-Za-z]+)", tex)) - std
custom = {u for u in used if u[0].isupper() or u.startswith(("Vs", "Sec", "Gate", "Fit", "Sens"))}
missing = sorted(u for u in custom if u not in defined and u not in {"Gate", "RaggedRight"})
check("all result macros in main.tex are defined in numbers.tex", not missing, str(missing[:10]))
bib = open(f"{MS}/references.bib").read(); keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set(k.strip() for c in re.findall(r"\\cite[pt]\{([^}]+)\}", tex) for k in c.split(","))
check("every citation key exists in references.bib", cites <= keys, str(cites - keys)); check("no unused bibliography entries", keys <= cites, str(keys - cites))
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex): check(f"figure exists {os.path.basename(f)}", os.path.exists(os.path.normpath(os.path.join(MS, f))))
for t in re.findall(r"\\input\{(tables/[^}]+)\}", tex): check(f"table exists {t}", os.path.exists(f"{MS}/{t}.tex"))
check("no file extensions such as .csv in manuscript text", not re.search(r"\b\w+\.(csv|npz|py|json)\b", tex))
body = tex.split("\\begin{itemize}")[0] + tex.split("\\end{itemize}")[1]
check("bullet points only in the contributions list", body.count("\\item") == 0 and open(f"{MS}/body.tex").read().count("\\begin{itemize}") == 1)
# --- independent recomputation of headline numbers from raw prediction files ---
meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]; yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34)
S = pd.read_csv(f"{R}/metrics_summary.csv", index_col=0)
for m in ["TRUST-GRAF", "LGBM-AF", "LGBM-AF+G", "RF", "XGB"]:
    vals_ap, vals_f1 = [], []
    for f in sorted(glob.glob(f"{R}/preds/{m}_seed*.npz")):
        d = np.load(f); pl = Platt().fit(d["oot"][cal], yi[cal]); q = pl.transform(d["test"][test]); thr = best_f1_threshold(yi[cal], pl.transform(d["oot"][cal]))
        vals_ap.append(average_precision_score(yi[test], q)); vals_f1.append(metrics(yi[test], q, thr)["f1"])
    check(f"recomputed PR-AUC {m}", abs(np.mean(vals_ap) - S.loc[m, "pr_auc_mean"]) < 1e-9, f"{np.mean(vals_ap):.4f}")
    check(f"recomputed F1 {m}", abs(np.mean(vals_f1) - S.loc[m, "f1_mean"]) < 1e-9, f"{np.mean(vals_f1):.4f}")
    check(f"macro matches summary {m} PR-AUC", f"{S.loc[m,'pr_auc_mean']:.3f}" in nums)
# --- test set untouched by selection: thresholds and Platt only from block 30..34 ---
src_txt = open(f"{ROOT}/experiments/statistical_tests.py").read()
check("Platt/threshold fitted on calibration block only", "Platt().fit(po, yi[cal])" in src_txt and "best_f1_threshold(yi[cal]" in src_txt)
run_txt = open(f"{ROOT}/run_experiments.py").read(); check("gate fitted on steps 15..29 only", "fit_m = M(15, 29)" in run_txt)
ref = open(f"{ROOT}/experiments/refit_fusion.py").read(); check("penalty selected on block 30..34 log-loss", "b4_logloss" in ref and "idxmin" in ref)
# --- dataset facts ---
A = json.load(open(f"{ROOT}/data/dataset_audit.json")); check("no cross-step edges", A["edges_crossing_time_steps"] == 0); check("node/edge counts", A["n_nodes"] == 203769 and A["n_edges"] == 234355)
cls = pd.read_csv(f"{ROOT}/data/class_by_timestep.csv", index_col=0); check("post-closure illicit count 169", int(cls.loc[43:49, "1"].sum()) == 169)
# --- hand-written report numbers ---
ST = pd.read_csv(f"{R}/statistical_results.csv").set_index("comparison")
r = ST.loc["TRUST-GRAF vs LGBM-AF"]; check("report: +0.0010 PR-AUC vs LGBM-AF", f"{r.delta_pr_auc:+.4f}" == "+0.0010", f"{r.delta_pr_auc:+.4f}")
check("report: Holm p about 0.10 vs LGBM-AF", 0.08 <= r.p_boot_pr_auc_holm <= 0.12, f"{r.p_boot_pr_auc_holm:.3f}")
fus = ST.loc[[f"TRUST-GRAF vs {k}" for k in ["A-average", "Stack-LR", "A-static-gate", "A-structural-only-gate"]]]
check("report: fusion variants within 0.001 PR-AUC", fus.delta_pr_auc.abs().max() <= 0.001, f"{fus.delta_pr_auc.abs().max():.4f}")
check("report: fusion variants within 0.007 F1", fus.delta_f1.abs().max() <= 0.007, f"{fus.delta_f1.abs().max():.4f}")
check("report: fusion variants not significant (bootstrap PR-AUC)", (fus.p_boot_pr_auc_holm > 0.05).all())

# --- revision claims (v2) ---
SS2 = S
check("post-closure F1 < 0.04 for all tree models", all(SS2.loc[m, "f1_post43_mean"] < 0.04 for m in ["TRUST-GRAF","LGBM-AF","LGBM-AF+G","RF","RF-Weber","XGB","LGBM-LF"]))
check("GNNs far below LightGBM (PR-AUC gap > 0.1)", all(SS2.loc[m, "pr_auc_mean"] < SS2.loc["LGBM-AF","pr_auc_mean"] - 0.1 for m in ["GCN","GraphSAGE","Mowst"]))
check("GCN, GraphSAGE and Mowst 5 seeds", all(SS2.loc[m,"n_seeds"]==5 for m in ["GCN","GraphSAGE","Mowst"]))
rb2 = pd.read_csv(f"{R}/robustness_v2_summary.csv"); pert = rb2[rb2.setting != "clean"]
best = pert.loc[pert.groupby(["setting","level"]).pr_auc_mean.idxmax()]
check("graph-augmented model most robust in every perturbation", (best.model == "LGBM-AF+G").all(), str(best[best.model!="LGBM-AF+G"][["setting","level","model"]].values.tolist()))
q = pert[(pert.setting=="quantile_noise")&(pert.level==0.05)].set_index("model").pr_auc_mean; clean = rb2[rb2.setting=="clean"].set_index("model").pr_auc_mean
check("quantile noise 0.05 changes PR-AUC < 0.005", (abs(q-clean)<0.005).all())
d25 = pert[(pert.setting=="feature_dropout")&(pert.level==0.25)].set_index("model").pr_auc_mean
check("25% dropout costs 0.02-0.05", ((clean-d25)>=0.02).all() and ((clean-d25)<=0.05).all(), str((clean-d25).round(3).to_dict()))
SDs = pd.read_csv(f"{R}/statistical_v2_seed.csv").set_index(["comparison","metric"])
check("graph vs node F1 positive in 5/5 seeds", SDs.loc[("LGBM-AF+G vs LGBM-AF","f1"),"n_positive"]==5)
check("real vs degree-preserving positive 5/5 both metrics", SDs.loc[("LGBM-AF+G vs LGBM-AF+Gdp","f1"),"n_positive"]==5 and SDs.loc[("LGBM-AF+G vs LGBM-AF+Gdp","pr_auc"),"n_positive"]==5)
for g in ["A-shuffled-graph","A-dp-graph-gate"]:
    al = np.mean([np.nanmean(np.load(f)["alpha"][test]) for f in glob.glob(f"{R}/preds/{g}_seed*.npz")])
    check(f"gate over rewired expert puts ~all weight on node expert ({g})", al > 0.99, f"{al:.4f}")
TO = pd.read_csv(f"{R}/statistical_v2_tost.csv"); T2 = TO[TO.scheme=="2stage"]
ok_eq = T2[T2.comparison.isin(["TRUST-GRAF vs A-static-gate","TRUST-GRAF vs Stack-LR"])].p_tost_holm.lt(0.05).all()
check("equivalent to constant gate and stacking (both metrics)", ok_eq)
check("equal average: PR-AUC equivalent, F1 not", T2[(T2.comparison=="TRUST-GRAF vs A-average")&(T2.metric=="pr_auc")].p_tost_holm.iloc[0]<0.05 and T2[(T2.comparison=="TRUST-GRAF vs A-average")&(T2.metric=="f1")].p_tost_holm.iloc[0]>=0.05)
K2 = pd.read_csv(f"{R}/statistical_v2_key.csv").set_index("comparison")
check("fusion variants not significant after family Holm", (K2.loc[["TRUST-GRAF vs A-static-gate","TRUST-GRAF vs Stack-LR","TRUST-GRAF vs A-average"], ["p_pr_auc_2stage_holm_family","p_f1_2stage_holm_family"]]>0.05).all().all())
check("OTRG vs node-only not significant after family Holm", (K2.loc["TRUST-GRAF vs LGBM-AF", ["p_pr_auc_2stage_holm_family","p_f1_2stage_holm_family"]]>0.05).all())
check("real vs DP not significant after family Holm (as stated)", (K2.loc["LGBM-AF+G vs LGBM-AF+Gdp", ["p_pr_auc_2stage_holm_family","p_f1_2stage_holm_family"]]>0.05).all())
SDD = pd.read_csv(f"{R}/second_dataset/summary.csv")
y = SDD[SDD.dataset=="yelp"].set_index("method").pr_auc_mean; a = SDD[SDD.dataset=="amazon"].set_index("method").pr_auc_mean
check("stacking >= OTRG on both second datasets", y["Stack-LR"] >= y["OTRG"] and a["Stack-LR"] >= a["OTRG"])

GSm = pd.read_csv(f"{R}/second_dataset/gnn_summary.csv")
for d in ["yelp","amazon"]:
    lg = SDD[(SDD.dataset==d)&(SDD.method=="LGBM-node")].pr_auc_mean.iloc[0]
    check(f"second-dataset GNNs below node-only LightGBM ({d})", (GSm[GSm.dataset==d].pr_auc_mean < lg).all())
DDs = pd.read_csv(f"{R}/dp_draws_summary.csv"); adr = DDs[DDs.row_type=="across_draws"].iloc[0]
check("real view ahead in all draw x seed pairs", adr.n_seeds_real_ahead_pr_auc == adr.n_pairs == adr.n_seeds_real_ahead_f1 == 15)
HRs = pd.read_csv(f"{R}/headroom_summary.csv"); check("post-closure oracle F1 <= 0.11", HRs[HRs.period=="post"].f1_oracle.max() <= 0.11)
print("\n".join(ok + fails)); print(f"\n{len(ok)} passed, {len(fails)} failed"); sys.exit(1 if fails else 0)
