# Executed inside experiments/ledger_v2.py (shares its namespace: out, mac, fp, R, T, pd, np).
LBL = {"TRUST-GRAF": "OTRG", "LGBM-AF": "LightGBM, all features", "LGBM-AF+G": "LightGBM + graph view", "A-static-gate": "constant gate",
       "Stack-LR": "logistic stacking", "A-average": "equal average", "LGBM-AF+Gshuf": "LightGBM + uniformly rewired view",
       "LGBM-AF+Gdp": "LightGBM + degree-preserving rewired view", "A-dp-graph-gate": "gate over node and degree-preserving experts"}
K = pd.read_csv(f"{R}/statistical_v2_key.csv"); SD = pd.read_csv(f"{R}/statistical_v2_seed.csv")
pos = {(r.comparison, r.metric): int(r.n_positive) for r in SD.itertuples()}
def fb(p): return "$\\le$0.007" if p <= 0.0072 + 1e-9 else f"{p:.3f}"
L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Comparison & $\\Delta$PR-AUC [95\\% CI] & $p_{\\mathrm{Holm}}$ & Seeds $+$ & $\\Delta$F1 [95\\% CI] & $p_{\\mathrm{Holm}}$ & Seeds $+$ \\\\", "\\midrule"]
for r in K.itertuples():
    if r.comparison.startswith("A-dp-graph-gate"): continue
    a, b = r.comparison.split(" vs ")
    L.append(f"{LBL[a]} vs.\\ {LBL[b]} & {r.delta_pr_auc:+.4f} [{r.ci_lo_pr_auc_2stage:+.4f}, {r.ci_hi_pr_auc_2stage:+.4f}] & {fb(r.p_pr_auc_2stage_holm_family)} & {pos[(r.comparison,'pr_auc')]}/5 & {r.delta_f1:+.4f} [{r.ci_lo_f1_2stage:+.4f}, {r.ci_hi_f1_2stage:+.4f}] & {fb(r.p_f1_2stage_holm_family)} & {pos[(r.comparison,'f1')]}/5 \\\\")
    k = "K" + "".join(ch for ch in (a + "Vs" + b).replace("+G", "G").replace("shuf", "Shuf").replace("dp", "Dp") if ch.isalpha())
    mac(k + "DPrAuc", f"{r.delta_pr_auc:+.4f}"); mac(k + "LoPrAuc", f"{r.ci_lo_pr_auc_2stage:+.4f}"); mac(k + "HiPrAuc", f"{r.ci_hi_pr_auc_2stage:+.4f}"); mac(k + "PPrAuc", fb(r.p_pr_auc_2stage_holm_family))
    mac(k + "DFOne", f"{r.delta_f1:+.4f}"); mac(k + "LoFOne", f"{r.ci_lo_f1_2stage:+.4f}"); mac(k + "HiFOne", f"{r.ci_hi_f1_2stage:+.4f}"); mac(k + "PFOne", fb(r.p_f1_2stage_holm_family))
    mac(k + "SeedsPrAuc", pos[(r.comparison, 'pr_auc')]); mac(k + "SeedsFOne", pos[(r.comparison, 'f1')])
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/stats_key_v2.tex", "w").write("\n".join(L) + "\n")
TO = pd.read_csv(f"{R}/statistical_v2_tost.csv"); TO = TO[TO.scheme == "2stage"]
L = ["\\begin{tabular}{llccccc}", "\\toprule", "OTRG versus & Metric & Margin & $\\Delta$ [90\\% CI] & $p_{\\mathrm{TOST}}$ (Holm) & Equivalent & Smallest margin \\\\", "\\midrule"]
for r in TO.itertuples():
    b = r.comparison.split(" vs ")[1]
    L.append(f"{LBL[b]} & {'PR-AUC' if r.metric=='pr_auc' else 'F1'} & $\\pm${r.margin:g} & {r.delta:+.4f} [{r.ci90_lo:+.4f}, {r.ci90_hi:+.4f}] & {r.p_tost_holm:.3f} & {'yes' if r.p_tost_holm < 0.05 else 'no'} & {max(abs(r.ci90_lo), abs(r.ci90_hi)):.4f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/tost_v2.tex", "w").write("\n".join(L) + "\n")
mac("TostAvgFOneP", f"{TO[(TO.comparison=='TRUST-GRAF vs A-average')&(TO.metric=='f1')].p_tost_holm.iloc[0]:.3f}")
C = pd.read_csv(f"{R}/statistical_v2_components.csv"); mac("NCommunities", 701)
# ---- GNN seed counts ----
S = pd.read_csv(f"{R}/metrics_summary.csv", index_col=0)
for m in ["GCN", "GraphSAGE", "Mowst"]:
    if m in S.index: mac(m + "NSeeds", int(S.loc[m, "n_seeds"]))
# ---- second dataset ----
SS = pd.read_csv(f"{R}/second_dataset/summary.csv"); ST2 = pd.read_csv(f"{R}/second_dataset/stats.csv")
NMS = {"LGBM-node": "LightGBM, node features", "LGBM-deg": "LightGBM + degree features only", "LGBM-G": "LightGBM + graph view", "LGBM-Grew": "LightGBM + rewired view (configuration model)",
       "OTRG": "OTRG", "A-constant-gate": "Constant gate", "Stack-LR": "Logistic stacking", "A-average": "Equal-weight average"}
L = ["\\begin{tabular}{lcccc}", "\\toprule", "& \\multicolumn{2}{c}{YelpChi} & \\multicolumn{2}{c}{Amazon} \\\\", "Model & PR-AUC & F1 & PR-AUC & F1 \\\\", "\\midrule"]
for m, lab in NMS.items():
    y = SS[(SS.dataset == "yelp") & (SS.method == m)].iloc[0]; a = SS[(SS.dataset == "amazon") & (SS.method == m)].iloc[0]
    L.append(f"{lab} & {y.pr_auc_mean:.3f} $\\pm$ {y.pr_auc_std:.3f} & {y.f1_mean:.3f} & {a.pr_auc_mean:.3f} $\\pm$ {a.pr_auc_std:.3f} & {a.f1_mean:.3f} \\\\")
    for d, row in (("Yelp", y), ("Amz", a)):
        mac(f"SD{d}{''.join(ch for ch in m if ch.isalpha())}PrAuc", f"{row.pr_auc_mean:.3f}")
GS = pd.read_csv(f"{R}/second_dataset/gnn_summary.csv")
for gm, glab in [("GraphSAGE", "GraphSAGE (relation union)"), ("RSAGE", "Relation-aware GraphSAGE")]:
    y = GS[(GS.dataset == "yelp") & (GS.method == gm)].iloc[0]; a = GS[(GS.dataset == "amazon") & (GS.method == gm)].iloc[0]
    L.append(f"{glab} & {y.pr_auc_mean:.3f} $\\pm$ {y.pr_auc_std:.3f} & {y.f1_mean:.3f} & {a.pr_auc_mean:.3f} $\\pm$ {a.pr_auc_std:.3f} & {a.f1_mean:.3f} \\\\")
    for d, row in (("Yelp", y), ("Amz", a)): mac(f"SD{d}{gm}PrAuc", f"{row.pr_auc_mean:.3f}")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/second_dataset.tex", "w").write("\n".join(L) + "\n")
for r in ST2[ST2.metric == "pr_auc"].itertuples():
    k = f"SD{'Yelp' if r.dataset=='yelp' else 'Amz'}{''.join(ch for ch in r.a if ch.isalpha())}Vs{''.join(ch for ch in r.b if ch.isalpha())}"
    mac(k + "D", f"{r.seed_mean_diff:+.4f}"); mac(k + "Seeds", int(r.seeds_a_better)); mac(k + "P", f"={r.paired_t_p_holm:.3f}" if r.paired_t_p_holm >= 0.001 else "<0.001")
import json as _j
for d in ("yelp", "amazon"):
    pass
GD = _j.load(open(f"{R}/gate_diagnostics_v2.json"))
mac("GatePoolTen", f"{GD['pooled_p10']:.2f}"); mac("GatePoolNinety", f"{GD['pooled_p90']:.2f}")
mac("GateSeedMin", f"{GD['seed_mean_min']:.2f}"); mac("GateSeedMax", f"{GD['seed_mean_max']:.2f}")
mac("GateWithinWidth", f"{GD['max_within_seed_p10_p90_width']:.3f}"); mac("GatePrePostShift", f"{GD['max_abs_pre_post_shift']:.3f}")
# ---- headroom analysis ----
HR = pd.read_csv(f"{R}/headroom_summary.csv")
NMH = {"TRUST-GRAF": "OTRG", "LGBM-AF": "LightGBM, all features", "LGBM-AF+G": "LightGBM + graph view", "XGB": "XGBoost", "RF": "Random forest (tuned)"}
L = ["\\begin{tabular}{lcccccc}", "\\toprule", "& \\multicolumn{3}{c}{Steps 35--42} & \\multicolumn{3}{c}{Steps 43--49} \\\\", "Model & PR-AUC & F1 & Oracle F1 & PR-AUC & F1 & Oracle F1 \\\\", "\\midrule"]
for mm, lab in NMH.items():
    a = HR[(HR.method == mm) & (HR.period == "pre")].iloc[0]; b = HR[(HR.method == mm) & (HR.period == "post")].iloc[0]
    L.append(f"{lab} & {a.pr_auc:.3f} & {a.f1_val_threshold:.3f} & {a.f1_oracle:.3f} & {b.pr_auc:.3f} & {b.f1_val_threshold:.3f} & {b.f1_oracle:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{T}/headroom.tex", "w").write("\n".join(L) + "\n")
pre, post = HR[HR.period == "pre"], HR[HR.period == "post"]
mac("HRPrePrevPct", f"{100*pre.prevalence.iloc[0]:.1f}"); mac("HRPostPrevPct", f"{100*post.prevalence.iloc[0]:.1f}")
mac("HRPostAPMin", f"{post.pr_auc.min():.3f}"); mac("HRPostAPMax", f"{post.pr_auc.max():.3f}")
mac("HRPostROCMin", f"{post.roc_auc.min():.2f}"); mac("HRPostROCMax", f"{post.roc_auc.max():.2f}")
mac("HRPostOracleMax", f"{post.f1_oracle.max():.2f}"); mac("HRPreOracleMax", f"{pre.f1_oracle.max():.3f}"); mac("HRPreFOneMax", f"{pre.f1_val_threshold.max():.3f}")
mac("HRPreGapMax", f"{(pre.f1_oracle - pre.f1_val_threshold).max():.3f}")
# ---- repeated degree-preserving draws ----
DD = pd.read_csv(f"{R}/dp_draws_summary.csv"); ad = DD[DD.row_type == "across_draws"].iloc[0]
mac("DrawsDPrAuc", f"{ad.delta_pr_auc:+.4f}"); mac("DrawsDFOne", f"{ad.delta_f1:+.4f}")
mac("DrawsPrAucMin", f"{ad.delta_pr_auc_min:+.4f}"); mac("DrawsPrAucMax", f"{ad.delta_pr_auc_max:+.4f}")
mac("DrawsFOneMin", f"{ad.delta_f1_min:+.4f}"); mac("DrawsFOneMax", f"{ad.delta_f1_max:+.4f}")
mac("DrawsAheadPrAuc", int(ad.n_seeds_real_ahead_pr_auc)); mac("DrawsAheadFOne", int(ad.n_seeds_real_ahead_f1)); mac("DrawsPairs", int(ad.n_pairs))
