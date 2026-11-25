"""Robustness v2 (revision Issue A): correctly scaled, evaluation-only test-time perturbations.

Reuses the seed-42 final experts saved by experiments/robustness.py (results/models/*.joblib),
the saved TRUST-GRAF gate weights, and the same gate-standardisation reconstruction, Platt maps
and F1-optimal thresholds (all fitted on out-of-time block 30..34 from results/preds; nothing is
refitted or tuned on perturbed data). The clean scores are asserted to reproduce the clean rows of
results/robustness_results.csv exactly.

Only node features of test-period transactions (all nodes of steps 35..49) are perturbed; the
propagated neighbourhood view and the novelty side-signal are recomputed from the perturbed local
features so the graph expert and the gate see consistent inputs. Training statistics (spread,
CDF, medians) are computed on labelled rows of steps 1..34, as in experiments/noise_diagnostic.py.

Perturbations (3 draws each, rng seeds 0, 1, 2):
  robust_noise   x + N(0,1) * sigma * s_j,  s_j = IQR_j/1.349, fallback MAD_j*1.4826,
                 features with zero robust spread left unperturbed; sigma in {0.05, 0.1, 0.25, 0.5}
  quantile_noise u = F_j(x); u' = clip(u + N(0, q), 0, 1); x' = x + Q_j(u') - Q_j(u),
                 q in {0.01, 0.02, 0.05} (q = 0 is the exact round trip). F_j / Q_j are the
                 piecewise-linear empirical training CDF / quantile function over unique values
                 (mid-rank for ties). For in-range x, Q_j(u) = x, so x' = Q_j(u'); the additive form
                 only keeps test values outside the training range from being clipped to it.
  feature_dropout a random 10% / 25% of the 165 features of each transaction set to the training median
Outputs: results/robustness_v2_raw.csv (one row per draw), results/robustness_v2_summary.csv,
         results/robustness_v2_spread.json
"""
import sys, os, json, joblib, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import preprocessing as pp
from src.fusion import ReliabilityGate, reliability_features
from src.evaluation import Platt, best_f1_threshold
from run_experiments import novelty, BLOCKS
from sklearn.metrics import average_precision_score, f1_score
R = f"{ROOT}/results"; SEED = 42
lam = json.load(open(f"{R}/gate_lambda_choice.json"))["TRUST-GRAF"]
df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values; lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int)
XL, XA = pp.feature_blocks(df); AF = np.hstack([XL, XA]); del XA; NL = XL.shape[1]
M = lambda a, b: lab & (t >= a) & (t <= b); test, tr, b4 = M(35, 49), M(1, 34), M(30, 34)
G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n)
fn = joblib.load(f"{R}/models/LGBM-AF_seed42.joblib"); fg = joblib.load(f"{R}/models/LGBM-AF+G_seed42.joblib")
# ---- gate reconstruction (identical to experiments/robustness.py) ----
d = np.load(f"{R}/preds/TRUST-GRAF_seed{SEED}.npz"); gate = ReliabilityGate(l2=lam); gate.w_ = d["w"]
pa, pb = np.load(f"{R}/preds/LGBM-AF_seed{SEED}.npz"), np.load(f"{R}/preds/LGBM-AF+G_seed{SEED}.npz"); struct = G[:, -3:]
Ro = np.full((n, 7), np.nan)
for (a, b) in BLOCKS[:3]:
    bm = M(a, b); Ro[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], np.column_stack([struct, novelty(XL, t, t < a)])[bm])
fm = M(15, 29); gate.mu_, gate.sd_ = Ro[fm].mean(0), Ro[fm].std(0) + 1e-9; del Ro
KEYS = ["LGBM-AF", "LGBM-AF+G", "TRUST-GRAF"]
oo = {k: np.load(f"{R}/preds/{k}_seed{SEED}.npz")["oot"][b4] for k in KEYS}
pl = {k: Platt().fit(oo[k], yi[b4]) for k in KEYS}; thr = {k: best_f1_threshold(yi[b4], pl[k].transform(oo[k])) for k in KEYS}
te_all = np.where(t >= 35)[0]; post = t[test] >= 43; rows = []


def score(AFp, setting, level, draw):
    """Recompute graph view and novelty from (perturbed) local features, then score test nodes."""
    XLp = AFp[:, :NL]
    Gp, _, _, _ = pp.neighbourhood_view(XLp, src, dst, n); Gt = Gp[test]; del Gp
    nov = novelty(XLp, t, t <= 34)          # fitted on unperturbed steps 1..34, applied to perturbed test rows
    At = AFp[test]
    p1 = fn.predict_proba(At)[:, 1]; p2 = fg.predict_proba(np.hstack([At, Gt]))[:, 1]
    Rt = reliability_features(p1, p2, np.column_stack([Gt[:, -3:], nov[test]])); pg, al = gate.predict(p1, p2, Rt)
    for k, p in (("LGBM-AF", p1), ("LGBM-AF+G", p2), ("TRUST-GRAF", pg)):
        q = pl[k].transform(p)
        rows.append(dict(setting=setting, level=level, draw=draw, model=k, pr_auc=average_precision_score(yi[test], q),
                         f1=f1_score(yi[test], q >= thr[k]), f1_post43=f1_score(yi[test][post], q[post] >= thr[k]),
                         mean_alpha=float(al.mean()) if k == "TRUST-GRAF" else np.nan))


# ---- clean reproduction check ----
score(AF, "clean", 0.0, -1)
ref = pd.read_csv(f"{R}/robustness_results.csv"); ref = ref[ref.setting == "clean"].set_index("model")
for r in rows:
    for c in ("pr_auc", "f1", "f1_post43"):
        assert abs(r[c] - ref.loc[r["model"], c]) < 1e-12, (r["model"], c, r[c], ref.loc[r["model"], c])   # CSV round-trip loses at most the last ulp
print("clean rows reproduce results/robustness_results.csv (|diff| < 1e-12, CSV float round-trip):", [(r["model"], r["pr_auc"], r["f1"]) for r in rows], flush=True)

# ---- training statistics (labelled rows of steps 1..34) ----
Xtr = AF[tr]; q1, med, q3 = np.percentile(Xtr, [25, 50, 75], axis=0); iqr = q3 - q1
mad = np.median(np.abs(Xtr - med), axis=0)
s_rob = np.where(iqr > 0, iqr / 1.349, mad * 1.4826); pert = s_rob > 0
spread = dict(n_features=int(AF.shape[1]), n_zero_iqr=int((iqr == 0).sum()), n_mad_fallback_used=int(((iqr == 0) & (mad > 0)).sum()),
              n_unperturbed_zero_robust_spread=int((~pert).sum()), n_unperturbed_local=int((~pert[:NL]).sum()),
              n_unperturbed_aggregate=int((~pert[NL:]).sum()),
              median_sd_over_robust_spread=float(np.median(Xtr.std(0)[pert] / s_rob[pert])))
json.dump(spread, open(f"{R}/robustness_v2_spread.json", "w"), indent=2); print(spread, flush=True)
# empirical CDF / quantile function over unique training values (mid-rank for ties)
CDF = []
Ntr = Xtr.shape[0]
for j in range(AF.shape[1]):
    s = np.sort(Xtr[:, j]); u, first = np.unique(s, return_index=True); last = np.r_[first[1:], Ntr] - 1
    c = (first + last) / 2.0 / (Ntr - 1); CDF.append((u, c))
del Xtr

for draw in (0, 1, 2):
    for sig in (0.05, 0.1, 0.25, 0.5):
        rng = np.random.default_rng(draw); AFp = AF.copy()
        AFp[te_all] += rng.normal(size=(len(te_all), AF.shape[1])) * (sig * s_rob)   # s_rob = 0 -> unperturbed
        score(AFp, "robust_noise", sig, draw); del AFp
    for qs in (0.0, 0.01, 0.02, 0.05):
        rng = np.random.default_rng(draw); AFp = AF.copy(); E = rng.normal(size=(len(te_all), AF.shape[1])) * qs
        for j, (u, c) in enumerate(CDF):
            x = AF[te_all, j]
            if len(u) == 1: continue                      # constant in training: quantile function is a point
            uu = np.interp(x, u, c); AFp[te_all, j] = x + (np.interp(np.clip(uu + E[:, j], 0, 1), c, u) - np.interp(uu, c, u))
        del E
        if qs == 0.0: assert np.array_equal(AFp, AF), "quantile round trip is not exact"
        score(AFp, "quantile_noise", qs, draw); del AFp
    for frac in (0.10, 0.25):
        rng = np.random.default_rng(draw); AFp = AF.copy(); k = int(round(frac * AF.shape[1]))
        drop = np.argsort(rng.uniform(size=(len(te_all), AF.shape[1])), axis=1)[:, :k]
        sub = AFp[te_all]; np.put_along_axis(sub, drop, med[drop], axis=1); AFp[te_all] = sub; del sub, drop
        score(AFp, "feature_dropout", frac, draw); del AFp
    pd.DataFrame(rows).to_csv(f"{R}/robustness_v2_raw.csv", index=False); print(f"draw {draw} done", flush=True)

raw = pd.DataFrame(rows); raw.to_csv(f"{R}/robustness_v2_raw.csv", index=False)
summ = raw.groupby(["setting", "level", "model"])[["pr_auc", "f1", "f1_post43", "mean_alpha"]].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ["n_draws"] = raw.groupby(["setting", "level", "model"]).size()
order = {"clean": 0, "robust_noise": 1, "quantile_noise": 2, "feature_dropout": 3}
summ = summ.reset_index(); summ["_o"] = summ.setting.map(order); summ = summ.sort_values(["_o", "level", "model"]).drop(columns="_o")
summ.to_csv(f"{R}/robustness_v2_summary.csv", index=False); print(summ.round(4).to_string())
