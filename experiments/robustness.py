"""Robustness of the final seed-42 models to test-time perturbations (no refitting, no choices
made on the test set): Gaussian feature noise (sigma x training s.d.) on all node features, and
random removal of a fraction of test-period edges before recomputing the neighbourhood view.
Also saves the trained final experts as model artefacts."""
import sys, os, json, joblib, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import preprocessing as pp
from src.model import make
from src.fusion import ReliabilityGate, reliability_features
from src.evaluation import Platt
from run_experiments import novelty
from sklearn.metrics import average_precision_score, f1_score
R = f"{ROOT}/results"; os.makedirs(f"{R}/models", exist_ok=True); SEED = 42
hp = json.load(open(f"{R}/hyperparameters.json")); lam = json.load(open(f"{R}/gate_lambda_choice.json"))["TRUST-GRAF"]
df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values; lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int)
XL, XA = pp.feature_blocks(df); AF = np.hstack([XL, XA]); M = lambda a, b: lab & (t >= a) & (t <= b); test, tr, b4 = M(35, 49), M(1, 34), M(30, 34)
G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n)
fn = make("LGBM-AF", SEED, hp["LGBM-AF"]).fit(AF[tr], yi[tr]); fg = make("LGBM-AF+G", SEED, hp["LGBM-AF+G"]).fit(np.hstack([AF[tr], G[tr]]), yi[tr])
joblib.dump(fn, f"{R}/models/LGBM-AF_seed42.joblib"); joblib.dump(fg, f"{R}/models/LGBM-AF+G_seed42.joblib")
d = np.load(f"{R}/preds/TRUST-GRAF_seed{SEED}.npz"); gate = ReliabilityGate(l2=lam); gate.w_ = d["w"]
# reconstruct gate standardisation from its fitting data (out-of-time blocks 15..29)
from run_experiments import BLOCKS
pa, pb = np.load(f"{R}/preds/LGBM-AF_seed{SEED}.npz"), np.load(f"{R}/preds/LGBM-AF+G_seed{SEED}.npz"); struct = G[:, -3:]
Ro = np.full((n, 7), np.nan)
for (a, b) in BLOCKS[:3]:
    bm = M(a, b); Ro[bm] = reliability_features(pa["oot"][bm], pb["oot"][bm], np.column_stack([struct, novelty(XL, t, t < a)])[bm])
fm = M(15, 29); gate.mu_, gate.sd_ = Ro[fm].mean(0), Ro[fm].std(0) + 1e-9
pl = {k: Platt().fit(np.load(f"{R}/preds/{k}_seed{SEED}.npz")["oot"][b4], yi[b4]) for k in ["LGBM-AF", "LGBM-AF+G", "TRUST-GRAF"]}
from src.evaluation import best_f1_threshold
thr = {k: best_f1_threshold(yi[b4], pl[k].transform(np.load(f"{R}/preds/{k}_seed{SEED}.npz")["oot"][b4])) for k in pl}
nov_te = novelty(XL, t, t <= 34); sd = AF[tr].std(0); rng = np.random.default_rng(7); rows = []
def score(AFp, Gp, setting, level):
    p1 = fn.predict_proba(AFp[test])[:, 1]; p2 = fg.predict_proba(np.hstack([AFp[test], Gp[test]]))[:, 1]
    Rt = reliability_features(p1, p2, np.column_stack([Gp[:, -3:], nov_te])[test]); pg, _ = gate.predict(p1, p2, Rt)
    for k, p in (("LGBM-AF", p1), ("LGBM-AF+G", p2), ("TRUST-GRAF", pg)):
        q = pl[k].transform(p); rows.append(dict(setting=setting, level=level, model=k, pr_auc=average_precision_score(yi[test], q),
                                                   f1=f1_score(yi[test], q >= thr[k]), f1_post43=f1_score(yi[test][t[test] >= 43], q[t[test] >= 43] >= thr[k])))
score(AF, G, "clean", 0.0)
for s in (0.1, 0.3, 0.5):
    AFp = AF.copy(); te_i = np.where(t >= 35)[0]; AFp[te_i] += rng.normal(size=(len(te_i), AF.shape[1])) * sd * s
    Gp, _, _, _ = pp.neighbourhood_view(AFp[:, :93], src, dst, n); score(AFp, Gp, "feature_noise", s); del AFp, Gp
for frac in (0.1, 0.3, 0.5):
    keep = ~((t[src] >= 35) & (rng.uniform(size=len(src)) < frac)); Gp, _, _, _ = pp.neighbourhood_view(XL, src[keep], dst[keep], n); score(AF, Gp, "edge_dropout", frac)
out = pd.DataFrame(rows); out.to_csv(f"{R}/robustness_results.csv", index=False); print(out.round(4).to_string())
