"""Per-seed and pooled gate-weight diagnostics on the Elliptic test period (fixes the round-2
finding that earlier percentiles were computed on seed-averaged weights)."""
import os, glob, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f"{ROOT}/results"
m = np.load(f"{R}/preds/_meta.npz"); test = m["test"]; t = m["t"][test]; out = {"per_seed": {}}
A = []
for f in sorted(glob.glob(f"{R}/preds/TRUST-GRAF_seed*.npz")):
    s = f.split("_seed")[1][:-4]; a = np.load(f)["alpha"][test]; A.append(a)
    p10, p90 = np.percentile(a, [10, 90])
    out["per_seed"][s] = dict(mean=float(a.mean()), p10=float(p10), p90=float(p90), pre43=float(a[t < 43].mean()), post43=float(a[t >= 43].mean()))
P = np.concatenate(A); out["pooled_p10"], out["pooled_p90"] = map(float, np.percentile(P, [10, 90]))
ps = out["per_seed"].values(); out["seed_mean_min"] = min(v["mean"] for v in ps); out["seed_mean_max"] = max(v["mean"] for v in ps)
out["max_within_seed_p10_p90_width"] = max(v["p90"] - v["p10"] for v in ps); out["max_abs_pre_post_shift"] = max(abs(v["pre43"] - v["post43"]) for v in ps)
json.dump(out, open(f"{R}/gate_diagnostics_v2.json", "w"), indent=2); print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))
