"""Repeated degree-preserving rewiring draws (round-2 reviewer: single rewiring realisation).

Reuses degree_preserving_rewire / verify / ap_fast from experiments/degree_preserving_control.py
(imported, not modified). For rewiring rng seeds 7 and 99 (the existing draw is rng 2024):
rewire, verify degrees with the same asserts, build AF+Gdp, and for model seeds 42 1 7 123 2024
train LGBM with the FIXED hyperparameters of results/dp_control_hyperparameters.json (no re-tuning):
  * fit on labelled steps 1-29 -> OOT prediction on block 30-34 (for Platt + threshold),
  * final fit on labelled steps 1-34 -> test steps 35-49.
Predictions go to results/preds_dp_draws/ (NOT results/preds, so statistical_tests.py ignores them).

Stages (heavy ones under `flock /tmp/cpu.lock`):
  train   draws 7 and 99; plus a reproducibility check that re-runs draw 2024 / seed 42 and
          compares with the existing results/preds/LGBM-AF+Gdp_seed42.npz.
  score   after the prediction files are frozen (digests in results/dp_draws_frozen_preds.json):
          Platt + best_f1_threshold on OOT 30-34, test PR-AUC/F1 per draw and seed, and the
          LGBM-AF+G minus LGBM-AF+Gdp gaps across the three draws.
"""
from __future__ import annotations
import sys, os, json, time, logging, argparse, hashlib
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.makedirs(f"{ROOT}/logs", exist_ok=True)
# configure logging BEFORE importing degree_preserving_control so its basicConfig is a no-op
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(f"{ROOT}/logs/dp_draws.log"), logging.StreamHandler()])
log = logging.info
from degree_preserving_control import degree_preserving_rewire, verify, ap_fast  # noqa: E402

R = f"{ROOT}/results"; PD = f"{R}/preds_dp_draws"; os.makedirs(PD, exist_ok=True)
SEEDS = [42, 1, 7, 123, 2024]; NEW_DRAWS = [7, 99]; ALL_DRAWS = [2024, 7, 99]; METHOD = "LGBM-AF+Gdp"


def pred_path(draw, seed):
    if draw == 2024:
        return f"{R}/preds/{METHOD}_seed{seed}.npz"          # existing draw, produced by degree_preserving_control.py
    return f"{PD}/{METHOD}_rng{draw}_seed{seed}.npz"


def stage_train():
    from src import preprocessing as pp
    from src.model import make
    T0 = time.time()
    hp = json.load(open(f"{R}/dp_control_hyperparameters.json"))[METHOD]; log(f"fixed hyperparameters {hp}")
    df, src, dst = pp.load(f"{ROOT}/data"); n = len(df); t = df.time_step.values; y = df.y.values
    lab = ~np.isnan(y); yi = np.nan_to_num(y).astype(int); M = lambda a, b: lab & (t >= a) & (t <= b)
    test, b4 = M(35, 49), M(30, 34)
    XL, XAgg = pp.feature_blocks(df)
    G, _, _, _ = pp.neighbourhood_view(XL, src, dst, n); logdeg_real = G[:, -3:-1].copy(); del G
    rew_info, runtime = {}, []

    def run_draw(draw, seeds, out_fn):
        rng = np.random.default_rng(draw); t1 = time.time()
        s2, d2, rstats = degree_preserving_rewire(src, dst, t, rng)
        ver = verify(src, dst, s2, d2, t, n)            # asserts in-/out-degree identical and same step
        ver.update(rstats, rewire_seconds=time.time() - t1, rng_seed=draw)
        ver["edges_sha256"] = hashlib.sha256(np.stack([s2, d2]).astype(np.int64).tobytes()).hexdigest()
        Gdp, _, _, _ = pp.neighbourhood_view(XL, s2, d2, n)
        assert np.allclose(logdeg_real, Gdp[:, -3:-1]), "log-degree features must be identical"
        ver["logdeg_identical_to_real"] = True
        log(f"draw rng{draw} verified: {json.dumps(ver)}")
        X = np.hstack([XL, XAgg, Gdp]); del Gdp
        tr29, tr34 = M(1, 29), M(1, 34)
        for seed in seeds:
            oot = np.full(n, np.nan); pt = np.full(n, np.nan)
            t1 = time.time(); mdl = make(METHOD, seed, hp).fit(X[tr29], yi[tr29]); oot[b4] = mdl.predict_proba(X[b4])[:, 1]; f29 = time.time() - t1
            t1 = time.time(); mdl = make(METHOD, seed, hp).fit(X[tr34], yi[tr34]); pt[test] = mdl.predict_proba(X[test])[:, 1]; f34 = time.time() - t1
            out_fn(draw, seed, oot, pt)
            runtime.append(dict(draw=draw, seed=seed, fit29_s=f29, fit34_s=f34)); log(f"draw rng{draw} seed {seed} done ({f29:.1f}s + {f34:.1f}s)")
        del X
        return ver

    def save(draw, seed, oot, pt):
        np.savez_compressed(pred_path(draw, seed), oot=oot, test=pt)

    for draw in NEW_DRAWS:
        rew_info[f"rng{draw}"] = run_draw(draw, SEEDS, save)

    # reproducibility check: draw 2024 / seed 42 must reproduce the existing file (oot block 30-34 + test)
    chk = {}
    def compare(draw, seed, oot, pt):
        ref = np.load(pred_path(2024, seed))
        chk.update(seed=seed, max_abs_diff_oot_30_34=float(np.nanmax(np.abs(ref["oot"][b4] - oot[b4]))),
                   max_abs_diff_test=float(np.nanmax(np.abs(ref["test"][test] - pt[test]))))
    ver24 = run_draw(2024, [42], compare)
    ref_sha = json.load(open(f"{R}/dp_control_rewiring.json"))["edges_sha256"]
    chk["rewired_edges_sha256_matches_existing"] = ver24["edges_sha256"] == ref_sha
    rew_info["rng2024_reproduction_check"] = chk; log(f"reproduction check: {chk}")

    json.dump(rew_info, open(f"{R}/dp_draws_rewiring.json", "w"), indent=2)
    pd.DataFrame(runtime).to_csv(f"{R}/dp_draws_runtime.csv", index=False)
    dig = {os.path.basename(pred_path(d, s)): hashlib.sha256(open(pred_path(d, s), "rb").read()).hexdigest()
           for d in NEW_DRAWS for s in SEEDS}
    json.dump(dig, open(f"{R}/dp_draws_frozen_preds.json", "w"), indent=2); log(f"ALL DONE {time.time()-T0:.0f}s")


def stage_score():
    from src.evaluation import best_f1_threshold, Platt, metrics
    for f, h in json.load(open(f"{R}/dp_draws_frozen_preds.json")).items():
        assert hashlib.sha256(open(f"{PD}/{f}", "rb").read()).hexdigest() == h, f"prediction file changed: {f}"
    meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]
    yi = np.nan_to_num(y).astype(int); cal = (~np.isnan(y)) & (t >= 30) & (t <= 34); yt, tt = yi[test], t[test]

    def score(path):
        d = np.load(path); po, pt = d["oot"][cal], d["test"][test]
        assert not np.isnan(po).any() and not np.isnan(pt).any()
        pl = Platt().fit(po, yi[cal]); thr = best_f1_threshold(yi[cal], pl.transform(po))
        m = metrics(yt, pl.transform(pt), thr, tt); return m["pr_auc"], m["f1"]

    rows = []
    for s in SEEDS:
        a, f = score(f"{R}/preds/LGBM-AF+G_seed{s}.npz"); rows.append(dict(view="real", draw="real", seed=s, pr_auc=a, f1=f))
        for d in ALL_DRAWS:
            a, f = score(pred_path(d, s)); rows.append(dict(view="dp", draw=d, seed=s, pr_auc=a, f1=f))
    per = pd.DataFrame(rows)
    real = per[per.view == "real"].set_index("seed")
    dp = per[per.view == "dp"].copy()
    dp["delta_pr_auc_real_minus_dp"] = dp.seed.map(real.pr_auc) - dp.pr_auc
    dp["delta_f1_real_minus_dp"] = dp.seed.map(real.f1) - dp.f1
    pd.concat([per[per.view == "real"], dp]).to_csv(f"{R}/dp_draws_metrics_per_seed.csv", index=False)

    # sanity: existing draw must match results/dp_control_metrics_per_seed.csv
    old = pd.read_csv(f"{R}/dp_control_metrics_per_seed.csv"); old = old[old.method == METHOD].set_index("seed")
    e = dp[dp.draw == 2024].set_index("seed"); mx = float(max((e.pr_auc - old.pr_auc).abs().max(), (e.f1 - old.f1).abs().max()))
    assert mx < 1e-9, f"existing draw does not reproduce dp_control metrics ({mx})"

    out = []
    rm = real[["pr_auc", "f1"]].mean()
    for d in ALL_DRAWS:
        g = dp[dp.draw == d]
        out.append(dict(row_type="draw", draw=d, dp_pr_auc_mean=g.pr_auc.mean(), dp_f1_mean=g.f1.mean(),
                        real_pr_auc_mean=rm.pr_auc, real_f1_mean=rm.f1,
                        delta_pr_auc=rm.pr_auc - g.pr_auc.mean(), delta_f1=rm.f1 - g.f1.mean(),
                        n_seeds_real_ahead_pr_auc=int((g.delta_pr_auc_real_minus_dp > 0).sum()),
                        n_seeds_real_ahead_f1=int((g.delta_f1_real_minus_dp > 0).sum()), n_pairs=len(g)))
    dr = pd.DataFrame(out)
    out.append(dict(row_type="across_draws", draw="2024,7,99",
                    dp_pr_auc_mean=dr.dp_pr_auc_mean.mean(), dp_f1_mean=dr.dp_f1_mean.mean(),
                    real_pr_auc_mean=rm.pr_auc, real_f1_mean=rm.f1,
                    delta_pr_auc=dr.delta_pr_auc.mean(), delta_pr_auc_min=dr.delta_pr_auc.min(), delta_pr_auc_max=dr.delta_pr_auc.max(),
                    delta_f1=dr.delta_f1.mean(), delta_f1_min=dr.delta_f1.min(), delta_f1_max=dr.delta_f1.max(),
                    n_seeds_real_ahead_pr_auc=int(dr.n_seeds_real_ahead_pr_auc.sum()),
                    n_seeds_real_ahead_f1=int(dr.n_seeds_real_ahead_f1.sum()), n_pairs=int(dr.n_pairs.sum()),
                    sd_dp_pr_auc_across_draws=dr.dp_pr_auc_mean.std(ddof=1), sd_dp_f1_across_draws=dr.dp_f1_mean.std(ddof=1)))
    summ = pd.DataFrame(out); summ.to_csv(f"{R}/dp_draws_summary.csv", index=False)
    json.dump(json.loads(summ.to_json(orient="records")), open(f"{R}/dp_draws_summary.json", "w"), indent=2)
    pd.set_option("display.width", 250); print(summ.round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["train", "score"])
    {"train": stage_train, "score": stage_score}[ap.parse_args().stage]()
