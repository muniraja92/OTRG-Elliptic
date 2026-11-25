"""GNN baseline runner, revision v2 (owner: GNN baseline specialist).
Subcommands
  gcn_seeds  SEED...   : GCN with the IDENTICAL configuration of experiments/run_gcn.py
                         (baselines/gcn.py, lr 1e-2, max 400 epochs, patience 60), so seeds
                         42/1/7 already on disk remain valid; writes GCN_seed<S>.npz.
  gcn_lrcheck          : validation-only lr check for GCN (seed 42, lr 1e-3); reports PR-AUC on
                         block 30-34 only; lr 1e-2 value is read from the existing seed-42 OOT preds.
  sage SEED...         : GraphSAGE (baselines/gnn_v2.py).
  mowst SEED...        : Mowst-style MLP/GraphSAGE mixture (baselines/gnn_v2.py).
  gat_time             : measures GAT per-epoch time (no predictions).
Protocol: oot block 30-34 from the run trained on labelled steps 1-29 and early-stopped on
30-34; test 35-49 from a refit on 1-34 for the selected number of epochs. No test metric is
computed anywhere in this script."""
import sys, os, time, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch; torch.set_num_threads(2)
from sklearn.metrics import average_precision_score
from src import preprocessing as pp
R = f"{ROOT}/results"; RT = f"{R}/runtime_gnn_v2.csv"
cmd, args = sys.argv[1], [int(s) for s in sys.argv[2:]]
df, src, dst = pp.load(f"{ROOT}/data"); t = df.time_step.values; y = df.y.values; lab = ~np.isnan(y)
XL, XA = pp.feature_blocks(df); AF = np.hstack([XL, XA]); del XL, XA
M = lambda a, b: lab & (t >= a) & (t <= b); test = M(35, 49)
tr29, va, tr34 = np.where(M(1, 29))[0], np.where(M(30, 34))[0], np.where(M(1, 34))[0]

def log_rt(row):
    old = pd.read_csv(RT) if os.path.exists(RT) else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])]).to_csv(RT, index=False)

def save(name, seed, p_oot, p_fin):
    oot = np.full(len(df), np.nan); oot[M(30, 34)] = p_oot[M(30, 34)]
    pt = np.full(len(df), np.nan); pt[test] = p_fin[test]
    assert not np.isnan(oot[M(30, 34)]).any() and not np.isnan(pt[test]).any()
    np.savez_compressed(f"{R}/preds/{name}_seed{seed}.npz", oot=oot, test=pt)

def vap(p): return float(average_precision_score(y[va], p[va]))

if cmd == "gcn_seeds":
    from baselines.gcn import train_predict
    for seed in args:
        t0 = time.time()
        p_oot, ep = train_predict(AF, src, dst, y, tr29, va, seed, epochs=400, lr=1e-2, patience=60)
        t1 = time.time()
        p_fin, _ = train_predict(AF, src, dst, y, tr34, None, seed, epochs=ep, lr=1e-2)
        save("GCN", seed, p_oot, p_fin)
        log_rt(dict(model="GCN", seed=seed, config="lr1e-2_max400_pat60 (run_gcn.py)", epochs=ep, val_pr_auc=vap(p_oot),
                    oot_fit_s=t1 - t0, refit_s=time.time() - t1, total_s=time.time() - t0))
        print(f"GCN seed {seed} epochs={ep} val_ap={vap(p_oot):.4f} {time.time()-t0:.0f}s", flush=True)

elif cmd == "gcn_lrcheck":
    from baselines.gcn import train_predict
    p42 = np.load(f"{R}/preds/GCN_seed42.npz")["oot"]
    out = {"seed": 42, "selection_block": "labelled steps 30-34 (validation only)",
           "lr_1e-2": {"val_pr_auc": float(average_precision_score(y[va], p42[va])), "source": "existing GCN_seed42.npz oot (max 400 epochs, patience 60)",
                       "epochs": 211}}
    t0 = time.time()
    p, ep = train_predict(AF, src, dst, y, tr29, va, 42, epochs=600, lr=1e-3, patience=60)
    out["lr_1e-3"] = {"val_pr_auc": vap(p), "epochs": ep, "max_epochs": 600, "patience": 60, "fit_s": time.time() - t0}
    json.dump(out, open(f"{R}/gnn_v2_gcn_lrcheck.json", "w"), indent=2); print(json.dumps(out, indent=2))
    log_rt(dict(model="GCN-lrcheck-1e-3", seed=42, config="lr1e-3_max600_pat60", epochs=ep, val_pr_auc=vap(p),
                oot_fit_s=time.time() - t0, refit_s=0, total_s=time.time() - t0))

elif cmd in ("sage", "mowst"):
    from baselines import gnn_v2 as G
    fn = G.sage_train_predict if cmd == "sage" else G.mowst_train_predict
    name = "GraphSAGE" if cmd == "sage" else "Mowst"
    for seed in args:
        t0 = time.time()
        p_oot, ep, info = fn(AF, src, dst, y, tr29, va, seed)
        t1 = time.time()
        p_fin, _, info2 = fn(AF, src, dst, y, tr34, None, seed, epochs=ep)
        save(name, seed, p_oot, p_fin)
        log_rt(dict(model=name, seed=seed, config=json.dumps(info.get("config", {})), epochs=json.dumps(ep) if not np.isscalar(ep) else ep,
                    val_pr_auc=vap(p_oot), oot_fit_s=t1 - t0, refit_s=time.time() - t1, total_s=time.time() - t0))
        json.dump({"oot": info, "refit": info2}, open(f"{R}/gnn_v2_{name}_seed{seed}_info.json", "w"), indent=2, default=float)
        print(f"{name} seed {seed} epochs={ep} val_ap={vap(p_oot):.4f} {time.time()-t0:.0f}s", flush=True)

elif cmd == "gat_time":
    from baselines import gnn_v2 as G
    out = G.gat_epoch_time(AF, src, dst, y, tr29, n_epochs=3)
    json.dump(out, open(f"{R}/gnn_v2_gat_timing.json", "w"), indent=2); print(out)
