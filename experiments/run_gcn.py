"""GCN baseline runner (separate process because it dominates CPU time).
Out-of-time run: train on labelled steps 1..29, early-stop on 30..34 (gives the calibration
block predictions); final run: refit on 1..34 for the selected number of epochs, score 35..49.
Note: early stopping and calibration share block 30..34 for the GCN only, which can only
favour this baseline."""
import sys, os, time, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import preprocessing as pp
from baselines.gcn import train_predict
seeds = [int(s) for s in sys.argv[1:]] or [42, 1, 7, 123, 2024]
df, src, dst = pp.load(f"{ROOT}/data"); t = df.time_step.values; y = df.y.values; lab = ~np.isnan(y)
XL, XA = pp.feature_blocks(df); AF = np.hstack([XL, XA]); M = lambda a, b: lab & (t >= a) & (t <= b); test = M(35, 49)
rows = []
for seed in seeds:
    t0 = time.time()
    p_oot, ep = train_predict(AF, src, dst, y, np.where(M(1, 29))[0], np.where(M(30, 34))[0], seed, epochs=400, lr=1e-2, patience=60)
    p_fin, _ = train_predict(AF, src, dst, y, np.where(M(1, 34))[0], None, seed, epochs=ep, lr=1e-2)
    oot = np.full(len(df), np.nan); oot[M(30, 34)] = p_oot[M(30, 34)]; pt = np.full(len(df), np.nan); pt[test] = p_fin[test]
    np.savez_compressed(f"{ROOT}/results/preds/GCN_seed{seed}.npz", oot=oot, test=pt)
    rows.append(dict(model="GCN", seed=seed, fit_s=time.time() - t0, infer_s=np.nan, n_features=AF.shape[1], epochs=ep))
    print(f"GCN seed {seed} epochs={ep} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/results/runtime_gcn.csv", index=False)
