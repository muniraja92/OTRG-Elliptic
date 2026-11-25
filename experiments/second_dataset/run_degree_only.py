"""Supplementary control added after inspecting dev-set (not test) results of run_second.py:
LightGBM on node features + per-relation log-degree + union-graph dispersion only (no propagated
neighbour means). Separates degree/dispersion information from neighbour-feature information.
Same splits, grid, dev usage and train+dev refit as run_second.py. Usage: python3 -I run_degree_only.py <ds>
"""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from src.model import lgbm, tune_lgbm
DS = sys.argv[1]; OUT = f"{ROOT}/results/second_dataset/preds"
D = np.load(f"{ROOT}/data/second_dataset/cache/{DS}_clean.npz", allow_pickle=False)
X, y = D["X"], D["y"].astype(int); n = len(y); RELS = [r for r in D["rel_names"] if r != "homo"]
def sym(r, c):
    m = r != c; A = sp.csr_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(n, n)); A = A + A.T; A.data[:] = 1.0; return A.tocsr()
ADJ = {r: sym(D[f"{r}_row"], D[f"{r}_col"]) for r in RELS}
U = sum(ADJ.values()); U.data[:] = 1.0; d = np.asarray(U.sum(1)).ravel(); d[d == 0] = 1; Pu = sp.diags(1 / d) @ U
logdeg = np.column_stack([np.log1p(np.asarray(ADJ[r].sum(1)).ravel()) for r in RELS])
for s in [42, 1, 7, 123, 2024]:
    I = np.load(f"{OUT}/{DS}_split{s}_idx.npz"); tr, dv, te = I["train"], I["dev"], I["test"]; trdv = np.concatenate([tr, dv])
    Z = StandardScaler().fit(X[tr]).transform(X); m1 = Pu @ Z; m2 = Pu @ (Z ** 2)
    disp = np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).mean(1)
    V = np.hstack([X, logdeg, disp[:, None]])
    hp, ap = tune_lgbm(V[tr], y[tr], V[dv], y[dv], seed=s)
    dev = np.full(n, np.nan); dev[dv] = lgbm(s, **hp).fit(V[tr], y[tr]).predict_proba(V[dv])[:, 1]
    tst = np.full(n, np.nan); tst[te] = lgbm(s, **hp).fit(V[trdv], y[trdv]).predict_proba(V[te])[:, 1]
    np.savez_compressed(f"{OUT}/{DS}_LGBM-deg_split{s}.npz", dev=dev, test=tst)
    print(time.strftime("%H:%M:%S"), DS, s, hp, f"devAP={ap:.4f}", flush=True)
