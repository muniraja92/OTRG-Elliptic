"""Second-dataset replication of the OTRG gating/view question on YelpChi and Amazon (CARE-GNN .mat files).

These graphs are NOT temporal, so the out-of-time element cannot be tested. Adapted protocol,
fixed before modelling:
  split      stratified train 40% / dev 20% / test 40% of labelled nodes, split seeds 42,1,7,123,2024
  tuning     LightGBM grid (src.model.tune_lgbm) trained on train, early-stopped/selected on dev
  cross-fit  5-fold stratified K-fold on train -> out-of-fold (OOF) expert predictions
  gate fit   OOF train predictions (src.fusion.ReliabilityGate, unchanged)
  lambda     gate penalty chosen from {0,1e-4,1e-3,1e-2} by dev log-loss (as refit_fusion.py)
  dev        expert predictions from models trained on train only -> Platt + F1-optimal threshold (every method)
  test       experts refit on train+dev, frozen gate/Platt/threshold applied, scored once
Graph view: per relation, label-free mean propagation of (train-standardised) node features over
1 and 2 hops, plus log(1+degree) per relation and 1-hop feature dispersion on the union graph.
Rewired control (v2, configuration model): within each relation, all edge stubs (both endpoints of
every undirected edge) are shuffled and re-paired, so each node keeps its degree up to removed
self-loops/multi-edges, fixed rng 2024. The first run permuted only the column endpoint of the
upper-triangular edge list ("onesided"); because YelpChi node indices are strongly label-ordered, that
control leaked index position through each node's row/column stub share and was discarded (files *-onesided).
Usage: python3 -I run_second.py <yelp|amazon> [rewired_only]
  rewired_only: retrain only LGBM-node (determinism check) and LGBM-Grew and refit A-rewired-gate.
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.model import lgbm, tune_lgbm
from src.fusion import ReliabilityGate, reliability_features, _logit

DS = sys.argv[1]; RONLY = len(sys.argv) > 2 and sys.argv[2] == "rewired_only"; SEEDS = [42, 1, 7, 123, 2024]; GRID = [0.0, 1e-4, 1e-3, 1e-2]
OUT = f"{ROOT}/results/second_dataset/preds"; os.makedirs(OUT, exist_ok=True)
D = np.load(f"{ROOT}/data/second_dataset/cache/{DS}_clean.npz", allow_pickle=False)
X, y, lab = D["X"], D["y"].astype(int), D["labelled"]; n = len(y)
RELS = [r for r in D["rel_names"] if r != "homo"]
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {DS} {s}", flush=True)


def sym(r, c):
    m = r != c; r, c = r[m], c[m]
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)); A = A + A.T
    A.data[:] = 1.0; return A.tocsr()


def row_norm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0; return sp.diags(1.0 / d) @ A


REAL = {r: sym(D[f"{r}_row"], D[f"{r}_col"]) for r in RELS}
rng = np.random.default_rng(2024); REW, rew_info = {}, {}
for r in RELS:
    rr, cc = D[f"{r}_row"], D[f"{r}_col"]; stubs = np.concatenate([rr, cc])[rng.permutation(2 * len(rr))]
    REW[r] = sym(stubs[0::2], stubs[1::2])
    rew_info[r] = dict(edges_real=int(REAL[r].nnz // 2), edges_rewired=int(REW[r].nnz // 2),
                       deg_corr=float(np.corrcoef(np.asarray(REAL[r].sum(1)).ravel(), np.asarray(REW[r].sum(1)).ravel())[0, 1]),
                       deg_exact_frac=float((np.asarray(REAL[r].sum(1)).ravel() == np.asarray(REW[r].sum(1)).ravel()).mean()))
log(f"rewiring {rew_info}")
json.dump(rew_info, open(f"{ROOT}/results/second_dataset/{DS}_rewiring.json", "w"), indent=1)


def graph_view(Z, ADJ):
    blocks = []
    for r in RELS:
        P = row_norm(ADJ[r]); H1 = P @ Z; H2 = P @ H1; blocks += [H1, H2]
    logdeg = np.column_stack([np.log1p(np.asarray(ADJ[r].sum(1)).ravel()) for r in RELS])
    U = sum(ADJ[r] for r in RELS); U.data[:] = 1.0; Pu = row_norm(U.tocsr())
    m1 = Pu @ Z; m2 = Pu @ (Z ** 2); disp = np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).mean(1)
    return np.hstack(blocks + [logdeg, disp[:, None]])   # last 4 columns = structural side signals


NCOMP = min(20, X.shape[1] // 2)


def novelty(fit_idx):   # identical construction to run_experiments.novelty, fitted on given nodes
    sc = StandardScaler().fit(X[fit_idx]); Z = sc.transform(X)
    pca = PCA(NCOMP, random_state=0).fit(Z[fit_idx]); R = Z - pca.inverse_transform(pca.transform(Z))
    return np.log1p((R ** 2).mean(1))


idx_lab = np.where(lab)[0]; T0 = time.time(); meta = {}
for s in SEEDS:
    rest, te = train_test_split(idx_lab, test_size=0.4, stratify=y[idx_lab], random_state=s)
    tr, dv = train_test_split(rest, test_size=1 / 3, stratify=y[rest], random_state=s)
    trdv = np.concatenate([tr, dv])
    sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)   # standardisation for propagation fitted on train
    Greal, Grew = graph_view(Z, REAL), graph_view(Z, REW)
    struct = Greal[:, -4:]; struct_rew = Grew[:, -4:]
    views = {"LGBM-node": X, "LGBM-G": np.hstack([X, Greal]), "LGBM-Grew": np.hstack([X, Grew])}
    if RONLY: del views["LGBM-G"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y[tr]))
    P, hp = {}, {}
    for name, V in views.items():
        hp[name], vap = tune_lgbm(V[tr], y[tr], V[dv], y[dv], seed=s)
        oof = np.full(n, np.nan)
        for a, b in folds:
            m = lgbm(s, **hp[name]).fit(V[tr[a]], y[tr[a]]); oof[tr[b]] = m.predict_proba(V[tr[b]])[:, 1]
        m = lgbm(s, **hp[name]).fit(V[tr], y[tr]); dev = np.full(n, np.nan); dev[dv] = m.predict_proba(V[dv])[:, 1]
        m = lgbm(s, **hp[name]).fit(V[trdv], y[trdv]); tst = np.full(n, np.nan); tst[te] = m.predict_proba(V[te])[:, 1]
        P[name] = dict(oof=oof, dev=dev, test=tst); log(f"split {s} {name} hp={hp[name]} devAP={vap:.4f} ({time.time()-T0:.0f}s)")
    # novelty: OOF fold k fitted on the other train folds; dev on train; test on train+dev
    nov_oof = np.full(n, np.nan)
    for a, b in folds: nov_oof[tr[b]] = novelty(tr[a])[tr[b]]
    nov_dev, nov_te = novelty(tr), novelty(trdv)

    def R_of(pa, pb, st):
        Ro = reliability_features(pa["oof"][tr], pb["oof"][tr], np.column_stack([st, nov_oof])[tr])
        Rd = reliability_features(pa["dev"][dv], pb["dev"][dv], np.column_stack([st, nov_dev])[dv])
        Rt = reliability_features(pa["test"][te], pb["test"][te], np.column_stack([st, nov_te])[te])
        return Ro, Rd, Rt
    variants = {"OTRG": ("LGBM-node", "LGBM-G", struct, list(range(8)), True),
                "A-constant-gate": ("LGBM-node", "LGBM-G", struct, list(range(8)), False),
                "A-structural-only-gate": ("LGBM-node", "LGBM-G", struct, [3, 4, 5, 6, 7], True),
                "A-rewired-gate": ("LGBM-node", "LGBM-Grew", struct_rew, list(range(8)), True)}
    if RONLY: variants = {"A-rewired-gate": variants["A-rewired-gate"]}
    gate_info = {}
    for vn, (ea, eb, st, use, feat) in variants.items():
        pa, pb = P[ea], P[eb]; Ro, Rd, Rt = R_of(pa, pb, st); Ro, Rd, Rt = Ro[:, use], Rd[:, use], Rt[:, use]
        best = None
        for lam in (GRID if feat else [1e-2]):
            g = ReliabilityGate(l2=lam, use_features=feat).fit(pa["oof"][tr], pb["oof"][tr], Ro, y[tr])
            pd_, _ = g.predict(pa["dev"][dv], pb["dev"][dv], Rd); ll = log_loss(y[dv], np.clip(pd_, 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]: best = (ll, lam, g, pd_)
        ll, lam, g, pd_ = best
        pt, al = g.predict(pa["test"][te], pb["test"][te], Rt)
        dev = np.full(n, np.nan); dev[dv] = pd_; tst = np.full(n, np.nan); tst[te] = pt; alpha = np.full(n, np.nan); alpha[te] = al
        P[vn] = dict(dev=dev, test=tst, alpha=alpha, w=g.w_)
        gate_info[vn] = dict(l2=lam, dev_logloss=ll, w=np.round(g.w_, 4).tolist(), alpha_mean=float(al.mean()),
                             alpha_p10=float(np.quantile(al, 0.1)), alpha_p90=float(np.quantile(al, 0.9)))
        log(f"split {s} {vn} {gate_info[vn]}")
    if RONLY:
        old = np.load(f"{OUT}/{DS}_LGBM-node_split{s}.npz")
        log(f"split {s} determinism check LGBM-node max|diff| dev={np.nanmax(np.abs(old['dev']-P['LGBM-node']['dev'])):.2e} test={np.nanmax(np.abs(old['test']-P['LGBM-node']['test'])):.2e}")
        for k in ("LGBM-Grew", "A-rewired-gate"):
            np.savez_compressed(f"{OUT}/{DS}_{k}_split{s}.npz", **{kk: vv for kk, vv in P[k].items() if kk != "oof"})
        meta[s] = dict(hp=hp, gates=gate_info); json.dump(meta, open(f"{ROOT}/results/second_dataset/{DS}_run_meta_rewired_v2.json", "w"), indent=1, default=str)
        continue
    pa, pb = P["LGBM-node"], P["LGBM-G"]
    P["A-average"] = dict(dev=0.5 * (pa["dev"] + pb["dev"]), test=0.5 * (pa["test"] + pb["test"]))
    Zf = lambda k, ix: np.column_stack([_logit(pa[k][ix]), _logit(pb[k][ix])])
    stk = LogisticRegression(C=1.0).fit(Zf("oof", tr), y[tr])
    dev = np.full(n, np.nan); dev[dv] = stk.predict_proba(Zf("dev", dv))[:, 1]; tst = np.full(n, np.nan); tst[te] = stk.predict_proba(Zf("test", te))[:, 1]
    P["Stack-LR"] = dict(dev=dev, test=tst)
    for k, v in P.items():
        np.savez_compressed(f"{OUT}/{DS}_{k}_split{s}.npz", **{kk: vv for kk, vv in v.items() if kk != "oof"})
    np.savez_compressed(f"{OUT}/{DS}_split{s}_idx.npz", train=tr, dev=dv, test=te, y=y)
    meta[s] = dict(hp=hp, gates=gate_info, n_train=len(tr), n_dev=len(dv), n_test=len(te),
                   fraud_train=int(y[tr].sum()), fraud_dev=int(y[dv].sum()), fraud_test=int(y[te].sum()), G_dims=int(Greal.shape[1]))
    json.dump(meta, open(f"{ROOT}/results/second_dataset/{DS}_run_meta.json", "w"), indent=1, default=str)
log(f"ALL DONE {time.time()-T0:.0f}s")
