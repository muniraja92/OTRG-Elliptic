"""GNN baselines on YelpChi and Amazon (round-2 reviewer: the second-dataset comparison lacked GNNs).

Protocol (fixed before any model was trained):
  splits     the five stratified 40/20/40 splits written by run_second.py (split seeds 42,1,7,123,2024),
             loaded from results/second_dataset/preds/<ds>_split<s>_idx.npz. run_second.py is a top-level
             script (it trains on import), so its split code cannot be imported; instead the same two
             train_test_split calls are re-executed here and asserted identical to the saved train/dev/test.
  features   the same raw node features X from data/second_dataset/cache/<ds>_clean.npz, standardised with
             training-node statistics (baselines.gnn_v2._setup, as for the Elliptic GNNs). Unlabelled Amazon
             users are feature carriers in message passing only; labels are never propagated.
  graphs     YelpChi: R-U-R, R-T-R, R-S-R; Amazon: U-P-U, U-S-U, U-V-U. The homogeneous graph is the union of
             the three relations built here; Amazon's supplied 'homo' matrix is NOT used (audit: inconsistent).
  models     (a) GraphSAGE, 2 layers, mean aggregator, hidden 64, dropout 0.3, on the union graph
             (baselines.gnn_v2.SAGE: h' = W_self h + W_nbr mean_N(v) h).
             (b) RSAGE, relation-aware: h' = W_self h + sum_r W_r mean_{N_r(v)} h, i.e. a linear map of the
             concatenation [h, mean_1 h, mean_2 h, mean_3 h] (R-GCN / RGraphSAGE-style), same sizes.
  training   full batch, plain PyTorch, sparse row-normalised adjacencies, 2 threads; class-weighted
             cross-entropy with balanced weights from training labels (n_tr / (2 n_c)); Adam lr 1e-2,
             weight decay 5e-4; max 300 epochs; dev PR-AUC every epoch, early stopping patience 30, best-epoch
             weights kept. Model seed = split seed.
  dev/test   dev predictions from the train-only model -> Platt + F1-optimal threshold (src.evaluation, as in
             evaluate_second.py). Mirroring the tree pipeline's train+dev refit, the test model is retrained
             on train+dev for the selected number of epochs (same seed) and scored once with the frozen
             Platt map and threshold. No choice uses test data.
Usage: flock /tmp/cpu.lock python3 -I run_gnn_second.py <yelp|amazon> [...]
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from baselines.gnn_v2 import SAGE, _setup, mean_adj
from src.evaluation import metrics, best_f1_threshold, Platt

SEEDS = [42, 1, 7, 123, 2024]; R = f"{ROOT}/results/second_dataset"; PRED = f"{R}/preds_gnn"; os.makedirs(PRED, exist_ok=True)
CFG = dict(hidden=64, dropout=0.3, lr=1e-2, wd=5e-4, max_epochs=300, patience=30, eval_every=1, class_weight="balanced(train)")


class RSAGE(nn.Module):
    """Relation-aware mean aggregation; layer-1 neighbour means of the fixed input are precomputed."""
    def __init__(self, d, k, h=64, p=0.3):
        super().__init__()
        self.s1 = nn.Linear(d, h); self.r1 = nn.ModuleList([nn.Linear(d, h, bias=False) for _ in range(k)])
        self.s2 = nn.Linear(h, 2); self.r2 = nn.ModuleList([nn.Linear(h, 2, bias=False) for _ in range(k)]); self.dp = nn.Dropout(p)

    def forward(self, Ps, X, PXs):
        H = self.s1(X)
        for W, PX in zip(self.r1, PXs): H = H + W(PX)
        H = self.dp(torch.relu(H)); out = self.s2(H)
        for W, P in zip(self.r2, Ps): out = out + torch.sparse.mm(P, W(H))
        return out


def train_predict(kind, X, ADJ, y, fit_idx, val_idx, seed, epochs=None):
    c = CFG; Xt = _setup(X, fit_idx, seed)
    yt = torch.tensor(y.astype(np.int64)); tr = torch.tensor(fit_idx)
    nc = np.bincount(y[fit_idx], minlength=2); cw = torch.tensor(len(fit_idx) / (2.0 * nc), dtype=torch.float32)
    if kind == "SAGE":
        P = ADJ["union"]
        with torch.no_grad(): PX = torch.sparse.mm(P, Xt)
        m = SAGE(X.shape[1], c["hidden"], c["dropout"]); fwd = lambda: m(P, Xt, PX)
    else:
        Ps = ADJ["rels"]
        with torch.no_grad(): PXs = [torch.sparse.mm(P, Xt) for P in Ps]
        m = RSAGE(X.shape[1], len(Ps), c["hidden"], c["dropout"]); fwd = lambda: m(Ps, Xt, PXs)
    opt = torch.optim.Adam(m.parameters(), lr=c["lr"], weight_decay=c["wd"]); lossf = nn.CrossEntropyLoss(weight=cw)
    n_max = epochs or c["max_epochs"]; best, best_state, best_ep, bad, curve = -1.0, None, n_max, 0, []; t0 = time.time()
    for ep in range(n_max):
        m.train(); opt.zero_grad(); loss = lossf(fwd()[tr], yt[tr]); loss.backward(); opt.step()
        if val_idx is None: continue
        m.eval()
        with torch.no_grad(): p = torch.softmax(fwd(), 1)[:, 1].numpy()
        ap = float(average_precision_score(y[val_idx], p[val_idx])); curve.append(round(ap, 5))
        if ap > best: best, best_state, best_ep, bad = ap, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1, 0
        else:
            bad += 1
            if bad >= c["patience"]: break
    n_run = ep + 1
    if best_state is not None: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad(): p = torch.softmax(fwd(), 1)[:, 1].numpy()
    return p, best_ep, dict(epochs_run=n_run, best_dev_ap=best, sec=time.time() - t0, class_weight=cw.tolist(), dev_curve=curve)


def run(ds):
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {ds} {s}", flush=True)
    D = np.load(f"{ROOT}/data/second_dataset/cache/{ds}_clean.npz", allow_pickle=False)
    X, y, lab = D["X"], D["y"].astype(int), D["labelled"]; n = len(y)
    RELS = [r for r in D["rel_names"] if r != "homo"]; assert len(RELS) == 3
    src = {r: D[f"{r}_row"].astype(np.int64) for r in RELS}; dst = {r: D[f"{r}_col"].astype(np.int64) for r in RELS}
    nd = lambda a, b: (lambda m: (a[m], b[m]))(a != b)    # drop self loops (none per audit; defensive)
    rels = [mean_adj(*nd(src[r], dst[r]), n) for r in RELS]
    us, ud = nd(np.concatenate([src[r] for r in RELS]), np.concatenate([dst[r] for r in RELS]))
    union = mean_adj(us, ud, n)                              # mean_adj symmetrises and deduplicates
    union_edges = int(union._nnz() // 2)
    log(f"relations {RELS}; union undirected edges {union_edges}"); ADJ = dict(union=union, rels=rels)
    idx_lab = np.where(lab)[0]; rows, meta = [], dict(config=CFG, relations=list(map(str, RELS)), union_edges=union_edges, splits={})
    for s in SEEDS:
        I = np.load(f"{R}/preds/{ds}_split{s}_idx.npz")
        rest, te = train_test_split(idx_lab, test_size=0.4, stratify=y[idx_lab], random_state=s)   # = run_second.py
        tr, dv = train_test_split(rest, test_size=1 / 3, stratify=y[rest], random_state=s)
        assert np.array_equal(te, I["test"]) and np.array_equal(tr, I["train"]) and np.array_equal(dv, I["dev"]), "split mismatch"
        assert np.array_equal(y, I["y"])
        trdv = np.concatenate([tr, dv]); meta["splits"][s] = {}
        for kind, name in (("SAGE", "GraphSAGE"), ("RSAGE", "RSAGE")):
            pdv_all, ep, info = train_predict(kind, X, ADJ, y, tr, dv, seed=s)
            pte_all, _, info_r = train_predict(kind, X, ADJ, y, trdv, None, seed=s, epochs=ep)
            dev = np.full(n, np.nan); dev[dv] = pdv_all[dv]; tst = np.full(n, np.nan); tst[te] = pte_all[te]
            np.savez_compressed(f"{PRED}/{ds}_{name}_split{s}.npz", dev=dev, test=tst)
            pl = Platt().fit(dev[dv], y[dv]); cd, ct = pl.transform(dev[dv]), pl.transform(tst[te])
            thr = best_f1_threshold(y[dv], cd); mt = metrics(y[te], ct, thr)   # test scored once
            rows.append(dict(dataset=ds, split_seed=s, method=name, **mt))
            meta["splits"][s][name] = dict(best_epoch=ep, refit_sec=info_r["sec"], **info)
            log(f"split {s} {name} best_ep={ep} run={info['epochs_run']} devAP={info['best_dev_ap']:.4f} "
                f"train {info['sec']:.0f}s refit {info_r['sec']:.0f}s")
        json.dump(meta, open(f"{R}/gnn_{ds}_meta.json", "w"), indent=1, default=str)
    return rows


if __name__ == "__main__":
    import pandas as pd
    torch.set_num_threads(2); T0 = time.time(); rows = []
    for ds in sys.argv[1:]: rows += run(ds)
    M = pd.DataFrame(rows); f = f"{R}/gnn_metrics_per_split.csv"
    if os.path.exists(f):                                    # keep other datasets' rows when run per dataset
        old = pd.read_csv(f); M = pd.concat([old[~old.dataset.isin(sys.argv[1:])], M])
    M.to_csv(f, index=False)
    print(f"ALL DONE {time.time()-T0:.0f}s", flush=True)
