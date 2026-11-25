"""GNN baselines for the revision (plain PyTorch, full-batch, CPU, sparse matrices).

GraphSAGE (Hamilton et al., 2017), mean aggregator, 2 layers, hidden 100, dropout 0.3, separate
self and neighbour weight matrices:  h' = W_self h + W_nbr mean_{u in N(v)} h_u.
Mowst (Zeng et al., ICLR 2024, arXiv:2311.05185): weak expert = MLP on node features, strong
expert = the GraphSAGE above; confidence C(p) = G(D(p)) with D = variance of the MLP softmax
normalised by its maximum (as in compute_confidence() of github.com/facebookresearch/mowst-gnn)
and G(x) = x (alpha = 1, the simplest fixed G); loss Eq. (1)
    L = mean_v [ C(p_v) L(p_v, y_v) + (1 - C(p_v)) L(p'_v, y_v) ],
trained "in turn" (Algorithm 2): the MLP is updated with the GNN fixed, then the GNN with the
MLP fixed, repeated for rounds. Continuous score for ranking metrics: the expectation of the
sampling inference rule (Algorithm 1), C p + (1 - C) p', which equals the Mowst* weighted
combination evaluated with the Mowst-trained experts.

Shared protocol with baselines/gcn.py: features standardised with training-node statistics,
class-weighted cross-entropy (0.3, 0.7), Adam with weight decay 5e-4, lr 1e-2, validation
PR-AUC every 10 epochs, patience 60, max 400 epochs; when val_idx is None (refit on steps 1-34)
the model is trained for the number of epochs selected out-of-time. Graph is symmetrised; edges
never cross time steps in Elliptic, and labels are never propagated."""
from __future__ import annotations
import time, numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import average_precision_score

CW = torch.tensor([0.3, 0.7])
CFG = dict(hidden=100, dropout=0.3, lr=1e-2, wd=5e-4, max_epochs=400, patience=60, eval_every=10)


def _setup(X, train_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
    return torch.tensor(((X - mu) / sd).astype(np.float32))


def mean_adj(src, dst, n):
    """Row-normalised symmetric adjacency without self loops (mean aggregator); isolated nodes -> 0."""
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    A = (sp.diags((1.0 / d).astype(np.float32)) @ A).tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), (n, n)).coalesce()


class SAGE(nn.Module):
    """Layer-1 neighbour mean of the (fixed) input is precomputed once: W_nbr (P X) == P (X W_nbr)."""
    def __init__(self, d, h=100, p=0.3):
        super().__init__()
        self.s1, self.n1 = nn.Linear(d, h), nn.Linear(d, h, bias=False)
        self.s2, self.n2 = nn.Linear(h, 2), nn.Linear(h, 2, bias=False); self.dp = nn.Dropout(p)

    def forward(self, P, X, PX):
        H = torch.relu(self.s1(X) + self.n1(PX)); H = self.dp(H)
        return self.s2(H) + torch.sparse.mm(P, self.n2(H))


class MLP(nn.Module):
    def __init__(self, d, h=100, p=0.3):
        super().__init__(); self.l1 = nn.Linear(d, h); self.l2 = nn.Linear(h, 2); self.dp = nn.Dropout(p)

    def forward(self, X):
        return self.l2(self.dp(torch.relu(self.l1(X))))


def _ap(y, p, idx):
    return float(average_precision_score(y[idx], p[idx]))


def sage_train_predict(X, src, dst, y, train_idx, val_idx, seed, epochs=None):
    c = CFG; epochs = epochs or c["max_epochs"]
    Xt = _setup(X, train_idx, seed); P = mean_adj(src, dst, len(X))
    with torch.no_grad():
        PX = torch.sparse.mm(P, Xt)
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(np.int64)); tr = torch.tensor(train_idx)
    m = SAGE(X.shape[1], c["hidden"], c["dropout"]); opt = torch.optim.Adam(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
    lossf = nn.CrossEntropyLoss(weight=CW); best, best_state, bad, best_ep = -1, None, 0, epochs; t0 = time.time()
    curve = []
    for ep in range(epochs):
        m.train(); opt.zero_grad(); loss = lossf(m(P, Xt, PX)[tr], yt[tr]); loss.backward(); opt.step()
        if val_idx is None or ep % c["eval_every"]:
            continue
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(P, Xt, PX), 1)[:, 1].numpy()
        ap = _ap(y, p, val_idx); curve.append((ep + 1, ap))
        if ap > best: best, best_state, bad, best_ep = ap, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep + 1
        else:
            bad += c["eval_every"]
            if bad >= c["patience"]: break
    n_run = ep + 1
    if best_state is not None: m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(P, Xt, PX), 1)[:, 1].numpy()
    return p, best_ep, dict(config=dict(model="GraphSAGE-mean", **c), epochs_run=n_run, best_val_ap=best,
                            sec_per_epoch=(time.time() - t0) / n_run, val_curve=curve)


# ------------------------------------------------------------------ Mowst
def variance_confidence(logits):
    """Normalised variance of the softmax: for 2 classes = (p1 - p0)^2 in [0, 1]; G(x) = x."""
    p = torch.softmax(logits, 1)
    v = p.var(1, unbiased=False)
    vmax = torch.tensor([1.0] + [0.0] * (logits.shape[1] - 1)).var(unbiased=False)
    return v / vmax


MOWST_CFG = dict(turn_max_epochs=150, turn_patience=40, turn_eval_every=10, max_rounds=3, round_patience=1,
                 alpha=1.0, first="MLP", pretrain="none", conf_grad=True)


def mowst_train_predict(X, src, dst, y, train_idx, val_idx, seed, epochs=None):
    """epochs=None: out-of-time run, early-stopped per turn on validation PR-AUC of the mixture score;
    returns the per-turn schedule [(expert, n_epochs), ...] which the refit (val_idx None) replays."""
    c, mc = CFG, MOWST_CFG
    Xt = _setup(X, train_idx, seed); P = mean_adj(src, dst, len(X))
    with torch.no_grad():
        PX = torch.sparse.mm(P, Xt)
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(np.int64)); tr = torch.tensor(train_idx)
    mlp, gnn = MLP(X.shape[1], c["hidden"], c["dropout"]), SAGE(X.shape[1], c["hidden"], c["dropout"])
    o_m = torch.optim.Adam(mlp.parameters(), lr=c["lr"], weight_decay=c["wd"])
    o_g = torch.optim.Adam(gnn.parameters(), lr=c["lr"], weight_decay=c["wd"])
    nll = lambda logit: F.cross_entropy(logit, yt[tr], weight=CW, reduction="none")
    wsum = CW[yt[tr]].sum()          # weighted mean, consistent with CrossEntropyLoss(weight=CW)

    def score():
        mlp.eval(); gnn.eval()
        with torch.no_grad():
            lm = mlp(Xt); C = variance_confidence(lm) ** mc["alpha"]
            p = C * torch.softmax(lm, 1)[:, 1] + (1 - C) * torch.softmax(gnn(P, Xt, PX), 1)[:, 1]
        return p.numpy(), C.numpy()

    def state():
        return ({k: v.clone() for k, v in mlp.state_dict().items()}, {k: v.clone() for k, v in gnn.state_dict().items()})

    def run_turn(expert, n_max, early):
        if expert == "MLP":
            gnn.eval()
            with torch.no_grad():
                l_g = nll(gnn(P, Xt, PX)[tr])                      # strong expert fixed: cache its loss
        else:
            mlp.eval()
            with torch.no_grad():
                lm = mlp(Xt)[tr]; C_fix = variance_confidence(lm) ** mc["alpha"]; l_m = nll(lm)
        best, best_st, best_e, bad = -1, None, n_max, 0
        for e in range(n_max):
            if expert == "MLP":
                mlp.train(); o_m.zero_grad(); lm = mlp(Xt)[tr]
                C = variance_confidence(lm) ** mc["alpha"]
                if not mc["conf_grad"]: C = C.detach()
                loss = (C * nll(lm) + (1 - C) * l_g).sum() / wsum; loss.backward(); o_m.step()
            else:
                gnn.train(); o_g.zero_grad()
                loss = (C_fix * l_m + (1 - C_fix) * nll(gnn(P, Xt, PX)[tr])).sum() / wsum; loss.backward(); o_g.step()
            if not early or (e + 1) % mc["turn_eval_every"]:
                continue
            ap = _ap(y, score()[0], val_idx)
            if ap > best: best, best_st, best_e, bad = ap, state(), e + 1, 0
            else:
                bad += mc["turn_eval_every"]
                if bad >= mc["turn_patience"]: break
        if early and best_st is not None:
            mlp.load_state_dict(best_st[0]); gnn.load_state_dict(best_st[1])
        return best_e, best

    t0 = time.time(); n_ep = 0
    if val_idx is None:                                            # refit: replay the selected schedule
        for expert, n in epochs:
            run_turn(expert, n, early=False); n_ep += n
        p, C = score()
        return p, epochs, dict(config=mc, schedule=epochs, sec_per_epoch=(time.time() - t0) / max(1, n_ep),
                               mean_conf_all=float(C.mean()))
    order = ["MLP", "GNN"] if mc["first"] == "MLP" else ["GNN", "MLP"]
    sched, g_best, g_state, g_sched, bad_rounds, log = [], -1, None, [], 0, []
    for r in range(mc["max_rounds"]):
        improved = False
        for expert in order:
            e, ap = run_turn(expert, mc["turn_max_epochs"], early=True); sched.append((expert, e)); n_ep += e
            log.append(dict(round=r, expert=expert, epochs=e, val_ap=ap))
            if ap > g_best: g_best, g_state, g_sched, improved = ap, state(), list(sched), True
        bad_rounds = 0 if improved else bad_rounds + 1
        if bad_rounds >= mc["round_patience"]: break
    mlp.load_state_dict(g_state[0]); gnn.load_state_dict(g_state[1])
    p, C = score()
    vi = np.asarray(val_idx)
    return p, g_sched, dict(config=mc, base=c, turns=log, best_val_ap=g_best, schedule=g_sched,
                            sec_per_epoch_approx=(time.time() - t0) / max(1, n_ep),
                            val_mean_conf=float(C[vi].mean()), val_frac_conf_gt_0_5=float((C[vi] > 0.5).mean()))


# ------------------------------------------------------------------ GAT timing only
class GATLayer(nn.Module):
    def __init__(self, d, h, heads=1):
        super().__init__(); self.W = nn.Linear(d, h * heads, bias=False); self.h, self.k = h, heads
        self.a_s = nn.Parameter(torch.randn(heads, h) * 0.1); self.a_d = nn.Parameter(torch.randn(heads, h) * 0.1)

    def forward(self, X, s, d, n):
        Z = self.W(X).view(-1, self.k, self.h)
        e = F.leaky_relu((Z * self.a_s).sum(-1)[s] + (Z * self.a_d).sum(-1)[d], 0.2)
        e = e - torch.zeros(n, self.k).index_reduce_(0, d, e.detach(), "amax", include_self=False)[d]
        w = e.exp(); den = torch.zeros(n, self.k).index_add_(0, d, w)[d]; w = w / (den + 1e-16)
        out = torch.zeros(n, self.k, self.h).index_add_(0, d, Z[s] * w.unsqueeze(-1))
        return out.reshape(n, -1)


def gat_epoch_time(X, src, dst, y, train_idx, n_epochs=3, heads=4, hidden=25):
    """Measures wall-clock time per full-batch training epoch of a 2-layer GAT (4 heads x 25 = 100 hidden)."""
    Xt = _setup(X, train_idx, 42); n = len(X)
    s = np.concatenate([src, dst, np.arange(n)]); d = np.concatenate([dst, src, np.arange(n)])
    st, dt = torch.tensor(s), torch.tensor(d)
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(np.int64)); tr = torch.tensor(train_idx)
    l1, l2 = GATLayer(X.shape[1], hidden, heads), GATLayer(hidden * heads, 2, 1)
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=1e-2)
    times = []
    for _ in range(n_epochs):
        t0 = time.time(); opt.zero_grad()
        out = l2(F.dropout(F.elu(l1(Xt, st, dt, n)), 0.3), st, dt, n)
        loss = F.cross_entropy(out[tr], yt[tr], weight=CW); loss.backward(); opt.step(); times.append(time.time() - t0)
    return dict(heads=heads, hidden_per_head=hidden, n_edges_with_selfloops=int(len(s)), sec_per_epoch=times,
                median_sec_per_epoch=float(np.median(times)))
