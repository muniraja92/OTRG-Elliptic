"""Full-batch 2-layer GCN baseline (Kipf & Welling 2017) in plain PyTorch, following the
Weber et al. (2019) setup: all features, weighted cross-entropy, training on labelled nodes of
steps 1..34. Elliptic has no cross-step edges, so test-step nodes never exchange messages with
training-step nodes; the test graph is only seen at inference."""
from __future__ import annotations
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
from sklearn.metrics import average_precision_score


def norm_adj(src, dst, n):
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32) + sp.eye(n, dtype=np.float32)
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    A = sp.diags(d) @ A @ sp.diags(d); A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), (n, n)).coalesce()


class GCN(nn.Module):
    def __init__(self, d, h=100, p=0.3):
        super().__init__(); self.l1 = nn.Linear(d, h); self.l2 = nn.Linear(h, 2); self.dp = nn.Dropout(p)

    def forward(self, A, X):
        H = torch.relu(torch.sparse.mm(A, self.l1(X))); H = self.dp(H)
        return torch.sparse.mm(A, self.l2(H))


def train_predict(X, src, dst, y, train_idx, val_idx, seed, epochs=1000, lr=1e-3, wd=5e-4, patience=100):
    torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); A = norm_adj(src, dst, len(X))
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(np.int64))
    m = GCN(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]))
    tr = torch.tensor(train_idx); best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad(); out = m(A, Xt); loss = lossf(out[tr], yt[tr]); loss.backward(); opt.step()
        if val_idx is None:          # final refit: fixed number of epochs chosen out-of-time
            continue
        if ep % 10 == 0:
            m.eval()
            with torch.no_grad():
                p = torch.softmax(m(A, Xt), 1)[:, 1].numpy()
            ap = average_precision_score(y[val_idx], p[val_idx])
            if ap > best: best, best_state, bad, best_ep = ap, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep + 1
            else:
                bad += 10
                if bad >= patience: break
    if best_state is not None:
        m.load_state_dict(best_state); ep = best_ep
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(A, Xt), 1)[:, 1].numpy(), ep
