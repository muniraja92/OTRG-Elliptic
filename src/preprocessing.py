"""Data loading, leakage-safe temporal splits, and label-free neighbourhood views for Elliptic.

All graph quantities are computed from node features and edges of the same time step only.
Elliptic has no edges between time steps (verified in the dataset audit), so propagation
never mixes information across the temporal train/validation/test boundary, and labels are
never propagated.
"""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp

N_LOCAL = 93       # f1..f93 local features (time_step excluded; see dataset_audit.md)
N_AGG = 72         # f94..f165 neighbour aggregates computed by the dataset creators


def load(data_dir: str):
    feat = pd.read_parquet(f"{data_dir}/features.parquet")
    cls = pd.read_parquet(f"{data_dir}/classes.parquet")
    edg = pd.read_parquet(f"{data_dir}/edges.parquet")
    df = feat.merge(cls, on="txId", how="left", validate="one_to_one")
    assert len(df) == 203769 and len(edg) == 234355, "unexpected dataset size"
    df["y"] = df["class"].map({"1": 1, "2": 0}).astype("float")  # unknown -> NaN
    idx = pd.Series(np.arange(len(df)), index=df.txId.values)
    src = idx.loc[edg.txId1.values].values; dst = idx.loc[edg.txId2.values].values
    assert (df.time_step.values[src] == df.time_step.values[dst]).all(), "cross-step edge"
    return df.reset_index(drop=True), src, dst


def feature_blocks(df):
    cols = [f"f{i}" for i in range(1, N_LOCAL + 1 + N_AGG)]
    X = df[cols].to_numpy(np.float64)
    return X[:, :N_LOCAL], X[:, N_LOCAL:]


def _row_norm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ A


def neighbourhood_view(XL, src, dst, n, hops=2):
    """Mean of neighbours' local features along in-edges and out-edges, 1..hops hops,
    plus log-degree and neighbourhood dispersion. Label-free by construction."""
    A_out = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))  # i -> j payment flow
    A_in = A_out.T.tocsr()
    P_out, P_in = _row_norm(A_out), _row_norm(A_in)
    blocks, names = [], []
    for tag, P in (("out", P_out), ("in", P_in)):
        H = XL
        for h in range(1, hops + 1):
            H = P @ H
            blocks.append(H); names += [f"{tag}{h}_f{j+1}" for j in range(XL.shape[1])]
    deg_out = np.asarray(A_out.sum(1)).ravel(); deg_in = np.asarray(A_in.sum(1)).ravel()
    # dispersion: mean over features of the std of 1-hop neighbours (undirected)
    A_u = ((A_out + A_in) > 0).astype(float); P_u = _row_norm(A_u)
    m1 = P_u @ XL; m2 = P_u @ (XL ** 2)
    disp = np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).mean(1)
    extra = np.column_stack([np.log1p(deg_in), np.log1p(deg_out), disp])
    G = np.hstack(blocks + [extra]); names += ["log_deg_in", "log_deg_out", "nbr_dispersion"]
    return G, names, deg_in, deg_out


def shuffle_edges_within_step(src, dst, time_step, rng):
    """Structure-destroying control: re-wire every edge endpoint to a random node of the
    same time step (keeps per-step edge counts, destroys who-pays-whom)."""
    s2, d2 = src.copy(), dst.copy()
    for t in np.unique(time_step):
        nodes = np.where(time_step == t)[0]
        m = time_step[src] == t
        s2[m] = rng.choice(nodes, m.sum()); d2[m] = rng.choice(nodes, m.sum())
    return s2, d2


def temporal_masks(df, train_end=34, test_start=35):
    lab = df.y.notna().values; t = df.time_step.values
    return dict(train=lab & (t <= train_end), test=lab & (t >= test_start), labelled=lab)
