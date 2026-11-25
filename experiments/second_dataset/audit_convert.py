"""Phase-0 forensic audit of the CARE-GNN YelpChi / Amazon .mat files and conversion to a
numeric-only cache (npz, no pickles). Run with python3 -I (reads untrusted downloads).
Usage: python3 -I audit_convert.py <mat_path> <name> <out_dir>
"""
import sys, json, hashlib, numpy as np, scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

mat, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
sha = hashlib.sha256(open(mat, "rb").read()).hexdigest()
d = loadmat(mat)
keys = sorted(k for k in d if not k.startswith("__"))
rels = [k for k in keys if k.startswith("net_")] + ["homo"]
y = np.asarray(d["label"]).ravel().astype(int)
X = d["features"]; X = X.toarray() if sp.issparse(X) else np.asarray(X); X = X.astype(np.float64)
n = len(y)
rep = dict(name=name, mat_sha256=sha, keys=keys, n_nodes=int(n), n_features=int(X.shape[1]),
           label_values={int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
lab = np.ones(n, bool)
if name == "amazon":
    first = y[:3305]
    rep["amazon_first3305_label_counts"] = {int(k): int(v) for k, v in zip(*np.unique(first, return_counts=True))}
    lab[:3305] = False
rep["n_labelled_used"] = int(lab.sum()); rep["n_fraud_labelled"] = int(y[lab].sum()); rep["fraud_rate_labelled"] = float(y[lab].mean())
rep["fraud_rate_all_nodes"] = float(y.mean())
rep["nan_features"] = int(np.isnan(X).sum()); rep["inf_features"] = int(np.isinf(X).sum())
rep["constant_features"] = int((X.std(0) == 0).sum())
# duplicate feature rows
_, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
dup = cnt[inv] > 1
rep["rows_in_duplicate_feature_groups"] = int(dup.sum()); rep["n_duplicate_groups"] = int((cnt > 1).sum())
lm = lab & dup
if lm.any():
    g = inv[lm]; yy = y[lm]; conflict = 0
    for gi in np.unique(g):
        v = yy[g == gi]
        if v.min() != v.max(): conflict += 1
    rep["dup_groups_with_conflicting_labels(labelled)"] = int(conflict)
    rep["labelled_rows_in_dup_groups"] = int(lm.sum())
A = {}
union = None
for r in rels:
    M = sp.csr_matrix(d[r]).astype(np.float64); A[r] = M
    deg = np.asarray((M != 0).sum(1)).ravel()
    sym = abs(M - M.T).sum() == 0
    info = dict(nnz=int(M.nnz), symmetric=bool(sym), undirected_edges=int((sp.triu(M, 1) != 0).sum()),
                self_loops=int((M.diagonal() != 0).sum()), weight_values=[float(v) for v in np.unique(M.data)[:5]],
                isolated_all=int((deg == 0).sum()), isolated_labelled=int(((deg == 0) & lab).sum()),
                mean_degree=float(deg.mean()), max_degree=int(deg.max()))
    # edge homophily (fraction of edges, among labelled-labelled pairs, joining same-label nodes)
    C = sp.triu(M, 1).tocoo(); m = lab[C.row] & lab[C.col]
    info["edge_label_homophily"] = float((y[C.row[m]] == y[C.col[m]]).mean()) if m.any() else None
    info["fraud_fraud_edge_share"] = float(((y[C.row[m]] == 1) & (y[C.col[m]] == 1)).mean()) if m.any() else None
    rep[f"rel_{r}"] = info
    if r != "homo": union = (M != 0) if union is None else (union + (M != 0))
U = (union > 0).astype(np.float64); H = (A["homo"] != 0).astype(np.float64)
rep["homo_equals_union_of_relations"] = bool(abs(U - H).sum() == 0); rep["homo_minus_union_nnz"] = int(abs(U - H).sum())
# single-feature AUC leakage screen on labelled nodes
aucs = []
for j in range(X.shape[1]):
    a = roc_auc_score(y[lab], X[lab, j]); aucs.append(max(a, 1 - a))
aucs = np.array(aucs); o = np.argsort(-aucs)
rep["single_feature_auc_top5"] = [(int(j), float(aucs[j])) for j in o[:5]]
rep["single_feature_auc_max"] = float(aucs.max()); rep["single_feature_auc_median"] = float(np.median(aucs))
np.savez_compressed(f"{out}/{name}_clean.npz", X=X, y=y, labelled=lab, rel_names=np.array([r for r in rels]),
                    **{f"{r}_row": sp.triu(A[r], 1).tocoo().row.astype(np.int32) for r in rels},
                    **{f"{r}_col": sp.triu(A[r], 1).tocoo().col.astype(np.int32) for r in rels})
json.dump(rep, open(f"{out}/{name}_audit.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
