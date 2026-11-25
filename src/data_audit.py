"""Phase 0 forensic audit of the Elliptic Bitcoin transaction dataset."""
import json, hashlib, sys, numpy as np, pandas as pd
RAW = sys.argv[1]; OUT = sys.argv[2]
feat = pd.read_csv(f"{RAW}/elliptic_txs_features.csv", header=None)
cls = pd.read_csv(f"{RAW}/elliptic_txs_classes.csv")
edg = pd.read_csv(f"{RAW}/elliptic_txs_edgelist.csv")
feat.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, feat.shape[1]-1)]
a = {}
a["n_nodes"] = int(len(feat)); a["n_feature_cols"] = int(feat.shape[1]-2)
a["n_edges"] = int(len(edg)); a["dup_txId"] = int(feat.txId.duplicated().sum())
a["dup_feature_rows_excl_id"] = int(feat.drop(columns="txId").duplicated().sum())
a["dup_edges"] = int(edg.duplicated().sum()); a["self_loops"] = int((edg.txId1==edg.txId2).sum())
a["missing_values"] = int(feat.isna().sum().sum())
a["class_counts"] = cls["class"].astype(str).value_counts().to_dict()
assert set(cls.txId)==set(feat.txId), "class/feature id mismatch"
ts = feat.set_index("txId").time_step
e_t1 = ts.loc[edg.txId1].values; e_t2 = ts.loc[edg.txId2].values
a["edges_crossing_time_steps"] = int((e_t1!=e_t2).sum())
a["time_steps"] = [int(ts.min()), int(ts.max())]
m = feat[["txId","time_step"]].merge(cls, on="txId")
tab = m.groupby("time_step")["class"].value_counts().unstack(fill_value=0)
tab.columns = [str(c) for c in tab.columns]
tab["illicit_rate_labelled"] = tab["1"]/(tab["1"]+tab["2"])
tab.to_csv(f"{OUT}/class_by_timestep.csv")
# feature near-constant check, correlation of features with label (leakage screen)
lab = m[m["class"]!="unknown"].copy(); y = (lab["class"]=="1").astype(int).values
X = feat.set_index("txId").loc[lab.txId].drop(columns="time_step").values
from sklearn.metrics import roc_auc_score
aucs = np.array([roc_auc_score(y, X[:,j]) for j in range(X.shape[1])])
a["max_single_feature_auc"] = float(np.max(np.maximum(aucs,1-aucs)))
a["n_features_single_auc_gt_0.9"] = int(np.sum(np.maximum(aucs,1-aucs)>0.9))
# exact-duplicate feature vectors with conflicting labels
d = pd.DataFrame(X).round(10); d["y"]=y
g = d.groupby(list(range(X.shape[1])))["y"].agg(["nunique","size"])
a["duplicate_vectors_labelled"] = int((g["size"]>1).sum()); a["conflicting_duplicate_vectors"] = int((g["nunique"]>1).sum())
deg = pd.concat([edg.txId1, edg.txId2]).value_counts()
a["isolated_nodes"] = int(len(set(feat.txId)-set(deg.index)))
a["sha256"] = {f: hashlib.sha256(open(f"{RAW}/{f}","rb").read()).hexdigest() for f in ["elliptic_txs_features.csv","elliptic_txs_edgelist.csv","elliptic_txs_classes.csv"]}
json.dump(a, open(f"{OUT}/dataset_audit.json","w"), indent=2)
print(json.dumps({k:v for k,v in a.items() if k!="sha256"}, indent=1)); print(tab.to_string())
# save compact processed arrays
feat.to_parquet(f"{OUT}/features.parquet"); cls.to_parquet(f"{OUT}/classes.parquet"); edg.to_parquet(f"{OUT}/edges.parquet")
