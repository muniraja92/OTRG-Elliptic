"""Download the public data sets used in the article and verify their SHA-256 digests.
Usage: python3 scripts/download_data.py [target_dir]   (default: rawdata/)
Elliptic: PyTorch Geometric mirror of the Elliptic Data Set (Weber et al., 2019; original release on Kaggle).
YelpChi / Amazon: data files of the CARE-GNN repository (Dou et al., 2020).
After downloading, run: python3 -I src/data_audit.py <target_dir>/elliptic data"""
import hashlib, os, sys, urllib.request, zipfile
T = sys.argv[1] if len(sys.argv) > 1 else "rawdata"
FILES = {
    "elliptic/elliptic_txs_features.csv.zip": ("https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip", "d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e"),
    "elliptic/elliptic_txs_edgelist.csv.zip": ("https://data.pyg.org/datasets/elliptic/elliptic_txs_edgelist.csv.zip", "a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec"),
    "elliptic/elliptic_txs_classes.csv.zip": ("https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip", "4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba"),
    "yelp/YelpChi.zip": ("https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip", "3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3"),
    "amazon/Amazon.zip": ("https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip", "c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9"),
}
EXTRACTED = {"elliptic/elliptic_txs_features.csv": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
             "elliptic/elliptic_txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
             "elliptic/elliptic_txs_classes.csv": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
for rel, (url, digest) in FILES.items():
    p = os.path.join(T, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p): print("downloading", url); urllib.request.urlretrieve(url, p)
    assert sha(p) == digest, f"checksum mismatch for {rel}"
    zipfile.ZipFile(p).extractall(os.path.dirname(p)); print("ok", rel)
for rel, digest in EXTRACTED.items():
    assert sha(os.path.join(T, rel)) == digest, f"checksum mismatch for {rel}"
print("all files verified")
