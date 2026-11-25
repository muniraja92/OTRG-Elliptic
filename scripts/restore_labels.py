"""Restore the class labels that were removed from the public release of the saved predictions.

The Elliptic data set is distributed under CC BY-NC-ND 4.0, so the public copy of this repository
does not redistribute its labels (nor the YelpChi/Amazon labels). The saved prediction archives keep
every model output, but the label vector `y` was removed from

    results/preds/_meta.npz
    results/preds_tuning_v2/_meta.npz
    results/second_dataset/preds/<dataset>_split<seed>_idx.npz

Run this script once after building the local data caches (see README) and before any analysis
script; it writes the labels back from your own downloaded copies and checks node counts.

    python3 -I scripts/restore_labels.py
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import preprocessing as pp  # noqa: E402


def write_back(path, y):
    d = dict(np.load(path, allow_pickle=False))
    if "t" in d and len(d["t"]) != len(y):
        raise SystemExit(f"{path}: node count {len(d['t'])} differs from labels {len(y)}")
    d["y"] = y
    np.savez_compressed(path, **d)
    print("restored", os.path.relpath(path, ROOT))


df, _, _ = pp.load(f"{ROOT}/data")
y_ell = df.y.values.astype(np.float64)
for p in (f"{ROOT}/results/preds/_meta.npz", f"{ROOT}/results/preds_tuning_v2/_meta.npz"):
    if os.path.exists(p):
        write_back(p, y_ell)

for ds in ("yelp", "amazon"):
    cache = f"{ROOT}/data/second_dataset/cache/{ds}_clean.npz"
    files = sorted(glob.glob(f"{ROOT}/results/second_dataset/preds/{ds}_split*_idx.npz"))
    if not files:
        continue
    if not os.path.exists(cache):
        print(f"skip {ds}: build {cache} first with experiments/second_dataset/audit_convert.py")
        continue
    y = np.load(cache, allow_pickle=False)["y"].astype(np.int64)
    for p in files:
        write_back(p, y)
