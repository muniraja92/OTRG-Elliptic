"""Explains the feature-noise result: tail heaviness of the training features."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import preprocessing as pp
df, src, dst = pp.load(f"{ROOT}/data"); XL, XA = pp.feature_blocks(df); X = np.hstack([XL, XA])[(df.time_step.values <= 34) & df.y.notna().values]
sd = X.std(0); q1, q3 = np.percentile(X, [25, 75], 0); iqr = q3 - q1; r = sd / np.where(iqr > 0, iqr, np.nan)
out = dict(n_features=int(X.shape[1]), n_zero_iqr=int((iqr == 0).sum()), median_sd_over_iqr=float(np.nanmedian(r)))
json.dump(out, open(f"{ROOT}/results/noise_diagnostic.json", "w"), indent=2); print(out)
