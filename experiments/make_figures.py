"""Paper figures, generated only from files in results/. Vector PDF + PNG."""
import os, sys, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import Platt
R, F = f"{ROOT}/results", f"{ROOT}/figures"; os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})
def save(fig, name): fig.tight_layout(); fig.savefig(f"{F}/{name}.pdf"); fig.savefig(f"{F}/{name}.png", dpi=200); plt.close(fig)
steps = pd.read_csv(f"{R}/per_timestep_f1.csv"); summ = pd.read_csv(f"{R}/metrics_summary.csv", index_col=0)
meta = np.load(f"{R}/preds/_meta.npz"); y, t, test = meta["y"], meta["t"], meta["test"]; yi = np.nan_to_num(y).astype(int)
cal = (~np.isnan(y)) & (t >= 30) & (t <= 34)
NM = {"TRUST-GRAF": "OTRG (proposed)", "LGBM-AF": "LightGBM, all features", "LGBM-AF+G": "LightGBM + graph view", "RF-Weber": "RF (Weber config.)", "GCN": "GCN", "GraphSAGE": "GraphSAGE", "Mowst": "Mowst-style", "A-static-gate": "Constant gate", "A-structural-only-gate": "Structural-only gate", "Stack-LR": "Logistic stacking", "A-average": "Equal average", "A-shuffled-graph": "Gate with shuffled graph", "RF": "RF (tuned)"}
MAIN = [m for m in ["TRUST-GRAF", "LGBM-AF", "LGBM-AF+G", "GraphSAGE", "GCN"] if m in summ.index]

# 1 per-time-step F1
fig, ax = plt.subplots(figsize=(6.8, 3.8)); ps = steps.groupby(["method", "time_step"]).f1.mean().unstack()
for k, m in enumerate(MAIN):
    ax.plot(ps.columns, ps.loc[m].values, color=C[k], marker="o", ms=4, label=NM[m])
ax.axvline(42.5, color=MUTED, ls="--", lw=1); ax.text(42.7, 0.97, "dark-market closure", color=MUTED, fontsize=8, va="top")
ax.set_xlabel("Test time step"); ax.set_ylabel("Illicit-class F1 (mean over seeds)"); ax.set_ylim(0, 1.0); ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "fig_per_timestep_f1")

# 2 reliability diagram
def load_cal(m, seed=42):
    d = np.load(f"{R}/preds/{m}_seed{seed}.npz"); pl = Platt().fit(d["oot"][cal], yi[cal]); return d["test"][test], pl.transform(d["test"][test])
fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
bins = np.linspace(0, 1, 11); yt = yi[test]
for k, (m, which, lab) in enumerate([("TRUST-GRAF", 1, "OTRG (Platt)"), ("TRUST-GRAF", 0, "OTRG (raw mixture)"), ("LGBM-AF", 1, "LightGBM, all features (Platt)")]):
    p = load_cal(m)[which]; idx = np.clip(np.digitize(p, bins) - 1, 0, 9); xs, ys = [], []
    for b in range(10):
        mm = idx == b
        if mm.sum() >= 20: xs.append(p[mm].mean()); ys.append(yt[mm].mean())
    ax.plot(xs, ys, color=C[k], marker="o", ms=4, label=lab)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed illicit frequency"); ax.legend(fontsize=7, loc="upper left"); ax.set_title("Seed 42, test steps 35–49", fontsize=8, color=MUTED)
save(fig, "fig_reliability")

# 3 ablation (two panels, one measure each)
abl = pd.read_csv(f"{R}/ablation_results.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
for ax, col, lab in [(axs[0], "pr_auc", "PR-AUC"), (axs[1], "f1_post43", "F1, steps 43–49")]:
    v, e = abl[f"{col}_mean"].values, abl[f"{col}_std"].values; yy = np.arange(len(abl))
    for i, n in enumerate(abl.index):
        col = C[0] if n == "TRUST-GRAF" else MUTED
        ax.errorbar(v[i], i, xerr=e[i], fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.2, capsize=3)
    pad = max(0.004, (v + e).max() - (v - e).min()) * 0.25
    ax.set_xlim((v - e).min() - pad, (v + e).max() + pad); ax.set_xlabel(lab + " (truncated axis)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(np.arange(len(abl))); axs[0].set_yticklabels([NM.get(n, n) for n in abl.index], fontsize=8)
save(fig, "fig_ablation")

# 4 gate weight by time step
al = []
for f in glob.glob(f"{R}/preds/TRUST-GRAF_seed*.npz"):
    d = np.load(f); al.append(pd.Series(d["alpha"][test]).groupby(t[test]).mean())
al = pd.concat(al, axis=1); fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.plot(al.index, al.mean(1), color=C[0], marker="o", ms=4); ax.fill_between(al.index, al.min(1), al.max(1), color=C[0], alpha=0.15, lw=0)
ax.axvline(42.5, color=MUTED, ls="--", lw=1); ax.set_xlabel("Test time step"); ax.set_ylabel(r"Mean gate weight $\alpha$ on node expert")
save(fig, "fig_gate_alpha")

# 5 PR curves seed 42
fig, ax = plt.subplots(figsize=(3.6, 3.4))
for k, m in enumerate(MAIN):
    p = load_cal(m)[1]; pr, rc, _ = precision_recall_curve(yt, p); ax.plot(rc, pr, color=C[k], lw=1.6, label=NM[m])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=7, loc="lower left"); ax.set_title("Seed 42, test steps 35–49", fontsize=8, color=MUTED)
save(fig, "fig_pr_curves")

# 6 runtime
rt = pd.read_csv(f"{R}/runtime_profile.csv")
if os.path.exists(f"{R}/runtime_gcn.csv"): rt = pd.concat([rt, pd.read_csv(f"{R}/runtime_gcn.csv")])
rs = rt.groupby("model").fit_s.mean().sort_values(); fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.barh(rs.index, rs.values, color="#9ec5f4", height=0.6); ax.set_xscale("log"); ax.set_xlabel("Final-fit wall time, s (log scale, 2 CPU cores)"); ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=8); save(fig, "fig_runtime")

# 7 architecture schematic
fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2)
def box(x, y, w, h, s, fc="#eef4fc"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=MUTED, lw=1)); ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=7.5, color=INK)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
box(0.1, 2.1, 1.7, 0.8, "Node features\n$\\mathbf{x}_i$ (165)"); box(0.1, 0.3, 1.7, 0.8, "Same-step graph\n$\\mathcal{G}_s$")
box(2.2, 0.3, 1.8, 0.8, "Label-free propagation\n$\\mathbf{g}_i$ (375)")
box(4.4, 2.1, 1.6, 0.8, "Node expert\n$f_\\theta(\\mathbf{x}_i)$"); box(4.4, 0.3, 1.6, 0.8, "Graph expert\n$h_\\phi([\\mathbf{x}_i;\\mathbf{g}_i])$")
box(4.4, 1.2, 1.6, 0.7, "Reliability $\\mathbf{r}_i$", fc="#fdf1e9")
box(6.4, 1.05, 1.6, 1.0, "Gate $\\alpha_i$ fitted\nout-of-time\n(steps 15–29)", fc="#fdf1e9")
box(8.3, 1.05, 1.6, 1.0, "Platt + threshold\n(steps 30–34)\n$\\to q_i$")
arr(1.8, 0.7, 2.2, 0.7); arr(1.8, 2.5, 4.4, 2.5); arr(4.0, 0.7, 4.4, 0.7); arr(1.8, 2.3, 4.4, 0.95)
arr(6.0, 2.5, 6.4, 1.8); arr(6.0, 0.7, 6.4, 1.3); arr(6.0, 1.55, 6.4, 1.55); arr(8.0, 1.55, 8.3, 1.55)
save(fig, "fig_architecture"); print("figures written:", sorted(os.listdir(F)))
