"""Prints the fusion-ablation table. The ablation variants themselves are produced in
run_experiments.py / refit_fusion.py and evaluated in statistical_tests.py."""
import os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = pd.read_csv(f"{ROOT}/results/ablation_results.csv", index_col=0)
print(a[["pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std", "f1_post43_mean", "ece_mean"]].round(4).to_string())
