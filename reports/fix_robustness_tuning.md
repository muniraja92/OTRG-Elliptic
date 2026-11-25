# Fix report: robustness re-analysis (Issue A) and tuning/calibration separation (Issue B)

Date: 23 Nov 2026. Files owned and written by this agent: `experiments/robustness_v2.py`, `experiments/tuning_sensitivity.py`, `results/robustness_v2_raw.csv`, `results/robustness_v2_summary.csv`, `results/robustness_v2_spread.json`, `results/tuning_v2_grid.csv`, `results/tuning_v2_hyperparameters.json`, `results/tuning_v2_gate.csv`, `results/tuning_v2_metrics.csv`, `results/tuning_v2_summary.csv`, `results/tuning_v2_paired.csv`, `results/preds_tuning_v2/*.npz`, logs `logs/robustness_v2.log` and `logs/tuning_sensitivity.log`. No file in `results/preds`, `manuscript/`, `src/` or any other agent's file was modified. Both scripts were run under the shared lock, `flock /tmp/cpu.lock python3 experiments/robustness_v2.py` and `flock /tmp/cpu.lock python3 experiments/tuning_sensitivity.py` (stages tune, fit, eval in one call), in the background with nohup. Peak memory was not measured formally because `/usr/bin/time` is not installed in the container (my first launch attempt with it failed and was relaunched without it); the resident set size observed with `ps` during the tuning job was about 1.85 GB, and no job was killed.

## Issue A: mis-scaled feature-noise test

### What was done

`experiments/robustness_v2.py` loads the seed-42 final experts saved in `results/models/LGBM-AF_seed42.joblib` and `results/models/LGBM-AF+G_seed42.joblib`, and the saved TRUST-GRAF gate weights. It rebuilds the gate standardisation exactly as `experiments/robustness.py` does (reliability features over out-of-time blocks 15 to 29). It also reuses the Platt maps and F1-optimal thresholds fitted on block 30 to 34 of `results/preds`. The clean scores reproduce the clean rows of `results/robustness_results.csv` to within 1e-12; the only difference is the last-digit loss of the CSV float round trip, for example f1_post43 0.030303030303030304 against 0.0303030303030303. The values are PR-AUC 0.8006, 0.8026 and 0.8020, and F1 0.8178, 0.8244 and 0.8138, for LGBM-AF, LGBM-AF+G and TRUST-GRAF.

Only the node features (all 165) of test-period transactions were perturbed, meaning every node of steps 35 to 49, labelled or not. After each perturbation the propagated neighbourhood view was recomputed from the perturbed local features, as in `robustness.py`. The novelty side-signal of the gate was also recomputed from the perturbed local features, using a scaler and PCA fitted on the unperturbed steps 1 to 34. This is a small departure from `robustness.py`, which kept the clean novelty score; the departure makes the gate inputs consistent with the experts' inputs. All training statistics come from the labelled rows of steps 1 to 34. Nothing was refitted or tuned on perturbed data. Each level was drawn three times with rng seeds 0, 1 and 2, and the tables give the mean and s.d. over draws (`results/robustness_v2_summary.csv`; per-draw rows in `results/robustness_v2_raw.csv`).

Three perturbations were applied. The first is robust-scale Gaussian noise. The scale is s_j = IQR_j/1.349, with MAD_j × 1.4826 as the fallback, at sigma 0.05, 0.1, 0.25 and 0.5. The second is quantile-space noise: u = F_j(x), u' = clip(u + N(0, q), 0, 1), mapped back through the training quantile function Q_j, at q 0.01, 0.02 and 0.05. F_j and Q_j are piecewise-linear over the unique training values, with mid-ranks for ties. I implemented the back-transformation as x' = x + (Q_j(u') − Q_j(u)). For test values inside the training range this equals Q_j(u') exactly. The additive form only prevents test values outside the training range from being clipped onto it. A q = 0 run confirms that the round trip is exact, with all predictions identical to clean. The third is feature dropout: in each test-period transaction a random 10% (16) or 25% (41) of the 165 features is replaced by the training median.

According to `results/robustness_v2_spread.json`, 27 of 165 features have zero IQR (13 local, 14 creator aggregates). The MAD fallback was never used. This is a mathematical necessity rather than a coincidence: zero IQR means that at least half the rows sit at the median, so the MAD is also zero. The robust-noise test therefore leaves these 27 features unperturbed. Among the perturbed features, the median ratio of training s.d. to robust spread is 7.1.

### Results

| Perturbation | Level | LGBM-AF PR-AUC | LGBM-AF+G PR-AUC | TRUST-GRAF PR-AUC | LGBM-AF F1 | LGBM-AF+G F1 | TRUST-GRAF F1 |
|---|---|---|---|---|---|---|---|
| clean | – | 0.8006 | 0.8026 | 0.8020 | 0.8178 | 0.8244 | 0.8138 |
| robust noise | 0.05 | 0.6525 ± 0.0102 | 0.7343 ± 0.0022 | 0.7014 ± 0.0076 | 0.1770 ± 0.0138 | 0.3561 ± 0.0080 | 0.3049 ± 0.0039 |
| robust noise | 0.10 | 0.5804 ± 0.0092 | 0.6875 ± 0.0055 | 0.6411 ± 0.0098 | 0.0993 ± 0.0155 | 0.2459 ± 0.0001 | 0.2025 ± 0.0112 |
| robust noise | 0.25 | 0.4602 ± 0.0096 | 0.5844 ± 0.0095 | 0.5257 ± 0.0127 | 0.0379 ± 0.0139 | 0.1159 ± 0.0057 | 0.0898 ± 0.0138 |
| robust noise | 0.50 | 0.3688 ± 0.0071 | 0.4896 ± 0.0122 | 0.4275 ± 0.0089 | 0.0153 ± 0.0010 | 0.0527 ± 0.0081 | 0.0333 ± 0.0072 |
| quantile noise | 0.01 | 0.8016 ± 0.0011 | 0.8053 ± 0.0017 | 0.8031 ± 0.0010 | 0.8166 ± 0.0012 | 0.8204 ± 0.0009 | 0.8120 ± 0.0009 |
| quantile noise | 0.02 | 0.8014 ± 0.0014 | 0.8056 ± 0.0010 | 0.8031 ± 0.0013 | 0.8161 ± 0.0021 | 0.8209 ± 0.0024 | 0.8137 ± 0.0021 |
| quantile noise | 0.05 | 0.7977 ± 0.0013 | 0.8040 ± 0.0016 | 0.7997 ± 0.0010 | 0.8126 ± 0.0010 | 0.8184 ± 0.0019 | 0.8128 ± 0.0001 |
| feature dropout | 10% | 0.7885 ± 0.0033 | 0.7965 ± 0.0012 | 0.7931 ± 0.0028 | 0.7858 ± 0.0048 | 0.8084 ± 0.0028 | 0.8054 ± 0.0028 |
| feature dropout | 25% | 0.7535 ± 0.0061 | 0.7776 ± 0.0024 | 0.7671 ± 0.0043 | 0.6553 ± 0.0092 | 0.7206 ± 0.0060 | 0.7197 ± 0.0084 |

Post-step-43 F1 is near zero in every setting: 0.029 to 0.031 clean, and at most 0.032 under any perturbation. The mean test gate weight of TRUST-GRAF stays between 0.876 and 0.889 across all perturbations.

### Interpretation and verdict

The rescaling does not remove the fragility; it reframes it. Gaussian noise of only 0.05 robust spreads, roughly one fourteenth of the original "0.1 s.d." on a typical feature (median s.d./robust-spread ratio 7.1), still lowers PR-AUC from 0.80 to 0.65 to 0.73 and invalidates the fixed threshold (F1 0.18 to 0.36). Quantile-space noise that moves each value by up to 5 percentiles leaves every model within 0.005 PR-AUC and 0.006 F1 of clean. Feature dropout degrades gracefully: at 10% the drop is at most 0.012 PR-AUC and 0.032 F1, and at 25% it is 0.025 to 0.047 PR-AUC.

The two noise models differ in one obvious respect. Quantile noise preserves the point masses of the many features in which a single value holds a large share of rows. Additive noise of any size moves every such row off its atom. A likely explanation for the additive-noise collapse is therefore that the trees split at exact atoms, not that they are sensitive to small changes in magnitude. I did not run a separate experiment that perturbs only atom-free features, so this explanation is a hypothesis consistent with the data, not a tested result.

In all three tests the graph-augmented expert is the most robust of the three models, and TRUST-GRAF lies between the two experts, as a near-constant convex mixture must. The claim that one model is "least affected" should therefore be made about LGBM-AF+G, not about the gate.

Issue A is resolved as a measurement problem. The mis-scaled test is replaced by three correctly scaled, evaluation-only tests with repeated draws, and the 27 unperturbable features are reported. The substantive conclusion changes. The manuscript should no longer say that "only one tenth of a s.d." destroys the models. It should say that the models are robust to rank-preserving and value-preserving perturbations (quantile noise and median imputation), but sensitive to additive noise that breaks the discrete structure of the features even at 5% of the interquartile spread.

## Issue B: block 30–34 used for both tuning and calibration; tiny grids

### What was done

`experiments/tuning_sensitivity.py` re-tunes both experts with seed 42 over an expanded grid of 60 configurations: num_leaves {7, 15, 31, 63, 127} × min_child_samples {10, 20, 50, 100} × colsample_bytree {0.3, 0.5, 0.8}. The learning rate is 0.05, and all other settings are as in `src/model.lgbm`. Training uses labelled steps 1 to 24. Early stopping (patience 100, at most 2000 trees) and model selection by average precision both use steps 25 to 29, so block 30 to 34 serves only for Platt scaling and the threshold. With the chosen hyperparameters, the script reruns the cross-fitting over `run_experiments.BLOCKS` and the final 1-to-34 fit for both experts, for seeds 42, 1, 7, 123 and 2024. It then refits three fusions. The TRUST-GRAF gate uses l2 = 1e-3 from `results/gate_lambda_choice.json` and fits on out-of-time blocks 15 to 29, with blockwise novelty as in `refit_fusion.py`. A-static-gate is intercept-only. A-average is the equal average. Predictions are stored only in `results/preds_tuning_v2/`, under the names LGBM-AF-T2, LGBM-AF+G-T2, TRUST-GRAF-T2, A-static-gate-T2 and A-average-T2. Evaluation follows the standard procedure of `experiments/statistical_tests.py`: Platt scaling and the F1-optimal threshold on out-of-time block 30 to 34, then a single scoring on test steps 35 to 49. Paired comparisons use seed-level differences and a time-step-stratified paired bootstrap (B = 2000) of the seed-averaged metric (`results/tuning_v2_paired.csv`). Because of the B = 2000 floor, bootstrap p = 0.0005 means p ≤ 0.0005.

The selected hyperparameters (`results/tuning_v2_hyperparameters.json`) are num_leaves 127, min_child_samples 100, colsample 0.5 and 500 trees for LGBM-AF, against 15/20/0.5/470 originally. For LGBM-AF+G they are 31, 100, 0.5 and 402 trees, against 31/50/0.5/322. Validation is close to saturated on block 25 to 29. Across the grid, AP ranges from 0.9899 to 0.9950 for LGBM-AF and from 0.9917 to 0.9948 for LGBM-AF+G, and the top five configurations of each expert lie within 0.0004 of each other (`results/tuning_v2_grid.csv`). The selection is therefore close to arbitrary among good configurations. That is itself informative: block 25 to 29 is easy, and it reveals little about the shift that follows.

### Results (mean ± s.d. over five seeds; `results/tuning_v2_summary.csv`)

| Method | PR-AUC | F1 | F1 steps 43–49 | Brier | ECE | Original PR-AUC / F1 |
|---|---|---|---|---|---|---|
| LGBM-AF-T2 | 0.7998 ± 0.0009 | 0.8047 ± 0.0040 | 0.0285 ± 0.0033 | 0.0209 | 0.0213 | 0.8016 / 0.8013 |
| LGBM-AF+G-T2 | 0.8005 ± 0.0010 | 0.8223 ± 0.0054 | 0.0306 ± 0.0009 | 0.0199 | 0.0193 | 0.8020 / 0.8202 |
| TRUST-GRAF-T2 | 0.8006 ± 0.0007 | 0.8134 ± 0.0116 | 0.0283 ± 0.0020 | 0.0203 | 0.0206 | 0.8026 / 0.8109 |
| A-static-gate-T2 | 0.8004 ± 0.0006 | 0.8091 ± 0.0066 | 0.0276 ± 0.0012 | 0.0205 | 0.0207 | 0.8024 / 0.8110 |
| A-average-T2 | 0.8008 ± 0.0004 | 0.8194 ± 0.0056 | 0.0296 ± 0.0010 | 0.0200 | 0.0199 | 0.8028 / 0.8118 |

Paired comparisons (Δ = first − second; seed-level s.d.; bootstrap 95% CI):

| Comparison | ΔPR-AUC [CI] | ΔF1 [CI], seeds with Δ>0 |
|---|---|---|
| TRUST-GRAF-T2 vs A-average-T2 | −0.0002 [−0.0011, +0.0006], p = 0.64 | −0.0059 [−0.0089, −0.0032], 2/5, seed s.d. 0.0076 |
| TRUST-GRAF-T2 vs A-static-gate-T2 | +0.0002 [−0.0001, +0.0005], p = 0.29 | +0.0043 [+0.0017, +0.0069], 3/5, seed s.d. 0.0099 |
| A-static-gate-T2 vs A-average-T2 | −0.0004 [−0.0014, +0.0005], p = 0.41 | −0.0102 [−0.0144, −0.0063], 0/5 |
| LGBM-AF+G-T2 vs LGBM-AF-T2 | +0.0007 [−0.0023, +0.0037], p = 0.67 | +0.0176 [+0.0100, +0.0254], 5/5, seed s.d. 0.0063 |
| TRUST-GRAF-T2 vs LGBM-AF-T2 | +0.0008 [+0.0000, +0.0017], p = 0.047 | +0.0087 [+0.0049, +0.0126], 4/5 |
| TRUST-GRAF-T2 vs LGBM-AF+G-T2 | +0.0001 [−0.0021, +0.0025], p = 0.91 | −0.0088 [−0.0139, −0.0040], 1/5 |

Gate diagnostics come from block 30 to 34 only, so no test predictions were used (`results/tuning_v2_gate.csv`). Within each seed the learned gate remains nearly constant: the 10th-to-90th-percentile range of alpha is 0.920 to 0.957 for seed 42, 0.738 to 0.859 for seed 1, 0.828 to 0.885 for seed 7, 0.857 to 0.925 for seed 123 and 0.919 to 0.955 for seed 2024. The largest feature weight in absolute value is 0.21 to 0.40, always on the graph-expert entropy. The intercept, and hence the constant blend, varies much more between seeds: the static-gate alpha runs from 0.71 to 0.94.

### Answer to the question and verdict

All three conclusions survive when tuning and calibration use disjoint blocks and the grid is ten times larger. First, the gate behaves like a constant gate and like averaging. PR-AUC differs by at most 0.0004 among TRUST-GRAF, the static gate and averaging, with no CI excluding zero, and within-seed variation in alpha is small. On F1, simple averaging is now nominally the best of the three fusions. It beats the learned gate by 0.006 under the transaction bootstrap, but only in three of five seeds, against a seed s.d. of 0.008. So the learned gate is at best equivalent to averaging, never better. Second, the graph-augmented expert's F1 advantage over the node-only expert persists: +0.018, positive in all five seeds, CI [+0.010, +0.025], close to the original +0.019. As in the main study, there is no PR-AUC advantage (+0.0007, CI spanning zero). The advantage is therefore one of thresholded classification and calibration, not of ranking. Third, the post-closure collapse holds: F1 over steps 43 to 49 is 0.028 to 0.031 for every method.

Absolute PR-AUC is about 0.001 to 0.002 lower than in the main study for every method. The larger grid with a disjoint, near-saturated validation block therefore did not improve the models, and the original choices were not overfitted to block 30 to 34 in any way that mattered.

Issue B is resolved for hyperparameter tuning. Two limitations remain and should be stated. The gate penalty l2 = 1e-3 was inherited from `results/gate_lambda_choice.json`, which was chosen by log-loss on block 30 to 34, so that one gate setting still touches the calibration block; the λ sensitivity analysis shows the choice is immaterial. The reviewer's second point is also not addressed by design: the Platt map and threshold are fitted on block-30-to-34 predictions from models trained on steps 1 to 29, then applied to models refitted on steps 1 to 34.
