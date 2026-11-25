# Fix: external validity on a second, independent fraud graph (YelpChi and Amazon)

## 1. Summary and verdict

The main study used Elliptic only. I added two real, labelled, multi-relation fraud graphs that are independent of Elliptic and of each other: YelpChi (review spam) and Amazon (fraudulent users). Both come from the CARE-GNN repository. Neither graph is temporal, so the out-of-time part of OTRG cannot be tested on them. Only the two questions that do not depend on time can be tested: whether a label-free propagated graph view is informative relative to a degree-preserving rewired control, and whether the reliability gate beats a constant gate, logistic stacking and equal averaging.

The answers are as follows.

- **Graph view vs rewired control (replicates).** On both datasets the real propagated view beats the rewired control.
  - YelpChi: +0.091 PR-AUC, 5/5 split seeds.
  - Amazon: +0.014 PR-AUC, 5/5 split seeds.
  - On YelpChi most of the raw gain of the view over the node-only model (+0.063 PR-AUC) comes from per-relation degree information rather than from neighbour features. A degree-plus-dispersion-only model recovers +0.046 of it. The real view still adds +0.017 PR-AUC over that degree-only model (5/5 seeds).
  - On Amazon the view's advantage over the degree-only model is +0.009 PR-AUC (4/5 seeds) and is not significant.
  - The graph view gives no F1 gain on Amazon.
- **Gate vs simpler combiners (largely replicates).** The full gate, the constant gate and logistic stacking are within 0.002 PR-AUC of each other on both datasets.
  - Stacking is nominally the best of the three on both.
  - Unlike Elliptic, the gate on YelpChi is not constant (node-expert weight ranges roughly from 0.09 to 0.65 between the 10th and 90th percentiles), but this buys only +0.0017 PR-AUC over a constant gate.
  - Equal averaging is slightly worse than the gate on YelpChi (−0.008 PR-AUC, 5/5 seeds), because the best constant weight on the node expert is about 0.2, far from 0.5. On Amazon averaging is equivalent.

A first version of the rewired control was flawed and was discarded; Section 4.3 describes the flaw. The discarded numbers are kept, clearly labelled, for transparency.

## 2. Provenance

Both files were downloaded from the CARE-GNN repository of Dou et al., using `raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/{YelpChi,Amazon}.zip`. Access through `github.com` itself was refused by the session proxy. Each archive was saved in its own new directory (`data/second_dataset/raw_yelp/`, `data/second_dataset/raw_amazon/`) and extracted into separate directories (`ext_yelp/`, `ext_amazon/`). Every script that reads them lives in `experiments/second_dataset/` and was run with `python3 -I` from `/tmp`.

**SHA-256 checksums**

| File | SHA-256 |
|---|---|
| YelpChi.zip | 3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3 |
| YelpChi.mat | fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42 |
| Amazon.zip | c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9 |
| Amazon.mat | 4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63 |

The `.mat` files are dated 2020-08-20 inside the archives.

**References (each verified on the web)**

- **Dou et al. (CIKM 2020).** Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, P. S. Yu, "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters", CIKM 2020, doi:10.1145/3340531.3411903, arXiv:2008.08692. Its Table 1 reports:
  - YelpChi: 45,954 nodes (14.5% fraud), with R-U-R 49,315, R-T-R 573,616, R-S-R 3,402,743 and ALL 3,846,979 edges.
  - Amazon: 11,944 nodes (9.5% fraud), with U-P-U 175,608, U-S-U 3,566,479, U-V-U 1,036,737 and ALL 4,398,392 edges.
  - Features: 32 handcrafted features for YelpChi and 25 for Amazon.
  - Amazon labels: users with more than 80% helpful votes are benign and users with less than 20% are fraudulent.
- **Rayana and Akoglu (KDD 2015)**, the YelpChi source. S. Rayana and L. Akoglu, "Collective Opinion Spam Detection: Bridging Review Networks and Metadata", KDD 2015, doi:10.1145/2783258.2783370.
  - Their YelpChi has 67,395 reviews of Chicago hotels and restaurants, originally collected by Mukherjee et al. (ICWSM 2013).
  - Labels are Yelp's filtered (spam) versus recommended reviews.
  - The paper itself calls these labels "near" ground truth.
- **Liu et al. (SIGIR 2020)**, the 45,954-review subset. Z. Liu, Y. Dou, P. S. Yu, Y. Deng, H. Peng, "Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection", SIGIR 2020, arXiv:2005.00625. The subset was derived by removing products with more than 800 reviews.
- **McAuley and Leskovec (WWW 2013)**, the Amazon source. J. McAuley and J. Leskovec, "From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews", WWW 2013, arXiv:1303.4402.
- **Zhang et al. (SIGIR 2020)**, the Amazon features. S. Zhang, H. Yin, T. Chen, Q. V. N. Hung, Z. Huang, L. Cui, "GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection", SIGIR 2020. Dou et al. cite it as the source of the features.

**Licence.** The CARE-GNN repository is under the Apache License 2.0, but that licence is attached to the code and does not explicitly address the data files. Rayana and Akoglu state that their datasets are available from the authors' web page but give no licence. I found no explicit licence or terms of use for either `.mat` file, so the data terms should be regarded as unclear and research-use only, with citation of the original papers.

**Counts compared with the published numbers.** All counts match the published figures exactly.

- YelpChi has 45,954 nodes, of which 6,677 are fraud (14.53%).
- The edge counts per relation (49,315, 573,616, 3,402,743) and the homogeneous graph (3,846,979) are identical to Table 1 of Dou et al.
- Amazon has 11,944 nodes. As in the CARE-GNN code (`train.py` uses `range(3305, len(labels))`), the first 3,305 users were excluded from training and evaluation. All 3,305 carry label 0, so they are unlabelled placeholders rather than benign users.
- The remaining 8,639 labelled users include 821 frauds (9.50%), matching "11,944 (9.5%)" only after this exclusion. Over all 11,944 nodes the rate would be 6.87%.
- The Amazon relation edge counts (175,608, 3,566,479, 1,036,737) and the homogeneous graph (4,398,392) match exactly.

## 3. Phase-0 forensic audit

The audit outputs are `data/second_dataset/cache/{yelp,amazon}_audit.json` and logs `results/second_dataset/audit_{yelp,amazon}.log`. They were produced by `experiments/second_dataset/audit_convert.py`, which also writes a numeric-only cache (`*_clean.npz`, no pickles) used by all later scripts.

**Integrity checks.**

- All adjacency matrices are binary, symmetric and free of self-loops.
- Neither dataset has NaN, infinite or constant features.

**YelpChi.**

- There are no duplicate feature rows.
- The homogeneous matrix equals the union of the three relations exactly.
- Isolated nodes:
  - R-U-R: 22,123 nodes, i.e. reviews by users with a single review.
  - R-T-R: 522 nodes.
  - R-S-R: 40 nodes.
  - Union graph: 13 nodes.
- **Edge label homophily.** Among labelled pairs this is 0.996 in R-U-R, against 0.77 and 0.76 in R-S-R and R-T-R. Reviews by the same user almost always share a label.
  - This makes the R-U-R neighbourhood a strong proxy for the label.
  - Under a random node split it means that test reviews are connected to training reviews of the same user.
  - This is a property of the transductive benchmark, not a label leak in our pipeline, because only features are propagated. It does make YelpChi a much easier graph setting than Elliptic.
- **Single-feature leakage screen.** The maximum single-feature ROC-AUC is 0.664 (median 0.530), so no single feature encodes the label.
- **Node index is label-ordered.** The audit found that the node index itself is strongly ordered by label: ROC-AUC of the raw index for fraud is 0.895. This finding became important for the rewiring control (Section 4.3).

**Amazon.**

- 2,104 nodes fall in 776 groups with identical feature vectors. 812 of them are labelled, and no group has conflicting labels.
- Isolated labelled nodes:
  - U-P-U: 703 nodes.
  - U-S-U: 82 nodes.
  - U-V-U: 76 nodes.
  - Union graph: none.
- **The supplied `homo` matrix is not the union of the three relations.** It has 4,398,392 edges, against 4,417,576 in the union, and about one million of its edges are absent from the union. I could not find a documented reason for this. The pipeline therefore uses only the three documented relations and their union, and never the `homo` matrix.
- **Single-feature leakage screen.** This is more worrying for Amazon. The best single feature reaches ROC-AUC 0.883 (features 19, 18 and 15 are all above 0.86, and the median is 0.669).
  - Labels are defined from the share of helpful votes, and the features come from Zhang et al. I could not verify their exact definitions.
  - So I cannot rule out that some features are partly derived from the same helpfulness information. If they are, the label construction leaks into the features.
  - This does not affect the paired comparisons between views and combiners, which all see the same node features, but it means that absolute Amazon scores should not be read as fraud-detection difficulty.
- Index ordering is weak on Amazon (ROC-AUC of the index 0.547).

## 4. Protocol adaptations

### 4.1 What is retained and what cannot be tested

The protocol was fixed in the header of `experiments/second_dataset/run_second.py` before any model was trained.

**Splits.** For each split seed (42, 1, 7, 123, 2024), the labelled nodes are split stratified into train 40%, dev 20% and test 40%:

| | Train | Dev | Test | Test frauds |
|---|---|---|---|---|
| YelpChi | 18,381 | 9,191 | 18,382 | 2,671 |
| Amazon | 3,455 | 1,728 | 3,456 | 328 |

**Pipeline.** The Elliptic gate protocol is emulated inside train and dev:

1. LightGBM is tuned on train with early stopping and selection on dev. The grid is the same small grid as the main study (`src.model.tune_lgbm`: `num_leaves` {15, 31, 63} by `min_child_samples` {20, 50}), and it is tuned separately per view and per split seed. The test set is never used.
2. Five-fold stratified cross-fitting on train gives out-of-fold expert predictions.
3. The gate (`src.fusion.ReliabilityGate`, unchanged) and logistic stacking are fitted on these out-of-fold predictions.
4. Experts trained on the full train set predict dev. Dev is used to select the gate penalty from {0, 1e-4, 1e-3, 1e-2} by log-loss, as in `experiments/refit_fusion.py`, and to fit Platt scaling and the F1-optimal threshold for every method.
5. For test, the experts are refitted on train plus dev and the frozen gate, Platt map and threshold are applied. This mirrors the refit on steps 1–34 in the main study.

**What cannot be tested.** Because the folds and splits are exchangeable random partitions, nothing here tests transfer forward in time. The novelty signal measures within-distribution atypicality rather than drift. These experiments therefore address only the view question and the gating question, not the out-of-time claim.

### 4.2 Experts, graph view and gate signals

**Experts.**

- The node expert is LightGBM on the raw node features (32 for YelpChi, 25 for Amazon).
- The graph-augmented expert also sees a label-free view built for each relation:
  - 1-hop and 2-hop mean propagation of the node features, standardised with training-node statistics;
  - log(1 + degree) per relation;
  - the mean 1-hop feature dispersion on the union graph.
- The view has 196 columns for YelpChi and 154 for Amazon.
- Unlabelled Amazon users take part in propagation as feature carriers only. Labels are never propagated.

**Gate signals.** The seven-signal reliability design was adapted to three undirected relations, which gives eight signals:

- disagreement;
- the two expert entropies;
- three per-relation log-degrees (in place of Elliptic's in-degree and out-degree);
- dispersion;
- PCA novelty.

Novelty uses `min(20, d/2)` principal components (16 for YelpChi, 12 for Amazon), because 20 components would leave almost no residual in 25 features. For out-of-fold rows the novelty model is fitted on the other train folds, for dev on train, and for test on train plus dev.

**Fusion variants.**

- OTRG;
- a constant gate;
- equal averaging;
- logistic stacking (C = 1) on the experts' logits;
- a structural-only gate (log-degrees, dispersion, novelty);
- a gate over the node expert and the rewired-graph expert.

### 4.3 The rewired control, and a discarded first version

The control must keep each node's degree in each relation and destroy who is connected to whom.

**First version (discarded).** It permuted the second endpoint of the upper-triangular edge list (pairs i < j). This keeps degrees, but it is not neutral when node indices carry information. After rewiring, a node's new neighbours are drawn from either the "row" or the "column" stub pool, depending on the share of its edges to higher-indexed nodes. That share is a function of index position.

On YelpChi the index is strongly label-ordered, and the row-stub share alone has ROC-AUC 0.950 for fraud in R-S-R and 0.931 in R-T-R. The first rewired expert therefore looked better than the real-graph expert on dev, with dev PR-AUC about 0.915 against 0.896. Its test PR-AUC was 0.929, above the real view's 0.917.

I diagnosed this from the dev results and the label-ordering audit, not from test results. I then replaced the rewiring by a configuration-model rewiring: all stubs of each relation are shuffled and re-paired. Self-loops and duplicate edges are dropped, the rng is fixed at 2024, and only the rewired expert and its gate were retrained. A determinism check confirmed that the retrained node expert reproduced its earlier dev and test predictions exactly (maximum absolute difference 0).

The discarded predictions remain as `*_LGBM-Grew-onesided_*` and `*_A-rewired-gate-onesided_*`. They appear in `summary.csv`, labelled as such, and must not be used as a control.

**Fidelity of the configuration-model rewiring.**

- YelpChi:
  - Edges kept after collapsing duplicates: 3,392,069 of 3,402,743 (R-S-R), 573,330 of 573,616 (R-T-R) and 49,290 of 49,315 (R-U-R).
  - The correlation between real and rewired degrees is above 0.9999 in every relation.
- Amazon:
  - Because the graph is dense, collapsing duplicates loses more edges: 15.4% in U-S-U, 8.0% in U-V-U and 7.7% in U-P-U.
  - Degree correlations are 0.993, 0.991 and 0.999 respectively.
  - On Amazon the control therefore preserves degrees only approximately.

This failure mode does not apply to the directed Elliptic control in `experiments/degree_preserving_control.py`. There, endpoints have a semantic direction (payer and payee), so permuting destinations does not encode node index. I note it here only because the same code pattern can leak information in undirected graphs with ordered indices.

### 4.4 Supplementary control added after inspecting dev results

**Degree-only control.** To separate degree information from neighbour features, I added `LGBM-deg`: node features plus the three per-relation log-degrees and dispersion, with no propagated means. It uses the same splits, grid and dev usage as the other models (`experiments/second_dataset/run_degree_only.py`). It was added after seeing dev-set results, so it is labelled supplementary.

### 4.5 Metrics and statistics

**Metrics.** Test metrics are PR-AUC, ROC-AUC, F1 at the threshold chosen on dev, Brier score and ECE, all after Platt scaling fitted on dev.

**Tests.**

- **Split-seed level.** Differences are computed between split seeds, with paired t-tests and Holm correction within each dataset and metric. Exact Wilcoxon tests are also reported, but with n = 5 their smallest attainable two-sided p is 0.0625.
- **Bootstrap on split 42.** A stratified bootstrap by class, with 1,000 resamples and seed 12345, holds the models and thresholds fixed and is Holm-corrected within each dataset and metric.

## 5. Results

The full results are in `results/second_dataset/summary.csv` (mean and SD over the five split seeds) and `results/second_dataset/stats.csv` (paired comparisons). Per-split metrics are in `metrics_per_split.csv` and gate fits in `{ds}_run_meta.json` and `{ds}_run_meta_rewired_v2.json`.

**Table 1. Test results, mean ± SD over five split seeds**

| Method | YelpChi PR-AUC | YelpChi ROC-AUC | YelpChi F1 | Amazon PR-AUC | Amazon ROC-AUC | Amazon F1 |
|---|---|---|---|---|---|---|
| LGBM-node | 0.854 ± 0.006 | 0.956 ± 0.001 | 0.768 ± 0.007 | 0.922 ± 0.005 | 0.979 ± 0.002 | 0.871 ± 0.006 |
| LGBM-G (real view) | 0.917 ± 0.006 | 0.977 ± 0.002 | 0.832 ± 0.007 | 0.937 ± 0.003 | 0.984 ± 0.003 | 0.863 ± 0.019 |
| LGBM-Grew (rewired) | 0.826 ± 0.006 | 0.950 ± 0.002 | 0.746 ± 0.010 | 0.923 ± 0.005 | 0.979 ± 0.004 | 0.869 ± 0.015 |
| LGBM-deg (supplementary) | 0.900 ± 0.008 | 0.972 ± 0.002 | 0.814 ± 0.009 | 0.928 ± 0.012 | 0.983 ± 0.003 | 0.872 ± 0.018 |
| OTRG | 0.923 ± 0.005 | 0.979 ± 0.001 | 0.842 ± 0.005 | 0.935 ± 0.005 | 0.984 ± 0.002 | 0.874 ± 0.010 |
| Constant gate | 0.921 ± 0.005 | 0.978 ± 0.002 | 0.839 ± 0.007 | 0.936 ± 0.005 | 0.984 ± 0.002 | 0.871 ± 0.013 |
| Equal average | 0.915 ± 0.005 | 0.975 ± 0.001 | 0.834 ± 0.005 | 0.934 ± 0.005 | 0.984 ± 0.002 | 0.876 ± 0.010 |
| Logistic stacking | 0.923 ± 0.005 | 0.979 ± 0.002 | 0.842 ± 0.006 | 0.937 ± 0.004 | 0.985 ± 0.002 | 0.867 ± 0.017 |
| Structural-only gate | 0.922 ± 0.005 | 0.978 ± 0.001 | 0.841 ± 0.006 | 0.935 ± 0.005 | 0.984 ± 0.001 | 0.874 ± 0.008 |
| Gate over node and rewired experts | 0.876 ± 0.005 | 0.964 ± 0.002 | 0.792 ± 0.005 | 0.926 ± 0.006 | 0.981 ± 0.002 | 0.870 ± 0.009 |

Calibration after Platt scaling:

- YelpChi: Brier scores lie between 0.033 and 0.053 and ECE between 0.009 and 0.011. OTRG has Brier 0.0334 and ECE 0.0091.
- Amazon: Brier 0.0176–0.0193 and ECE 0.0077–0.0091 for all methods.

**Graph view.**

- **YelpChi.**
  - The real view improves on the node-only model by +0.063 PR-AUC (split-seed SD 0.005, 5/5 seeds, Holm p = 1e-4; bootstrap 95% CI +0.061 to +0.077) and by +0.064 F1.
  - Against the configuration-model rewired control it improves by +0.091 PR-AUC (5/5, Holm p < 1e-4) and +0.086 F1.
  - The rewired view is worse than no view at all: −0.029 PR-AUC (5/5 seeds, Holm p = 3e-4). This matches the direction observed on Elliptic.
  - The degree-only control explains a large part of the raw gain: +0.046 PR-AUC over the node-only model.
  - The real view still adds +0.017 PR-AUC over degree-only (5/5, Holm p = 0.005; bootstrap CI +0.015 to +0.025) and +0.018 F1 (5/5).
- **Amazon.**
  - The real view improves on the node-only model by +0.015 PR-AUC (5/5, Holm p = 0.008; bootstrap CI +0.008 to +0.028).
  - It improves on the rewired control by +0.014 PR-AUC (5/5, Holm p = 0.004; bootstrap CI +0.005 to +0.027).
  - The rewired view equals the node-only model (+0.001).
  - Against degree-only the gain is +0.009 PR-AUC (4/5, Holm p = 0.82, bootstrap Holm p = 0.066), so it is not significant.
  - F1 differences between views are all within ±0.009 and non-significant, because seed-to-seed F1 SD is 0.01–0.02 on a test set with only 328 frauds.

**Gating.**

- **YelpChi.** The gate is genuinely non-constant.
  - The mean test weight on the node expert is 0.32–0.39 across seeds, with the 10th and 90th percentiles at about 0.09–0.15 and 0.54–0.65.
  - The largest standardised coefficient is 0.65–1.18, and the selected penalty is 0 or 1e-4 in four of five seeds.
  - The constant gate puts 0.16–0.21 weight on the node expert.
  - Against the constant gate, OTRG gains +0.0017 PR-AUC (5/5 seeds, split-seed Holm p = 0.043, but split-42 bootstrap +0.0002 with CI −0.0008 to +0.0012) and +0.0025 F1 (Holm p = 0.09).
  - Logistic stacking is slightly better than OTRG (−0.0008 PR-AUC for OTRG, Holm p = 0.043).
  - Equal averaging is worse than OTRG by 0.008 PR-AUC (5/5, Holm p < 1e-4) and 0.0075 F1 (Holm p = 0.011).
  - OTRG beats the graph expert alone by +0.006 PR-AUC and +0.009 F1 (Holm p = 0.015 and 0.024). This is the usual gain from mixing two experts, and the constant gate and stacking obtain it equally.
- **Amazon.** All combiners are statistically indistinguishable.
  - OTRG against the constant gate: −0.0005 PR-AUC and +0.003 F1.
  - Against stacking: −0.0016 PR-AUC.
  - Against averaging: +0.0013 PR-AUC.
  - The gate is close to constant in two seeds, with the selected penalty at 1e-2 and coefficients up to 0.05. In the others it varies moderately.
  - OTRG does not beat the graph expert alone (−0.0014 PR-AUC).

**Gate with a rewired second expert.** The Elliptic finding that the gate falls back entirely onto the node expert is only partly reproduced.

- On YelpChi the gate places a mean weight of 0.41–0.58 on the node expert, and the mixture (0.876 PR-AUC) is better than both the node expert (0.854) and the rewired expert (0.826). The rewired expert carries useful degree features, so falling back to the node expert would be the wrong choice.
- On Amazon the gate weight on the node expert is 0.36–0.97, and the mixture matches the node expert.

## 6. Do the Elliptic conclusions replicate?

**Graph view vs rewired control: replicates on both datasets.** On both graphs, label-free propagated neighbourhood features given to LightGBM beat a degree-preserving rewired graph, and the rewired view is no better than (YelpChi: worse than) no view.

Two qualifications apply:

- The size of the effect depends strongly on the graph. On YelpChi it is large, helped by the near-perfect label homophily of the same-user relation. On Amazon it is small.
- A considerable share of what a naive comparison would credit to "graph structure" is degree information. Only on YelpChi does topology add significantly beyond degree and dispersion.

These datasets provide no evidence about the temporal question. On Amazon the view raises PR-AUC but not F1, the opposite of the F1-led gain on Elliptic.

**Gate vs simpler combiners: replicates in substance.** The reliability gate, the constant gate and logistic stacking are practically equivalent on both datasets: differences are at most 0.002 PR-AUC, and stacking is nominally best on both.

The YelpChi result strengthens the negative conclusion. Here the gate does learn per-node routing, yet that routing yields only a 0.0017 PR-AUC advantage over a constant weight. The improvement is consistent across split seeds but is not resolved by the bootstrap on a single split, and it is far below the gains that matter in practice.

Equivalence with equal averaging holds on Amazon. It fails on YelpChi because the two experts differ greatly in quality there, and the best weight is far from 0.5. The manuscript should say "a constant learned gate or stacking" rather than "averaging" if it generalises this conclusion.

**Out-of-time behaviour: not testable here.** These graphs have no time dimension, so they cannot support or refute the claim that out-of-time gating handles temporal shift.

## 7. Limitations

- Both datasets are transductive, random-split benchmarks with derived labels:
  - Yelp's filter for YelpChi, which Rayana and Akoglu call "near" ground truth;
  - helpful-vote thresholds for Amazon, where one feature alone reaches ROC-AUC 0.883 and features may share information with the label definition.
- YelpChi node indices are label-ordered. This invalidated the first rewiring control and would invalidate any method that uses node order.
- The Amazon rewiring loses up to 15% of edges when duplicates are collapsed.
- Five split seeds give low power for seed-level tests.
- The bootstrap uses one split and ignores dependence within the graph.
- The degree-only control was added after inspecting dev-set results.
- Total lock time was about 55 minutes, somewhat above the 45-minute target, because the rewired control was rerun:
  - YelpChi main run 20.4 min;
  - Amazon main run 1.5 min;
  - rewired-v2 rerun 9.2 + 0.7 min;
  - degree-only control about 2.5 min;
  - two evaluation passes of about 6–8 min each.

## 8. Commands and files

Run from any directory (I used `/tmp`), with the shared lock:

```
flock /tmp/cpu.lock python3 -I experiments/second_dataset/audit_convert.py data/second_dataset/ext_yelp/YelpChi.mat yelp data/second_dataset/cache
flock /tmp/cpu.lock python3 -I experiments/second_dataset/audit_convert.py data/second_dataset/ext_amazon/Amazon.mat amazon data/second_dataset/cache
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_second.py yelp      # first run used one-sided rewiring (superseded parts renamed *-onesided)
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_second.py amazon
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_second.py yelp rewired_only     # configuration-model rewiring (current code)
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_second.py amazon rewired_only
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_degree_only.py yelp
flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_degree_only.py amazon
flock /tmp/cpu.lock python3 -I experiments/second_dataset/evaluate_second.py yelp amazon
```

Absolute paths were used in practice; see `experiments/second_dataset/chain*.sh`. With the current code, a fresh `run_second.py <ds>` already uses configuration-model rewiring, so the `rewired_only` step is needed only to reproduce the history.

**Outputs**

- Main results: `results/second_dataset/summary.csv`, `stats.csv` and `metrics_per_split.csv`.
- Fitted configuration: `{yelp,amazon}_run_meta*.json` and `{yelp,amazon}_rewiring*.json`.
- Predictions and split indices: `results/second_dataset/preds/`.
- Superseded results: `summary_onesided_rewiring_INVALID.csv` and `stats_onesided_rewiring_INVALID.csv`, kept as a record of the discarded control.

## 9. GNN baselines

*Added in response to the round-2 reviewer comment that the second-dataset comparison contained no member of the model family the paper argues against.*

**What was done.** `experiments/second_dataset/run_gnn_second.py` trains two message-passing GNNs on the same five stratified 40/20/40 splits (split seeds 42, 1, 7, 123, 2024) and the same raw node features (32 for YelpChi, 25 for Amazon, from `data/second_dataset/cache/{ds}_clean.npz`) as the tree pipeline. Because `run_second.py` is a top-level script that trains on import, its split code cannot be imported; the script therefore loads the split files that `run_second.py` wrote (`results/second_dataset/preds/{ds}_split{s}_idx.npz`), re-executes the same two `train_test_split` calls, and asserts that the train, dev and test indices (and labels) are identical for every split. All ten assertions passed. The first model is GraphSAGE with two layers, a mean aggregator, hidden width 64 and dropout 0.3 (the `SAGE` class of `baselines/gnn_v2.py`, i.e. separate self and neighbour weights), run on the homogeneous union of the three relations, built in the script from R-U-R, R-T-R and R-S-R for YelpChi and from U-P-U, U-S-U and U-V-U for Amazon. Amazon's supplied `homo` matrix was not used; the union built here has 4,417,576 undirected edges, which equals the union count found in the audit (Section 3). The second model, labelled RSAGE, is a relation-aware variant in which each layer computes a separate mean aggregation per relation and applies a linear map to the concatenation of the node's own representation and the three relation means, which is equivalent to an R-GCN/RGraphSAGE-style layer h' = W_self h + sum_r W_r mean_{N_r(v)} h with the same widths and dropout. Both are trained full-batch in plain PyTorch with sparse row-normalised adjacencies on two threads. Unlabelled Amazon users take part in message passing as feature carriers only, and labels are never propagated. Features are standardised with training-node statistics.

Training uses class-weighted cross-entropy with balanced weights computed from the training labels (n_train / (2 n_class); this weighting rule was fixed before training because the brief did not specify one), Adam with learning rate 1e-2 and weight decay 5e-4, at most 300 epochs, and early stopping on dev PR-AUC evaluated every epoch with patience 30, keeping the best-epoch weights. The model seed equals the split seed. Dev predictions from the train-only model are used to fit Platt scaling and choose the F1-optimal threshold with exactly the functions used by `evaluate_second.py` (`src.evaluation.Platt`, `best_f1_threshold`, `metrics`). Mirroring the tree pipeline's refit on train plus dev, the test model is retrained on train plus dev for the selected number of epochs with the same seed, and its test predictions are scored once with the frozen Platt map and threshold. No architectural or training choice was tuned, and none used test data.

**Commands.** `experiments/second_dataset/chain_gnn.sh` ran, from `/tmp`, `flock /tmp/cpu.lock python3 -I experiments/second_dataset/run_gnn_second.py amazon` and then the same for `yelp` (logs `logs/gnn_second_amazon.log`, `logs/gnn_second_yelp.log`, `logs/gnn_second_chain.log`). The summary was produced without the lock by `python3 -I experiments/second_dataset/summarise_gnn_second.py`. Outputs: `results/second_dataset/gnn_metrics_per_split.csv`, `gnn_summary.csv`, `gnn_paired.csv` (paired differences against LGBM-G and OTRG taken from `metrics_per_split.csv`, with sign counts, paired t-tests and Holm correction over the four comparisons within each dataset and metric), `gnn_paired_per_split.csv`, `gnn_{yelp,amazon}_meta.json` (selected epochs, dev curves, class weights, timings) and the prediction files `results/second_dataset/preds_gnn/{ds}_{GraphSAGE,RSAGE}_split{s}.npz` with arrays `dev` and `test`.

**Table 2. GNN test results, mean ± SD over five split seeds (LGBM-G and OTRG from Table 1 for reference)**

| Method | YelpChi PR-AUC | YelpChi ROC-AUC | YelpChi F1 | Amazon PR-AUC | Amazon ROC-AUC | Amazon F1 |
|---|---|---|---|---|---|---|
| GraphSAGE (union graph) | 0.703 ± 0.004 | 0.902 ± 0.003 | 0.630 ± 0.005 | 0.912 ± 0.013 | 0.979 ± 0.003 | 0.868 ± 0.011 |
| RSAGE (relation-aware) | 0.780 ± 0.007 | 0.937 ± 0.002 | 0.700 ± 0.008 | 0.912 ± 0.011 | 0.979 ± 0.003 | 0.851 ± 0.019 |
| LGBM-G | 0.917 ± 0.006 | 0.977 ± 0.002 | 0.832 ± 0.007 | 0.937 ± 0.003 | 0.984 ± 0.003 | 0.863 ± 0.019 |
| OTRG | 0.923 ± 0.005 | 0.979 ± 0.001 | 0.842 ± 0.005 | 0.935 ± 0.005 | 0.984 ± 0.002 | 0.874 ± 0.010 |

**Paired differences (GNN minus reference, mean over splits; splits on which the GNN is better in brackets).** On YelpChi both GNNs are far behind on every split. GraphSAGE trails LGBM-G by −0.214 PR-AUC, −0.075 ROC-AUC and −0.202 F1, and trails OTRG by −0.220, −0.077 and −0.212 (0/5 in every case). RSAGE trails LGBM-G by −0.137 PR-AUC, −0.041 ROC-AUC and −0.133 F1, and OTRG by −0.142, −0.042 and −0.142 (0/5 in every case). All twelve YelpChi comparisons have Holm-adjusted paired-t p ≤ 1.3e-5. On Amazon the gap is smaller but the direction is the same for ranking metrics. GraphSAGE is −0.025 PR-AUC behind LGBM-G (0/5, Holm p = 0.027) and −0.024 behind OTRG (0/5, Holm p = 0.027). RSAGE is −0.025 behind LGBM-G (0/5, Holm p = 0.025) and −0.024 behind OTRG (0/5, Holm p = 0.027). ROC-AUC differences are −0.005 to −0.006 (GNN better on 1/5 splits against LGBM-G and 0/5 against OTRG; Holm p 0.049–0.078). F1 on Amazon does not separate the models: GraphSAGE versus LGBM-G is +0.005 (2/5, Holm p = 0.75) and versus OTRG −0.006 (1/5), RSAGE versus LGBM-G −0.012 (1/5) and versus OTRG −0.023 (1/5, Holm p = 0.46), consistent with the seed-to-seed F1 noise noted in Section 5. Both GNNs are also below the node-only LightGBM of Table 1 in PR-AUC (YelpChi 0.854, Amazon 0.922), so in this setting message passing as implemented here does not even recover the information that a tree model extracts from the node features alone. The relation-aware layer helps clearly on YelpChi (+0.077 PR-AUC over GraphSAGE on the union), as expected given that the three relations differ greatly in homophily (Section 3) and the union mixes the near-pure R-U-R neighbourhood with the much larger, less homophilous R-S-R and R-T-R neighbourhoods; on Amazon the two GNNs are indistinguishable in PR-AUC.

**Caveats.** First, on YelpChi the fixed 300-epoch cap was binding: GraphSAGE reached the cap on four of five splits (selected epochs 214–296) and RSAGE on three (176–297), and dev PR-AUC was still rising slowly at the cap (for GraphSAGE on split 42 it went from 0.666 at epoch 200 to 0.679 at its best epoch 290). The YelpChi GNN numbers are therefore those of models trained under the stated budget, not of fully converged models; given the size of the gap (more than 0.13 PR-AUC) and the slow rate of improvement it is very unlikely that convergence alone would close it, but a modest further gain cannot be excluded. On Amazon early stopping triggered on every split (selected epochs 55–134). Second, no hyperparameters were tuned for either GNN; the configuration was the one specified for the revision, so these are standard untuned baselines rather than the strongest possible GNNs, and specialised fraud GNNs (for instance CARE-GNN or other camouflage-aware models) were not run. Third, the GNNs see only the raw node features through learned aggregation, whereas LGBM-G sees hand-built propagated means, per-relation degrees and dispersion; part of the gap may reflect the tree model's use of degree information, which the mean aggregator does not expose directly (Section 5 showed degree features alone are worth +0.046 PR-AUC on YelpChi). Fourth, the compute budget was exceeded: the lock was held for 978 s (Amazon) plus 2,325 s (YelpChi), about 55 minutes in total against a 40-minute target, because most YelpChi runs went to the epoch cap and the train-plus-dev refit doubles the training cost; nothing was rerun.

**Verdict.** The reviewer's point is addressed: the second-dataset comparison now includes a homogeneous GraphSAGE and a relation-aware GraphSAGE trained on identical splits and features. Under the stated protocol both are clearly worse than graph-augmented LightGBM and OTRG on YelpChi on every split and every metric, and worse in PR-AUC (5/5 splits) and ROC-AUC on Amazon, where F1 differences are not resolvable. The result supports the paper's position that, on these benchmarks, feature-engineered graph views fed to gradient-boosted trees are a stronger baseline than standard message-passing GNNs, subject to the caveats on untuned GNN hyperparameters and the binding epoch cap on YelpChi.
