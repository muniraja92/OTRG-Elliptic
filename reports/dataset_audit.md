# Dataset audit: Elliptic Bitcoin transaction data set

Audit date: 23 November 2026. Script: `src/data_audit.py`; machine-readable output: `data/dataset_audit.json` and `data/class_by_timestep.csv`.

## Provenance

The data set was introduced by Weber et al. (2019, arXiv:1908.02591, KDD 2019 Workshop on Anomaly Detection in Finance). Its canonical distribution is the Kaggle page "Elliptic Data Set" published by Elliptic; Kaggle was not reachable from the execution sandbox, so the three CSV files were obtained from the PyTorch Geometric mirror (`https://data.pyg.org/datasets/elliptic/`), which PyG's `EllipticBitcoinDataset` loader uses. The files are dated 31 July 2019 inside the archives, matching the release date of the original paper. SHA-256 digests of the extracted files are recorded in `data/dataset_audit.json` so that any reader who downloads the Kaggle copy can check byte identity. Licence: the licence on the Kaggle page could not be inspected from the sandbox and is therefore recorded as unverified; the data set is widely used for non-commercial research, but the licence must be confirmed on the Kaggle page before the manuscript's data-availability statement is finalised. No access control was bypassed.

## Structural verification

All counts match the published description exactly. The graph has 203,769 transaction nodes and 234,355 directed payment-flow edges, with no duplicate transaction identifiers, no duplicate edges, no self-loops, no isolated nodes and no missing feature values. The feature file has 165 anonymised numeric columns after the identifier and time step; following Weber et al., the first 93 are local transaction features and the remaining 72 are one-hop neighbour aggregates computed by the data creators. Labels are 4,545 illicit (class 1), 42,019 licit (class 2) and 157,205 unknown. The 46,564 labelled nodes are therefore 9.8% illicit overall, and the illicit share of labelled nodes varies from 0.28% (step 46) to 36.0% (step 13) across the 49 time steps.

## Leakage analysis

No edge connects transactions in different time steps (0 of 234,355 edges cross a step boundary). This is the single most important property for this study: the standard temporal split (train on steps 1 to 34, test on 35 to 49) yields 49 mutually disconnected subgraphs, so neither neighbourhood aggregation nor graph convolution can carry information from test-period nodes into training, or vice versa. Neighbourhood views in this study propagate features only, never labels, and are computed from same-step information that would be available when a time step is scored.

No single feature separates the classes trivially: the largest single-feature ROC-AUC on labelled nodes is 0.843 and no feature exceeds 0.9, so the task is not saturated by label recovery from one column. Thirty-four feature rows are exact duplicates of other rows when the identifier is dropped; among labelled nodes there are 11 duplicated feature vectors and none carries conflicting labels. Because duplicates are rare and the split is temporal, they cannot inflate test performance materially.

The time-step index itself is excluded from the model inputs. Weber et al. include it in their "local features", but the test period consists entirely of step values never seen in training, so a tree model would map every test step to the boundary value 34 and a linear model would extrapolate; neither behaviour is meaningful.

## Split design and its justification

The test set is the conventional one (steps 35 to 49, 16,670 labelled nodes), which keeps results comparable with Weber et al. and later work. Within the development period, a forward cross-fitting scheme produces out-of-time predictions for blocks 15–19, 20–24, 25–29 and 30–34, each from models trained only on earlier steps. Gate parameters are fitted on blocks 15–29; Platt scaling and the decision threshold are fitted on block 30–34; hyperparameters are selected by training on 1–29 and validating on 30–34. The test period is scored once after all choices are frozen.

## Known biases and limitations

The labelled subset is not a random sample of transactions: 77% of nodes are unknown, and labels were assigned by Elliptic heuristics, so "licit" means "attributed to a licit-category entity" rather than verified innocence. Features are anonymised, so attributions explain model behaviour in terms of feature indices but cannot be read as forensic statements. The test period contains the dark-market closure around step 43, after which very few illicit transactions are labelled (2 to 56 per step), so post-closure F1 rests on small counts and is reported alongside its uncertainty.

A second independent data set (Elliptic++ or Elliptic2) was not used: neither is served from a host reachable from the sandbox, and Elliptic++ reuses the same transactions, so it would not be independent at the transaction level. This is recorded as an external-validity limitation.
