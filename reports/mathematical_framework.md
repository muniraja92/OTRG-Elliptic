# Mathematical framework (Agent 3 role)

Every equation below corresponds to code in `src/preprocessing.py` and `src/fusion.py`; the correspondence is checked in the integrity report.

## Data and task

The data form a sequence of directed graphs $\mathcal{G}_s=(\mathcal{V}_s,\mathcal{E}_s)$ for time steps $s=1,\dots,49$, with $\mathcal{E}_s\subseteq\mathcal{V}_s\times\mathcal{V}_s$ (no edge crosses steps). Node $i$ carries local features $\mathbf{x}^{L}_i\in\mathbb{R}^{93}$ and creator-supplied neighbour aggregates $\mathbf{x}^{A}_i\in\mathbb{R}^{72}$; write $\mathbf{x}_i=[\mathbf{x}^{L}_i;\mathbf{x}^{A}_i]\in\mathbb{R}^{165}$. A subset of nodes carries a label $y_i\in\{0,1\}$ (1 = illicit). The task is to estimate $\Pr(y_i=1\mid\cdot)$ for labelled nodes of future steps $s\ge 35$ from labelled nodes of steps $s\le 34$, and to produce calibrated probabilities and a binary decision.

## Neighbourhood view (label-free)

Let $A^{\mathrm{out}}_{ij}=1$ if $i$ pays $j$, and $A^{\mathrm{in}}=(A^{\mathrm{out}})^\top$. Define row-normalised operators $P^{\mathrm{d}}=D_{\mathrm{d}}^{-1}A^{\mathrm{d}}$ for $\mathrm{d}\in\{\mathrm{in},\mathrm{out}\}$, with $D_{\mathrm{d}}$ the diagonal degree matrix and rows of zero degree left at zero. With $X^{L}$ the stacked local features, the neighbourhood view is
$$\mathbf{g}_i=\big[(P^{\mathrm{out}}X^{L})_i;(P^{\mathrm{out}}P^{\mathrm{out}}X^{L})_i;(P^{\mathrm{in}}X^{L})_i;(P^{\mathrm{in}}P^{\mathrm{in}}X^{L})_i;\log(1+d^{\mathrm{in}}_i);\log(1+d^{\mathrm{out}}_i);\delta_i\big]\in\mathbb{R}^{375},$$
where $\delta_i=\frac{1}{93}\sum_{k}\sqrt{\mathbb{E}_{j\sim\mathcal{N}(i)}[x^{L}_{jk}{}^2]-\mathbb{E}_{j\sim\mathcal{N}(i)}[x^{L}_{jk}]^2}$ is the mean feature-wise standard deviation over the undirected one-hop neighbourhood $\mathcal{N}(i)$. Since every operator acts within one step, $\mathbf{g}_i$ depends only on same-step features and contains no labels.

## Experts

Two gradient-boosted tree ensembles are trained with the binary log-loss:
$$p^{\mathrm{node}}_i=f_{\theta}(\mathbf{x}_i),\qquad p^{\mathrm{graph}}_i=h_{\phi}([\mathbf{x}_i;\mathbf{g}_i]).$$
The graph expert sees the node's own features as well, so it is the "concatenation" model; the mixture decides per node how much of the graph-augmented model to trust. Gradient-boosting objectives and optimisation are standard (Ke et al., 2017) and are not claimed as contributions.

## Reliability signals

With $\ell(p)=\log\frac{p}{1-p}$ and $H(p)=-p\log p-(1-p)\log(1-p)$, the reliability vector is
$$\mathbf{r}_i=\big[\,|\ell(p^{\mathrm{node}}_i)-\ell(p^{\mathrm{graph}}_i)|,\;H(p^{\mathrm{node}}_i),\;H(p^{\mathrm{graph}}_i),\;\log(1+d^{\mathrm{in}}_i),\;\log(1+d^{\mathrm{out}}_i),\;\delta_i,\;\nu_i\,\big]\in\mathbb{R}^7,$$
where $\nu_i=\log\big(1+\tfrac{1}{93}\|\mathbf{z}_i-U U^\top\mathbf{z}_i\|^2\big)$ is a novelty score: $\mathbf{z}_i$ is $\mathbf{x}^L_i$ standardised with statistics of nodes from steps strictly before the block being scored, and $U\in\mathbb{R}^{93\times 20}$ spans the leading principal subspace of those earlier nodes. All seven signals are computable without labels at prediction time.

## Gated mixture

$$\alpha_i=\sigma(w_0+\mathbf{w}^\top\tilde{\mathbf{r}}_i),\qquad \hat p_i=\alpha_i\,p^{\mathrm{node}}_i+(1-\alpha_i)\,p^{\mathrm{graph}}_i,$$
with $\tilde{\mathbf{r}}_i$ the reliability vector standardised by its mean and standard deviation on the gate-fitting set. Because $\hat p_i$ is a convex combination of two probabilities it lies in $[\min(p^{\mathrm{node}}_i,p^{\mathrm{graph}}_i),\max(p^{\mathrm{node}}_i,p^{\mathrm{graph}}_i)]$, so the gate cannot produce a score more extreme than either expert, and its effect is bounded by the experts' disagreement.

## Out-of-time gate objective

Let $\mathcal{B}_1,\dots,\mathcal{B}_4$ be the labelled nodes of steps 15–19, 20–24, 25–29 and 30–34. For node $i\in\mathcal{B}_k$, let $p^{\mathrm{node},(-k)}_i$ and $p^{\mathrm{graph},(-k)}_i$ be predictions of experts trained only on labelled nodes of steps before $\mathcal{B}_k$. The gate parameters minimise the penalised negative log-likelihood on $\mathcal{F}=\mathcal{B}_1\cup\mathcal{B}_2\cup\mathcal{B}_3$:
$$\mathcal{L}(w_0,\mathbf{w})=-\frac{1}{|\mathcal{F}|}\sum_{i\in\mathcal{F}}\Big[y_i\log\hat p^{(-k)}_i+(1-y_i)\log(1-\hat p^{(-k)}_i)\Big]+\lambda\|\mathbf{w}\|_2^2,\qquad\lambda=10^{-2}.$$
This is the only new objective term; no auxiliary losses are used. Its gradient is
$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}}=-\frac{1}{|\mathcal{F}|}\sum_{i\in\mathcal{F}}\Big(\frac{y_i}{\hat p_i}-\frac{1-y_i}{1-\hat p_i}\Big)\big(p^{\mathrm{node}}_i-p^{\mathrm{graph}}_i\big)\,\alpha_i(1-\alpha_i)\,\tilde{\mathbf{r}}_i+2\lambda\mathbf{w},$$
with the analogous expression for $w_0$ without the penalty. The problem is solved with L-BFGS-B; the analytic gradient is verified numerically in `tests/test_units.py` (relative error about $10^{-8}$). The objective is smooth but not convex in general, so no global-optimality or convergence-rate claim is made beyond L-BFGS-B's usual local guarantees.

## Calibration and decision

A Platt map $q_i=\sigma(a\,\ell(\hat p_i)+b)$ is fitted on $\mathcal{B}_4$ (disjoint from $\mathcal{F}$), and the decision threshold $\tau$ maximises illicit-class F1 of $q$ on $\mathcal{B}_4$. Every baseline receives the same treatment from its own $\mathcal{B}_4$ predictions, so calibration is not a source of advantage for the proposed method.

## Algorithm

```
Input: graphs G_1..G_49, features, labels on steps 1..34
1  compute g_i for all nodes (same-step propagation)
2  for k = 1..4: train f, h on labelled steps < start(B_k); predict B_k  -> out-of-time p^node, p^graph
3  compute r_i on B_1..B_4 (novelty fitted on steps < start(B_k))
4  fit (w0, w) on B_1..B_3 by L-BFGS-B
5  fit Platt map and threshold tau on gated outputs for B_4
6  refit f, h on labelled steps 1..34; compute r_i for test nodes (novelty fitted on steps <= 34)
7  output q_i = Platt(alpha_i p^node_i + (1 - alpha_i) p^graph_i) and decision q_i >= tau
```

## Complexity

Propagation costs $O(|\mathcal{E}|\,d_L\,K)$ for $K=2$ hops and $d_L=93$, about $4.4\times10^7$ multiply-adds here. Each boosted expert costs $O(T\,n\,d)$ for $T$ trees with histogram binning; cross-fitting multiplies expert training by five (four blocks plus the final refit). The gate has eight parameters; fitting is $O(|\mathcal{F}|\cdot 8)$ per L-BFGS iteration and inference is $O(1)$ per node beyond the two expert predictions. Measured times are in `results/runtime_profile.csv`.
