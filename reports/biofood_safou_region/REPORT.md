# Virtual Population Lab — Report

_Auto-generated on 2026-09-21 17:46 from `biofood_safou_region`. Re-run `make run CONFIG=biofood_safou_region` to refresh._

## Objective

Compare modeling engines on how well each generates a synthetic population that preserves the real population's feature distributions, correlations, and uncertainty.

## Data

- Source file: `data/cleaned-biofood-safou-region.csv`
- Source dataset: https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/
- Features (4): Water, Fat, Palmitic, Stearic
- Rows after cleaning: 41 — split into 28 train / 13 test (`TEST_SIZE=0.3`)

## Methodology

**Physics-Informed Monte Carlo** (`src.generators.physics_mc_generator.generate`)

Generates synthetic rows via forward Monte Carlo sampling over a
caller-supplied structural graph:

- Each variable in `root_variables` is drawn from its own real marginal
  distribution (assumed Gaussian).
- Each (parent, child) edge in `causal_graph` draws `child` from a
  linear relationship fit on the real data, plus residual noise sampled
  at that fit's real residual std.

Every conditional here is a Gaussian we can sample directly, so plain
ancestral Monte Carlo sampling (this function) is exact — there's no
intractable distribution to approximate, so no need for MCMC.

Where the caller supplies `constraints` — genuine conservation / mass-balance
laws  sum_i w_i x_i <= bound  in source units, not fitted from data — the
generated population is projected onto the feasible region with the same hard
feasibility projection the PI-VAE uses, so those physical laws hold exactly
(0% violations). This is what makes the engine physics-informed rather than
only structure-informed; with no constraints it is plain structural Monte
Carlo. The structural graph, roots and constraints are all dataset knowledge
supplied by the caller (see src/configs/), so the function works unchanged
for a different set of features and relationships.

**Mcmc** (`src.generators.mcmc_generator.generate`)

Generate `n_samples` synthetic rows by Gaussian-copula Gibbs MCMC.

`df`      real data (in source units); `features` the columns to model.
`constraints`  optional list of callables row_dict -> bool (True = feasible).
               Infeasible Gibbs draws are rejected and redrawn (constrained
               MCMC); pass None for the unconstrained sampler.
`chain_length` total Gibbs sweeps; defaults to burn_in + thin*n_samples.
`burn_in`, `thin`  standard MCMC burn-in and thinning.
`max_reject`   per-sample redraw cap before the draw is accepted anyway
               (keeps a hard constraint from stalling the chain).

Returns a DataFrame in real units with the real marginals reproduced and the
data's joint correlation preserved through the copula.

**Regression** (`src.generators.regression_generator.generate`)

For each feature, fits one RandomForestRegressor (on train_df) predicting it
from the other features. Synthetic rows are built from a base row's
prediction plus noise, so outputs vary instead of collapsing to a
deterministic point estimate.

Both the base rows and the residual noise come from out-of-bag (OOB)
predictions rather than a held-out split. A random forest bags each tree on a
bootstrap sample, so roughly a third of the trees never saw any given
training row; `oob_prediction_` averages only those trees, giving a
leakage-free prediction for every training row. That lets this generator seed
itself entirely from train_df, never touching the evaluation set, with no
data sacrificed to a separate base split. It also makes the residuals
out-of-sample rather than in-sample, so the bootstrapped noise reflects
predictive uncertainty instead of understating it.

**Variational Autoencoder** (`src.generators.vae_generator.generate`)

Trains on `x` (expected to already be reasonably scaled) and samples
`n_samples` new rows from the prior. Pass a fitted `scaler` only if `x`
needs to be inverse-transformed back to source units afterwards —
leave it None when the source data is already standardized, since
there's no original unscaled space to return to.

Loss is the standard VAE ELBO (reconstruction MSE plus the analytic KL
divergence between the approximate posterior N(mu, sigma^2) and the
standard normal prior, weighted by `beta`), plus a covariance-matching
term weighted by `cov_weight`: the squared difference between the
reconstructed batch's covariance matrix and the real batch's. The ELBO
alone has no direct incentive to get cross-feature correlations right — it
only rewards accurate per-point reconstruction — so this term applies
direct pressure toward preserving joint structure. `beta` is linearly
annealed from 0 up to its target value over the first `kl_warmup_frac` of
training — without this, the KL term can dominate before reconstruction has
learned anything, collapsing the model onto near-zero variance (i.e.
generated samples clustering tightly around the mean).

With `use_minibatch=True`, each epoch shuffles the data and takes one
gradient step per `batch_size` chunk instead of one step over the whole
dataset — the added stochastic noise can help escape a plateaued loss
that full-batch descent gets stuck in. Set `use_minibatch=False` for
very small datasets, where a mini-batch may be too small to estimate a
stable gradient at all.

Training stops early once the loss hasn't improved for `patience`
epochs, checked only after the KL warmup finishes — beta is still
rising during warmup, so the loss naturally shifts then regardless of
real convergence, and checking during that window would trigger a
spurious early stop.

`hidden_dim` and `dropout` control model capacity — the defaults (128,
0.0) fit a few-thousand-row dataset comfortably, but on a small dataset
(dozens to low hundreds of rows) that many parameters can memorize
training noise instead of generalizing. Shrink `hidden_dim` and/or add
dropout (e.g. 0.1-0.3) if the generalization check shows a large
train/test gap.

`free_bits` floors the KL cost per latent dimension: once a dimension's
KL drops below this many nats, it stops contributing gradient to the KL
term (though reconstruction can still make it informative). Without
this (free_bits=0.0), KL annealing alone can still let every dimension
collapse fully to the prior on some datasets — recon_loss plateauing
at ~1.0 on standardized data (i.e. no better than always predicting the
mean) with KL near zero is the signature to watch for. Try 0.5-2.0 if
that happens.

**Physics-Informed VAE** (`src.generators.hybrid_vae_generator.generate`)

The physics-informed VAE (PI-VAE): the same generative model as the plain
VAE, plus a physics-consistency loss that injects the caller-supplied causal
graph into training. It keeps the VAE's data-driven strengths (a learned
latent joint, novel-individual sampling, no handcrafted marginals) while
borrowing the physics-informed Monte Carlo generator's domain knowledge: the
fitted linear-Gaussian relationship on each causal edge.

Each (parent, child) edge in `causal_graph` is fit once on the real data
(slope, intercept, residual variance — the same conditionals the physics-MC
generator samples). Those constants become a differentiable penalty (see
`_physics_loss`), weighted by `physics_weight`, that pushes each batch onto
the mechanistic relationships while matching their real residual spread, so
the constraint guides the generative manifold toward physically consistent
samples without re-introducing variance shrinkage.

Trains on `x` (expected already reasonably scaled) and samples `n_samples`
rows from the prior; pass a fitted `scaler` only if `x` needs
inverse-transforming back to source units. Loss is the standard VAE ELBO
(reconstruction MSE + analytic KL to the standard-normal prior, `beta`
linearly annealed from 0 over the first `kl_warmup_frac` of training to avoid
posterior collapse), plus a covariance-matching term (`cov_weight`) that
rewards preserving the full correlation matrix, plus the physics term
(`physics_weight`). `cov_weight` shapes the whole covariance structure from
data; `physics_weight` anchors the specific mechanistic edges an expert
asserts. Set `physics_weight=0.0` to recover the plain VAE.

A fourth term, weighted by `marginal_weight`, is the per-feature 1D
Wasserstein distance (see `_marginal_loss`) between samples drawn from the
prior and the real batch. It shapes every generated feature's whole
distribution onto real, including the causal-graph root variables that the
physics term leaves unconstrained; set `marginal_weight=0.0` to disable it.

`prior_type` chooses how latents are drawn at generation and for the
generated-sample loss terms: "standard" samples the N(0, I) prior,
"aggregate" samples the aggregate posterior over the training rows (see
`_decode_samples`).

`constrain_generated` routes the covariance and physics terms onto a batch of
freshly generated rows instead of the reconstructions, so those constraints
shape the population actually sampled at generation. The default (False)
keeps them on reconstructions, matching the plain VAE's convention.

`calibrate_marginals` applies the empirical-copula post-step (see
`_calibrate_marginals`): each feature is mapped onto the real training
marginal at matching quantiles, forcing every marginal to match real almost
exactly while preserving the learned rank dependence. A run using it is
marginal-calibrated rather than purely learned. Default False.

`use_minibatch`, `batch_size`, `patience`, `hidden_dim`, `dropout`, and
`free_bits` behave as in the plain VAE generator — see its docstring for the
capacity, collapse, and early-stopping trade-offs.

## Results

| Engine | Correlation Distance (Euclidean) | Mean KS Statistic |
|---|---|---|
| Physics-Informed Monte Carlo | **0.3348** | **0.2489** |
| Mcmc | 0.5002 | 0.2706 |
| Regression | 0.5838 | 0.2924 |
| Variational Autoencoder | 0.5959 | 0.3090 |
| Physics-Informed VAE | 0.5596 | 0.2961 |

(Lower is better for both metrics; bold = best.)

## Findings

- Best **Correlation Distance (Euclidean)**: Physics-Informed Monte Carlo (0.3348)
- Best **Mean KS Statistic**: Physics-Informed Monte Carlo (0.2489)

Per-feature marginal fit (two-sample KS test, real vs. generated; lower ks_stat / higher p_value = closer):

- **Physics-Informed Monte Carlo**: 0/4 features statistically distinguishable from real (p < 0.05)
- **Mcmc**: 0/4 features statistically distinguishable from real (p < 0.05)
- **Regression**: 0/4 features statistically distinguishable from real (p < 0.05)
- **Variational Autoencoder**: 0/4 features statistically distinguishable from real (p < 0.05)
- **Physics-Informed VAE**: 0/4 features statistically distinguishable from real (p < 0.05)

## Per-Feature KS Statistic

Lower = closer to real; bold = best per feature.

| Feature | Physics-Informed Monte Carlo | Mcmc | Regression | Variational Autoencoder | Physics-Informed VAE |
|---|---|---|---|---|---|
| Water | **0.2535** | 0.3013 | 0.2984 | 0.3355 | 0.3353 |
| Fat | **0.2485** | 0.2793 | 0.2834 | 0.3355 | 0.2993 |
| Palmitic | **0.2385** | 0.2634 | 0.2715 | 0.3175 | 0.2934 |
| Stearic | 0.2552 | **0.2383** | 0.3163 | 0.2473 | 0.2563 |

## Feature Spread Comparison

**Correlation Distance (Euclidean)** is scale-invariant — Pearson correlation ignores absolute spread, so a method can score well on it while its population is uniformly shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is narrower than real).

| Feature | Real Std | Physics-Informed Monte Carlo | Mcmc | Regression | Variational Autoencoder | Physics-Informed VAE |
|---|---|---|---|---|---|---|
| Water | 11.395 | 9.915 (0.87x) | 9.501 (0.83x) | 9.431 (0.83x) | 7.627 (0.67x) | 9.456 (0.83x) |
| Fat | 10.947 | 9.144 (0.84x) | 9.295 (0.85x) | 9.346 (0.85x) | 7.064 (0.65x) | 9.155 (0.84x) |
| Palmitic | 4.135 | 3.612 (0.87x) | 3.685 (0.89x) | 3.699 (0.89x) | 2.796 (0.68x) | 3.684 (0.89x) |
| Stearic | 0.168 | 0.215 (1.28x) | 0.216 (1.28x) | 0.218 (1.30x) | 0.170 (1.01x) | 0.220 (1.31x) |

## Generalization Check

Correlation distance for each engine's synthetic data against the train split it was fit on vs. the held-out test split it's actually scored on. A small gap is evidence results generalize rather than matching training-set noise a real (unseen) population wouldn't share.

| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |
|---|---|---|---|
| Physics-Informed Monte Carlo | 0.1234 | 0.3348 | 0.2114 |
| Mcmc | 0.1410 | 0.5002 | 0.3592 |
| Regression | 0.2864 | 0.5838 | 0.2974 |
| Variational Autoencoder | 0.1702 | 0.5959 | 0.4257 |
| Physics-Informed VAE | 0.1278 | 0.5596 | 0.4318 |

## Uncertainty Calibration (Coverage)

Tests the *calibrated uncertainty* claim directly. For each feature, the central 90% interval of each engine's generated population is formed, and we measure the fraction of real held-out values that fall inside it (averaged over features). Well-calibrated ⇒ coverage ≈ the nominal 0.90; **below** = over-confident (intervals too narrow), **above** = intervals too wide. `calibration_error` is the mean absolute gap between empirical and nominal coverage across interval levels from 0.10 to 0.95 (lower = better calibrated across the whole range).

| Engine | Coverage @ 90% (nominal 0.90) | Calibration Error |
|---|---|---|
| Physics-Informed Monte Carlo | 0.962 | 0.1247 |
| Mcmc | 0.865 | **0.0989** |
| Regression | 0.904 | 0.1276 |
| Variational Autoencoder | 0.673 | 0.1960 |
| Physics-Informed VAE | 0.904 | 0.1255 |

(Coverage closest to nominal and lowest calibration error = best-calibrated uncertainty.)

![Uncertainty calibration reliability curve](../../results/biofood_safou_region/figures/marginals/coverage_calibration.png)

## Support-Aware Uncertainty (Near/Far Transfer)

Same-model transfer over `Region`: one conditional-VAE ensemble is trained per region, then queried for the **held-out same region (NEAR)** vs a **different region (FAR)**. A model that 'knows what it doesn't know' has ensemble **disagreement** that widens for the unseen region (FAR > NEAR) while fidelity and coverage degrade. `coverage-conformal` recalibrates each interval's width on the NEAR held-out reals (target 0.90) and transfers that width off-support — it fixes NEAR coverage and partially closes the FAR gap (the residual is genuine distribution shift). (From the latest `make loro` run.)

| Train region | disagreement FAR / NEAR | coverage FAR / NEAR | coverage-conformal FAR / NEAR | corr FAR / NEAR |
|---|---|---|---|---|
| Congo | 0.0577 / 0.0553 | 0.250 / 0.750 | 0.550 / 0.875 | 1.727 / 0.577 |
| Guinea | 0.1082 / 0.0692 | 0.405 / 0.429 | 0.714 / 0.571 | 1.345 / 0.137 |
| **MEAN** | 0.0830 / 0.0623 | 0.327 / 0.589 | 0.632 / 0.723 | 1.536 / 0.357 |

Mean ensemble disagreement is **+33%** for unseen vs held-out same regions — widens off-support.

![Near/far transfer](../../results/biofood_safou_region/figures/loro/loro_Region.png)

## Figures

### Joint variability (correlation)

**Correlation matrices**

![Real correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_real.png)
![Physics-Informed Monte Carlo correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_physics_mc.png)
![Mcmc correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_mcmc.png)
![Regression correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_regression.png)
![Variational Autoencoder correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_vae.png)
![Physics-Informed VAE correlation matrix](../../results/biofood_safou_region/figures/correlation/correlation_hybrid_vae.png)

**PCA projection — real vs. each method, plus all combined**

![PCA projection of real vs generated populations](../../results/biofood_safou_region/figures/correlation/pca_real_vs_generated.png)

**PCA projection — each population shown individually, no overlay**

![PCA projection of each population individually](../../results/biofood_safou_region/figures/correlation/pca_individual.png)

### Marginal distributions (KS)

**KS statistic by feature and method**

![KS statistic heatmap](../../results/biofood_safou_region/figures/marginals/ks_heatmap.png)

**Empirical CDFs — real vs. generated, per feature**

![ECDF overlay per feature](../../results/biofood_safou_region/figures/marginals/ecdf_overlay.png)

**P-value by feature and method**

![P-value heatmap](../../results/biofood_safou_region/figures/marginals/pvalue_heatmap.png)

**Volcano plot — effect size vs. significance**

![Volcano plot](../../results/biofood_safou_region/figures/marginals/volcano_plot.png)

---

Raw data: `results/biofood_safou_region/summary_metrics.csv`, `results/biofood_safou_region/ks_marginals.csv`, `results/biofood_safou_region/generalization_gap.csv`, `results/biofood_safou_region/synthetic_data/`.
