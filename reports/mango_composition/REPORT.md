# Virtual Population Lab — Report

_Auto-generated on 2026-09-21 13:11 from `mango_composition`. Re-run `make run CONFIG=mango_composition` to refresh._

## Objective

Compare modeling engines on how well each generates a synthetic population that preserves the real population's feature distributions, correlations, and uncertainty.

## Data

- Source file: `data/cleaned-mango-composition.csv`
- Source dataset: https://data.mendeley.com/datasets/b9d6s7hr33/1
- Features (3): VitaminC, TA, SSC
- Rows after cleaning: 186 — split into 130 train / 56 test (`TEST_SIZE=0.3`)

## Methodology

**Physics-Informed Monte Carlo** (`src.generators.physics_mc_generator.generate`)

Generates synthetic rows via forward Monte Carlo sampling over a
caller-supplied causal graph:

- Each variable in `root_variables` is drawn from its own real marginal
  distribution (assumed Gaussian).
- Each (parent, child) edge in `causal_graph` draws `child` from a
  linear relationship fit on the real data, plus residual noise sampled
  at that fit's real residual std.

Every conditional here is a Gaussian we can sample directly, so plain
ancestral Monte Carlo sampling (this function) is exact — there's no
intractable distribution to approximate, so no need for MCMC.

The causal structure and roots are dataset knowledge supplied by the caller
(see src/configs/) — this function has no dataset-specific assumptions baked
in, so it works unchanged for a different set of features and relationships.

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
| Physics-Informed Monte Carlo | **0.0772** | **0.0939** |
| Mcmc | 0.3145 | 0.1046 |
| Regression | 0.1626 | 0.1254 |
| Variational Autoencoder | 0.1266 | 0.1367 |
| Physics-Informed VAE | 0.1893 | 0.1006 |

(Lower is better for both metrics; bold = best.)

## Findings

- Best **Correlation Distance (Euclidean)**: Physics-Informed Monte Carlo (0.0772)
- Best **Mean KS Statistic**: Physics-Informed Monte Carlo (0.0939)

Per-feature marginal fit (two-sample KS test, real vs. generated; lower ks_stat / higher p_value = closer):

- **Physics-Informed Monte Carlo**: 0/3 features statistically distinguishable from real (p < 0.05)
- **Mcmc**: 0/3 features statistically distinguishable from real (p < 0.05)
- **Regression**: 0/3 features statistically distinguishable from real (p < 0.05)
- **Variational Autoencoder**: 0/3 features statistically distinguishable from real (p < 0.05)
- **Physics-Informed VAE**: 0/3 features statistically distinguishable from real (p < 0.05)

## Per-Feature KS Statistic

Lower = closer to real; bold = best per feature.

| Feature | Physics-Informed Monte Carlo | Mcmc | Regression | Variational Autoencoder | Physics-Informed VAE |
|---|---|---|---|---|---|
| VitaminC | **0.0901** | 0.1254 | 0.1306 | 0.1207 | 0.1154 |
| TA | **0.0904** | 0.1114 | 0.1394 | 0.1574 | 0.0984 |
| SSC | 0.1010 | **0.0770** | 0.1061 | 0.1320 | 0.0880 |

## Feature Spread Comparison

**Correlation Distance (Euclidean)** is scale-invariant — Pearson correlation ignores absolute spread, so a method can score well on it while its population is uniformly shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is narrower than real).

| Feature | Real Std | Physics-Informed Monte Carlo | Mcmc | Regression | Variational Autoencoder | Physics-Informed VAE |
|---|---|---|---|---|---|---|
| VitaminC | 13.253 | 14.439 (1.09x) | 13.569 (1.02x) | 15.932 (1.20x) | 15.368 (1.16x) | 13.828 (1.04x) |
| TA | 160.170 | 169.963 (1.06x) | 167.245 (1.04x) | 212.404 (1.33x) | 191.136 (1.19x) | 170.431 (1.06x) |
| SSC | 3.973 | 3.893 (0.98x) | 3.835 (0.97x) | 4.846 (1.22x) | 4.497 (1.13x) | 3.873 (0.97x) |

## Generalization Check

Correlation distance for each engine's synthetic data against the train split it was fit on vs. the held-out test split it's actually scored on. A small gap is evidence results generalize rather than matching training-set noise a real (unseen) population wouldn't share.

| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |
|---|---|---|---|
| Physics-Informed Monte Carlo | 0.3319 | 0.0772 | 0.2546 |
| Mcmc | 0.0790 | 0.3145 | 0.2355 |
| Regression | 0.3835 | 0.1626 | 0.2209 |
| Variational Autoencoder | 0.1931 | 0.1266 | 0.0666 |
| Physics-Informed VAE | 0.0949 | 0.1893 | 0.0944 |

## Uncertainty Calibration (Coverage)

Tests the *calibrated uncertainty* claim directly. For each feature, the central 90% interval of each engine's generated population is formed, and we measure the fraction of real held-out values that fall inside it (averaged over features). Well-calibrated ⇒ coverage ≈ the nominal 0.90; **below** = over-confident (intervals too narrow), **above** = intervals too wide. `calibration_error` is the mean absolute gap between empirical and nominal coverage across interval levels from 0.10 to 0.95 (lower = better calibrated across the whole range).

| Engine | Coverage @ 90% (nominal 0.90) | Calibration Error |
|---|---|---|
| Physics-Informed Monte Carlo | 0.917 | **0.0341** |
| Mcmc | 0.893 | 0.0407 |
| Regression | 0.958 | 0.0831 |
| Variational Autoencoder | 0.940 | 0.0831 |
| Physics-Informed VAE | 0.899 | 0.0428 |

(Coverage closest to nominal and lowest calibration error = best-calibrated uncertainty.)

![Uncertainty calibration reliability curve](../../results/mango_composition/figures/marginals/coverage_calibration.png)

## Support-Aware Uncertainty (Near/Far Transfer)

Same-model transfer over `Cultivar`: one conditional-VAE ensemble is trained per region, then queried for the **held-out same region (NEAR)** vs a **different region (FAR)**. A model that 'knows what it doesn't know' has ensemble **disagreement** that widens for the unseen region (FAR > NEAR) while fidelity and coverage degrade. `coverage-conformal` recalibrates each interval's width on the NEAR held-out reals (target 0.90) and transfers that width off-support — it fixes NEAR coverage and partially closes the FAR gap (the residual is genuine distribution shift). (From the latest `make loro` run.)

| Train region | disagreement FAR / NEAR | coverage FAR / NEAR | coverage-conformal FAR / NEAR | corr FAR / NEAR |
|---|---|---|---|---|
| Cengkir | 0.0867 / 0.0606 | 0.460 / 0.524 | 0.711 / 0.810 | 0.528 / 0.994 |
| Kent | 0.1102 / 0.0621 | 0.703 / 0.911 | 0.648 / 0.911 | 0.699 / 0.395 |
| Kweni | 0.1202 / 0.0787 | 0.806 / 0.818 | 0.858 / 0.939 | 0.616 / 0.991 |
| Palmer | 0.1190 / 0.0803 | 0.837 / 0.930 | 0.770 / 0.895 | 0.545 / 0.895 |
| **MEAN** | 0.1090 / 0.0704 | 0.701 / 0.796 | 0.747 / 0.889 | 0.597 / 0.819 |

Mean ensemble disagreement is **+55%** for unseen vs held-out same regions — widens off-support.

![Near/far transfer](../../results/mango_composition/figures/loro/loro_Cultivar.png)

## Figures

### Joint variability (correlation)

**Correlation matrices**

![Real correlation matrix](../../results/mango_composition/figures/correlation/correlation_real.png)
![Physics-Informed Monte Carlo correlation matrix](../../results/mango_composition/figures/correlation/correlation_physics_mc.png)
![Mcmc correlation matrix](../../results/mango_composition/figures/correlation/correlation_mcmc.png)
![Regression correlation matrix](../../results/mango_composition/figures/correlation/correlation_regression.png)
![Variational Autoencoder correlation matrix](../../results/mango_composition/figures/correlation/correlation_vae.png)
![Physics-Informed VAE correlation matrix](../../results/mango_composition/figures/correlation/correlation_hybrid_vae.png)

**PCA projection — real vs. each method, plus all combined**

![PCA projection of real vs generated populations](../../results/mango_composition/figures/correlation/pca_real_vs_generated.png)

**PCA projection — each population shown individually, no overlay**

![PCA projection of each population individually](../../results/mango_composition/figures/correlation/pca_individual.png)

### Marginal distributions (KS)

**KS statistic by feature and method**

![KS statistic heatmap](../../results/mango_composition/figures/marginals/ks_heatmap.png)

**Empirical CDFs — real vs. generated, per feature**

![ECDF overlay per feature](../../results/mango_composition/figures/marginals/ecdf_overlay.png)

**P-value by feature and method**

![P-value heatmap](../../results/mango_composition/figures/marginals/pvalue_heatmap.png)

**Volcano plot — effect size vs. significance**

![Volcano plot](../../results/mango_composition/figures/marginals/volcano_plot.png)

---

Raw data: `results/mango_composition/summary_metrics.csv`, `results/mango_composition/ks_marginals.csv`, `results/mango_composition/generalization_gap.csv`, `results/mango_composition/synthetic_data/`.
