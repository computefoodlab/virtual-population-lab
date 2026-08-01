# Virtual Population Lab — Report

_Auto-generated on 2026-07-23 21:50 from `apple_quality`. Re-run `make run CONFIG=apple_quality` to refresh._

## Objective

Compare modeling engines on how well each generates a synthetic population that preserves the real population's feature distributions, correlations, and uncertainty.

## Data

- Source file: `data/cleaned-scaled-apple-quality.csv`
- Source dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
- Features (7): Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity
- Rows after cleaning: 4000 — split into 2800 train / 1200 test (`TEST_SIZE=0.3`)

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

The causal structure and roots are dataset knowledge supplied by the
caller (see src/configs/apple_quality.py) — this function has no
dataset-specific assumptions baked in, so it works unchanged for a
different set of features/relationships.

**Regression** (`src.generators.regression_generator.generate`)

For each feature, fits one RandomForestRegressor (on train_df) predicting
it from the other features. Synthetic rows are built from a base apple's
other features plus noise, so outputs vary instead of collapsing to a
deterministic point estimate.

Both the base rows and the residual noise come from out-of-bag (OOB)
predictions rather than a held-out split. A random forest bags each tree on
a bootstrap sample, so ~1/3 of the trees never saw any given training row;
`oob_prediction_` averages only those trees, giving a leakage-free
prediction for every training row for free. That lets this generator seed
itself entirely from train_df — never touching the evaluation set — exactly
like the other engines, with no memorization leakage and no data sacrificed
to a separate base split. It also makes the residuals honest (out-of-sample
rather than overfit in-sample), so the bootstrapped noise reflects real
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
alone has no direct incentive to get cross-feature correlations right —
it only rewards accurate per-point reconstruction — so this term gives
the model direct pressure toward the metric this project actually scores
correlation preservation on. `beta` is linearly annealed from 0 up to
its target value over the first `kl_warmup_frac` of training — without
this, the KL term can dominate before reconstruction has learned
anything, collapsing the model onto near-zero variance (i.e. generated
samples clustering tightly around the mean).

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
0.0) fit apple_quality's 2800 rows comfortably, but on a small dataset
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

**Physics-Informed VAE (Hybrid)** (`src.generators.hybrid_vae_generator.generate`)

A hybrid physics-informed VAE: the same generative model as the plain
VAE, plus a physics-consistency loss that injects the caller-supplied
causal graph directly into training. This is the bridge between the two
other engines — it keeps the VAE's fully data-driven strengths (a learned
latent joint, novel-individual sampling, no handcrafted marginals) while
borrowing the physics-informed Monte Carlo generator's one piece of real
domain knowledge: the fitted linear-Gaussian relationship on each causal
edge.

Each (parent, child) edge in `causal_graph` is fit once on the real data
(slope, intercept, residual variance — the identical conditionals the
physics-MC generator samples). Those constants become a differentiable
penalty (see `_physics_loss`) added to the loss, weighted by
`physics_weight`, that pushes every reconstructed batch onto the
mechanistic relationships while matching their real residual spread — so
the constraint guides the generative manifold toward physically
consistent samples without re-introducing the VAE's variance-shrinkage
tendency.

Everything else matches the plain VAE. Trains on `x` (expected already
reasonably scaled) and samples `n_samples` rows from the prior; pass a
fitted `scaler` only if `x` needs inverse-transforming back to source
units. Loss is the standard VAE ELBO (reconstruction MSE + analytic KL to
the standard-normal prior, `beta` linearly annealed from 0 over the first
`kl_warmup_frac` of training to avoid posterior collapse), plus the
covariance-matching term (`cov_weight`) that rewards preserving the full
correlation matrix, plus the physics term (`physics_weight`) that rewards
honoring the causal graph specifically. `cov_weight` shapes the whole
covariance structure from data; `physics_weight` anchors the particular
mechanistic edges an expert asserts — set `physics_weight=0.0` to recover
the plain VAE exactly.

A fourth term, weighted by `marginal_weight`, is the per-feature 1D
Wasserstein distance (see `_marginal_loss`) between samples drawn from the
prior and the real batch. The physics term only constrains parent->child
edges, leaving the causal graph's *root* variables free to be compressed
by the VAE — the main reason marginal fit otherwise trails physics-MC,
which samples roots straight from their real marginal. This term shapes
every generated feature's whole distribution onto real, roots included;
set `marginal_weight=0.0` to disable it.

`prior_type` chooses how latents are drawn at generation (and for the
generated-sample loss terms): "standard" samples the N(0, I) prior;
"aggregate" samples the aggregate posterior over the training rows (see
`_decode_samples`), which keeps samples on the region of latent space the
decoder actually learned and avoids the prior-hole mismatch that inflates
both KS and correlation error when the aggregate posterior doesn't fill
the prior.

`constrain_generated` routes the covariance and physics terms onto a batch
of freshly generated rows instead of the reconstructions — so those
constraints shape the population that's actually sampled at generation,
not just the model's reconstruction of real inputs. The default (False)
keeps them on reconstructions, matching the plain VAE's covariance-term
convention.

`calibrate_marginals` applies an empirical-copula post-step (see
`_calibrate_marginals`): after generation, each feature is mapped onto the
real training marginal at matching quantiles, forcing every marginal to
match real almost exactly while preserving the learned rank-correlation
structure. It injects the real empirical margins (like physics-MC does for
its roots), so a run using it should be described as marginal-calibrated
rather than purely learned. Default False.

`use_minibatch`, `batch_size`, `patience`, `hidden_dim`, `dropout`, and
`free_bits` behave exactly as in the plain VAE generator — see its
docstring for the capacity/collapse/early-stopping trade-offs; the causal
constraint doesn't change any of them.

## Results

| Engine | Correlation Distance (Euclidean) | Mean KS Statistic |
|---|---|---|
| Physics-Informed Monte Carlo | 0.9434 | 0.0467 |
| Regression | 0.7410 | 0.0335 |
| Variational Autoencoder | 0.8392 | 0.0774 |
| Physics-Informed VAE (Hybrid) | **0.5330** | **0.0237** |

(Lower is better for both metrics; bold = best.)

## Findings

- Best **Correlation Distance (Euclidean)**: Physics-Informed VAE (Hybrid) (0.5330)
- Best **Mean KS Statistic**: Physics-Informed VAE (Hybrid) (0.0237)

Per-feature marginal fit (two-sample KS test, real vs. generated; lower ks_stat / higher p_value = closer):

- **Physics-Informed Monte Carlo**: 2/7 features statistically distinguishable from real (p < 0.05)
- **Regression**: 0/7 features statistically distinguishable from real (p < 0.05)
- **Variational Autoencoder**: 7/7 features statistically distinguishable from real (p < 0.05)
- **Physics-Informed VAE (Hybrid)**: 0/7 features statistically distinguishable from real (p < 0.05)

## Per-Feature KS Statistic

Lower = closer to real; bold = best per feature.

| Feature | Physics-Informed Monte Carlo | Regression | Variational Autoencoder | Physics-Informed VAE (Hybrid) |
|---|---|---|---|---|
| Size | 0.0458 | 0.0357 | 0.0808 | **0.0223** |
| Weight | 0.0715 | 0.0350 | 0.0862 | **0.0330** |
| Sweetness | 0.0610 | **0.0270** | 0.0628 | 0.0278 |
| Crunchiness | 0.0562 | 0.0312 | 0.1245 | **0.0245** |
| Juiciness | 0.0298 | 0.0345 | 0.0688 | **0.0217** |
| Ripeness | 0.0240 | 0.0287 | 0.0602 | **0.0222** |
| Acidity | 0.0387 | 0.0425 | 0.0583 | **0.0147** |

## Feature Spread Comparison

**Correlation Distance (Euclidean)** is scale-invariant — Pearson correlation ignores absolute spread, so a method can score well on it while its population is uniformly shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is narrower than real).

| Feature | Real Std | Physics-Informed Monte Carlo | Regression | Variational Autoencoder | Physics-Informed VAE (Hybrid) |
|---|---|---|---|---|---|
| Size | 1.927 | 2.007 (1.04x) | 1.906 (0.99x) | 1.543 (0.80x) | 1.926 (1.00x) |
| Weight | 1.600 | 1.618 (1.01x) | 1.539 (0.96x) | 1.188 (0.74x) | 1.600 (1.00x) |
| Sweetness | 1.954 | 1.944 (0.99x) | 1.894 (0.97x) | 1.691 (0.87x) | 1.935 (0.99x) |
| Crunchiness | 1.411 | 1.393 (0.99x) | 1.339 (0.95x) | 0.885 (0.63x) | 1.396 (0.99x) |
| Juiciness | 1.898 | 1.922 (1.01x) | 1.880 (0.99x) | 1.633 (0.86x) | 1.941 (1.02x) |
| Ripeness | 1.883 | 1.859 (0.99x) | 1.888 (1.00x) | 1.587 (0.84x) | 1.868 (0.99x) |
| Acidity | 2.117 | 2.157 (1.02x) | 2.004 (0.95x) | 1.772 (0.84x) | 2.104 (0.99x) |

## Generalization Check

Correlation distance for each engine's synthetic data against the train split it was fit on vs. the held-out test split it's actually scored on. A small gap is evidence results generalize rather than matching training-set noise a real (unseen) population wouldn't share.

| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |
|---|---|---|---|
| Physics-Informed Monte Carlo | 0.9142 | 0.9434 | 0.0292 |
| Regression | 0.7275 | 0.7410 | 0.0135 |
| Variational Autoencoder | 0.7821 | 0.8392 | 0.0571 |
| Physics-Informed VAE (Hybrid) | 0.3887 | 0.5330 | 0.1443 |

## Downstream Utility (TSTR)

Train-on-Synthetic, Test-on-Real for the `Quality` label: a RandomForest is trained on each engine's synthetic population (labels produced by generating each class separately) and scored on the real held-out test set. **Real (TRTR)** — a classifier trained on real data — is the ceiling; the closer an engine gets to it, the more genuinely useful its synthetic population is. This rewards preserving the feature-label joint structure, not just the marginals.

| Trained on | Accuracy | ROC-AUC |
|---|---|---|
| **Real (TRTR ceiling)** | 0.8858 | 0.9600 |
| Physics-Informed Monte Carlo | 0.7600 | 0.8445 |
| Regression | 0.8050 | 0.9032 |
| Variational Autoencoder | 0.7967 | 0.8883 |
| Physics-Informed VAE (Hybrid) | 0.8625 | 0.9425 |

(Higher is better; closer to the Real ceiling = more useful synthetic data.)

## Uncertainty Calibration (Coverage)

Tests the *calibrated uncertainty* claim directly. For each feature, the central 90% interval of each engine's generated population is formed, and we measure the fraction of real held-out values that fall inside it (averaged over features). Well-calibrated ⇒ coverage ≈ the nominal 0.90; **below** = over-confident (intervals too narrow), **above** = intervals too wide. `calibration_error` is the mean absolute gap between empirical and nominal coverage across interval levels from 0.10 to 0.95 (lower = better calibrated across the whole range).

| Engine | Coverage @ 90% (nominal 0.90) | Calibration Error |
|---|---|---|
| Physics-Informed Monte Carlo | 0.904 | 0.0132 |
| Regression | 0.887 | 0.0199 |
| Variational Autoencoder | 0.815 | 0.0835 |
| Physics-Informed VAE (Hybrid) | 0.901 | **0.0098** |

(Coverage closest to nominal and lowest calibration error = best-calibrated uncertainty.)

![Uncertainty calibration reliability curve](../../results/apple_quality/figures/marginals/coverage_calibration.png)

## Figures

### Joint variability (correlation)

**Correlation matrices**

![Real correlation matrix](../../results/apple_quality/figures/correlation/correlation_real.png)
![Physics-Informed Monte Carlo correlation matrix](../../results/apple_quality/figures/correlation/correlation_physics_mc.png)
![Regression correlation matrix](../../results/apple_quality/figures/correlation/correlation_regression.png)
![Variational Autoencoder correlation matrix](../../results/apple_quality/figures/correlation/correlation_vae.png)
![Physics-Informed VAE (Hybrid) correlation matrix](../../results/apple_quality/figures/correlation/correlation_hybrid_vae.png)

**PCA projection — real vs. each method, plus all combined**

![PCA projection of real vs generated populations](../../results/apple_quality/figures/correlation/pca_real_vs_generated.png)

**PCA projection — each population shown individually, no overlay**

![PCA projection of each population individually](../../results/apple_quality/figures/correlation/pca_individual.png)

### Marginal distributions (KS)

**KS statistic by feature and method**

![KS statistic heatmap](../../results/apple_quality/figures/marginals/ks_heatmap.png)

**Empirical CDFs — real vs. generated, per feature**

![ECDF overlay per feature](../../results/apple_quality/figures/marginals/ecdf_overlay.png)

**P-value by feature and method**

![P-value heatmap](../../results/apple_quality/figures/marginals/pvalue_heatmap.png)

**Volcano plot — effect size vs. significance**

![Volcano plot](../../results/apple_quality/figures/marginals/volcano_plot.png)

---

Raw data: `results/apple_quality/summary_metrics.csv`, `results/apple_quality/ks_marginals.csv`, `results/apple_quality/generalization_gap.csv`, `results/apple_quality/synthetic_data/`.
