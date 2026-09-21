# Model architecture & hyperparameters (supplement-ready)

Answers Reviewer "architecture and tuning parameters not sufficiently reported"
and the minor "number of ensemble models". Extracted verbatim from the code
(`src/generators/`, `src/configs/base.py`, `src/multiseed.py`). Private draft —
fold into the supplement. Will be extended with the citrus mechanistic-CVAE once
the flagship models are finalised.

## PI-VAE (physics-informed VAE) — `hybrid_vae_generator.py`

**Network** (fully connected, one hidden layer):
- Encoder: `Linear(d → 128) → ReLU → Dropout(0.0) → Linear(128 → 2·L)` (outputs μ and logσ²)
- Decoder: `Linear(L → 128) → ReLU → Dropout(0.0) → Linear(128 → d)`
- `d` = number of features (dataset-specific); `L` = latent dim (default 4)
- logvar clamped to [−8, 8]; reparameterised sampling
- Prior: standard N(0, I); at generation, samples drawn from the prior (novel
  individuals) or the aggregate posterior (see `_decode_samples`)

**Loss** `L = L_rec + β·L_KL + λ_cov·L_cov + λ_phys·L_phys + λ_marg·L_marg`, then
an empirical-copula marginal calibration post-step:
- `L_rec` reconstruction (MSE)
- `L_KL` KL to N(0, I) with a free-bits floor; β linearly annealed from 0 over the
  first 30% of epochs (`kl_warmup_frac = 0.3`)
- `L_cov` covariance-matching (generated vs real covariance)
- `L_phys` structural/edge term: per parent→child linear-Gaussian conditional
  (child = slope·parent + intercept + noise); on citrus replaced by explicit
  mechanistic equations (see mechanistic-CVAE below)
- `L_marg` per-feature 1-D Wasserstein between prior-drawn samples and real
- **Calibration** (`VAE_CALIBRATE_MARGINALS`): empirical-copula remap — each
  generated feature is mapped to the real training marginal at matching quantiles,
  preserving the learned rank (Spearman) dependence while making marginals match

**Optimiser / training**: Adam, lr = 1e-3; full-batch (minibatch optional,
batch 128); up to 5000 epochs; early-stop patience 400; train/test split 0.3.

## Default hyperparameters (`src/configs/base.py`)

| Parameter | Default | Note |
|---|---|---|
| LATENT_DIM | 4 | latent dimension L |
| VAE_HIDDEN_DIM | 128 | hidden width |
| VAE_DROPOUT | 0.0 | |
| VAE_BETA | 1.0 | KL weight (annealed) |
| VAE_FREE_BITS | 0.0 | **per-dataset override to 1.0** (e.g. date, safou, grape, tomato, apple) |
| VAE_COV_WEIGHT | 1.0 | **override 0.0 for safou** (near-deterministic 4-feature physics) |
| VAE_PHYSICS_WEIGHT | 1.0 | **override 3.0 for safou** |
| VAE_MARGINAL_WEIGHT | 1.0 | |
| VAE_PRIOR_TYPE | standard | N(0, I) |
| VAE_CALIBRATE_MARGINALS | True | empirical-copula post-step |
| VAE_CONSTRAIN_GENERATED | True | |
| VAE_EPOCHS | 5000 | max; early-stopped |
| VAE_PATIENCE | 400 | early-stop patience |
| VAE_USE_MINIBATCH | False | full-batch by default |
| VAE_BATCH_SIZE | 128 | if minibatch on |
| TEST_SIZE | 0.3 | train/test split |

Per-dataset overrides live in each `src/configs/<dataset>.py` (FEATURES,
CAUSAL_GRAPH/ROOT_VARIABLES, and any of the above). State the effective values per
dataset in the supplement.

## Ensemble / epistemic uncertainty

- Multi-seed evaluation: **seeds 41–45 (5 seeds)** (`src/multiseed.py`); every
  headline metric reported as mean ± s.d. across these 5.
- Epistemic-uncertainty / off-support detection uses the **5-model seed ensemble**
  (near/far disagreement).
- Downstream significance tests extend to seeds 41–50 where noted.

## Conditional VAE (transfer) — `conditional_vae_generator.py`

Same encoder/decoder, but both take a condition vector (group one-hot, or
continuous covariates) concatenated to their input; generation draws z ~ N(0, I)
and appends the target condition, so a population can be generated for a specified
(and even unseen) condition. Optional physics term as above.

## Mechanistic CVAE (citrus PI-VFP) — `mechanistic_cvae.py`

Conditional VAE whose constraint term is built from **explicit equations**, not
data-fitted edges:
- Mass-balance conservation (inequality pairs x[heavy] ≥ x[light]; non-negativity),
  penalised on the decoder output in raw units and projected at generation.
- Kinetic-mean anchor: the conditional mean is pushed to a rate-law prediction
  (e.g. chilling-degree `%CI = k·max(Tcrit − T, 0)·t`) fit on training conditions.
- Otherwise standard ELBO + covariance matching.
Hyperparameters as above (latent 4, hidden 128); constraint weights λ_cons, λ_kin.
