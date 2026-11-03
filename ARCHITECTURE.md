# Model architecture and training specification

This document reports the model architecture, training procedure, and
hyperparameters so the results are reproducible. Every value here is set in
[`src/configs/base.py`](src/configs/base.py); each dataset config imports it and
overrides only what it must, so the full specification is version-controlled rather
than described only in prose.

## Generative backbone (VAE / PI-VAE / conditional PI-VAE)

All variants share one small feed-forward VAE
([`src/generators/hybrid_vae_generator.py`](src/generators/hybrid_vae_generator.py),
class `VAE`):

- **Encoder:** `Linear(D → 128) → ReLU → Dropout(0.0) → Linear(128 → 2·4)` — outputs
  the mean and log-variance of the latent.
- **Latent dimension:** 4.
- **Decoder:** `Linear(4 → 128) → ReLU → Dropout(0.0) → Linear(128 → D)`.
- **Hidden width** 128, **activation** ReLU, **dropout** 0.0.
- **Log-variance** clamped to `[-8, 8]`; reparameterised sampling.
- **D** (input dimension) = number of features: citrus 5, tomato 6, grape 7,
  apple 8, mango 3, safou 4.

**Loss:** `L = L_rec + β·L_KL + λ_cov·L_cov + λ_marg·L_marg (+ λ_cons·L_cons)`,
followed by an empirical-marginal calibration post-step.
- `L_rec` reconstruction (MSE); `L_KL` KL to N(0, I) with a free-bits floor, β
  annealed over the first 30% of epochs.
- `L_cov` covariance matching (generated vs real), which is how feature dependence
  is learned.
- `L_marg` per-feature marginal matching.
- `L_cons` conservation penalty, applied **only where a governing law exists**
  (citrus, tomato, safou), with a hard feasibility projection at generation.
- **The data-fitted parent→child edge/consistency term is disabled**
  (`VAE_PHYSICS_WEIGHT = 0`); feature dependence is learned by `L_cov`, not imposed
  as a fitted edge. This is deliberate (see the manuscript's terminology section):
  "physics" is reserved for the conservation laws and the published citrus kinetics.
- **Calibration** (`VAE_CALIBRATE_MARGINALS`): empirical inverse-CDF remap of each
  generated feature onto the real training support, preserving the learned rank
  dependence.

**Conditional variant** (digital twin and transfer;
[`mechanistic_cvae.py`](src/generators/mechanistic_cvae.py)): the descriptor vector
is concatenated to both the encoder input and the decoder (latent) input, so a
single trained model generates the population for a supplied descriptor without being
rebuilt. Its mechanistic terms (conservation penalty, kinetic conditioning) are
switched on where a governing law and published kinetics exist (the citrus digital
twin) and off otherwise, in which case it reduces to a plain conditional VAE
(verified to reproduce the earlier standalone conditional VAE identically).

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 0.001 |
| Epochs | up to 5000, early stopping (patience 400) |
| Batching | full-batch (minibatch optional, batch 128) |
| ELBO β (KL to N(0, I)) | 1.0, annealed over first 30% of epochs |
| Free bits | 0.0 |
| Covariance-matching weight | 1.0 |
| Marginal-matching weight | 1.0 |
| Edge / consistency-term weight | 0.0 (disabled) |
| Conservation constraint | soft penalty + hard projection, where a law exists |
| Marginal calibration | empirical inverse-CDF onto real support (on) |
| Latent draw at generation | aggregated posterior (standard prior) |
| Generated population size | 1000 (unless an experiment sets otherwise) |
| Train/test split | 0.3 |

## Imposed physics (only where a genuine law holds)

Declared per dataset as `CONSTRAINTS` in the config:

- **citrus:** `RindDry ≤ RindFresh` (rind moisture ≥ 0)
- **tomato:** `Glucose + Fructose ≤ 10·SSC` (sugars within soluble solids)
- **safou:** water + fat, and the fatty-acid sum
- **grape, apple, mango:** none (no governing law; nothing imposed)

The citrus digital twin additionally conditions on the commodity's published
chilling-injury, colour and moisture-loss kinetics.

## Seeds / ensemble

| Experiment | Seeds |
|---|---|
| Multiseed fidelity (Table S6) | 5 (41–45), mean ± s.d. |
| TSTR (Table S11) | 5 (41–45), random forest 200 trees |
| Digital twin (Tables S9/S10) | 3 (41–43), leave-one-condition-out |
| Transfer decision | 15 |

Epistemic-uncertainty / off-support detection uses the 5-model seed ensemble
(near/far disagreement).

## Reproducing

```
make setup                      # venv + requirements
python -m src.multiseed --seeds 41 42 43 44 45 --loro   # Table 1 / S6
python -m src.experiments.deep_baselines                # Table S7
python -m src.experiments.ablation                      # Table S8
python -m src.experiments.citrus_digital_twin           # Tables S9 / S10
python -m src.experiments.transfer_decision             # transfer decision
python -m src.experiments.drift_detection               # Table S19
```

The citrus flagship expects the unpublished, available-on-request raw file at
`data/cleaned-citrus-exp6.csv`; the other five datasets are public and cleaned copies
are in `data/`.
