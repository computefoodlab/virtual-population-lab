# Results summary — generative virtual food populations

Feasibility results for the code in this repository: generating **virtual food
populations** (synthetic populations of individual food items that preserve a
real population's marginals, joint dependence, uncertainty calibration, and
downstream usefulness) across five public datasets with four generators.

All numbers below are **mean ± standard deviation over 5 random seeds**
{41, 42, 43, 44, 45} for the comparison metrics and **3 seeds** {41, 42, 43} for
the transfer experiment. Every engine is seeded from the same RNG state before
generation, so the comparison is order-invariant. Lower is better for
correlation distance, mean KS, and calibration error; coverage@90 targets the
nominal 0.90; TSTR accuracy is higher-is-better.

Reproduce with:

```
python -m src.multiseed --seeds 41 42 43 44 45 --loro
```

## Engines

| Paper name | Repo key | Description |
|---|---|---|
| PI-MC | `physics_mc` | Physics-informed Monte Carlo: ancestral sampling over a supplied causal graph, roots from their real marginals, edges from fitted linear-Gaussian conditionals. |
| RFR | `regression` | Per-feature random forest with out-of-bag predictions plus bootstrapped OOB residuals. |
| VAE | `vae` | Variational autoencoder with a covariance-matching term. |
| PI-VAE | `hybrid_vae` | VAE plus causal-graph physics consistency, covariance matching, marginal matching, and an empirical-copula calibration post-step. |

Loss terms, per-dataset weights, and the full algorithm are generated into
`results/_summary/` by `python -m src.loss_table` and
`python -m src.paper_algorithms`.

## Datasets

| Dataset | Type (n) | Causal graph | Transfer groups | Downstream label |
|---|---|---|---|---|
| `apple_quality` | quality/sensor (4,000) | 5 edges | — | Quality (2-class) |
| `banana_quality` | quality/sensor (1,000) | empty | region (8) | quality_category (4-class) |
| `biofood_date_region` | nutrients: minerals (77) | 4 edges (dilution) | UAE / Tunisia / Oman | — |
| `biofood_safou_region` | nutrients: lipids, individual-level (41) | 3 edges (near-deterministic) | Congo / Guinea | — |
| `mango_composition` | nutrients: Vitamin C, individual-level (186) | empty | 4 cultivars (all ≥18) | — |

Apple, banana, safou, and mango are sampled individuals; date is a compiled
composition table and is used as a corroborating case.

## Distributional fidelity and calibration

Per-dataset best value per column (`results/_summary/multiseed_metrics.csv`):

| Dataset | Corr. dist. | Mean KS | Calib. err. | Coverage@90 | TSTR acc. |
|---|---|---|---|---|---|
| Apple | **PI-VAE 0.524** | **PI-VAE 0.029** | **PI-VAE 0.012** | PI-MC 0.901 ≈ PI-VAE 0.898 | **PI-VAE 0.866** |
| Banana | PI-VAE 0.298 ≈ PI-MC 0.297 | **PI-VAE 0.059** | **PI-VAE 0.024** | **PI-VAE 0.895** | **PI-VAE 0.911** |
| Date | **PI-VAE 0.560** | **PI-VAE 0.187** | **PI-VAE 0.078** | PI-MC 0.915 (PI-VAE 0.858) | — |
| Safou | VAE 0.158 (PI-VAE 0.231) | VAE 0.230 (PI-VAE 0.273) | VAE 0.097 (PI-VAE 0.122) | **PI-VAE 0.915** | — |
| Mango | **PI-VAE 0.347** | PI-MC 0.130 ≈ PI-VAE 0.135 | PI-MC 0.042 ≈ PI-VAE 0.046 | PI-MC 0.904 ≈ PI-VAE 0.889 | — |

- **Apple** (largest, most causally structured) — PI-VAE wins every fidelity
  axis, with the correlation-distance margin (0.524 vs 0.703 for the next best)
  well outside seed variation.
- **Banana and mango** — features are near-independent, so correlation distance
  sits at the sampling-noise floor for every engine. On banana, the *real*
  train-vs-test correlation matrices already differ by 0.28 ± 0.06 from
  finite-sample noise alone; every engine scores 0.29–0.36. PI-MC matches or
  edges the marginal metric there because sampling each independent feature
  from its own marginal reproduces marginals by construction. Confirmed against
  a synthetic oracle (`src.experiments.oracle_independent`) and a latent-dim
  sweep (`src.experiments.banana_latent_sweep`).
- **Date** — PI-VAE wins fidelity; coverage falls slightly below nominal.
- **Safou** (n = 41, near-deterministic linear structure) — all four engines sit
  within seed noise on fidelity (KS 0.230–0.273, ±0.065–0.071). The plain VAE
  has the best point estimates on correlation, KS, and calibration error but
  under-covers (0.827); PI-VAE is the only engine with calibrated coverage
  (0.915).

## Downstream utility (TSTR)

Classifier trained only on synthetic data, scored on real held-out data
(`results/<dataset>/tstr.csv`):

| Dataset | Real ceiling (TRTR) | PI-VAE | Next best |
|---|---|---|---|
| Apple (accuracy) | 0.888 ± 0.014 | **0.866 ± 0.008** | RFR 0.809 |
| Banana (accuracy) | 0.926 ± 0.018 | **0.911 ± 0.015** | PI-MC 0.847 |

## Loss-term ablation

Full table: `python -m src.ablation` → `results/_summary/ablation.{csv,md,tex}`.

- **Covariance matching is the main driver of joint structure where structure
  exists.** Removing it worsens correlation distance on apple (+0.87) and
  banana (+0.36). On safou it is set to zero in the shipped config: a 4×4
  covariance estimated from ~28 training rows is too noisy to be a useful
  target once the near-deterministic physics edges already pin the structure.
- **Physics weight on safou.** Across 5 seeds every positive
  `physics_weight ∈ {0.5, 1, 2, 3}` gives the same correlation distance
  (~0.22–0.25, ±0.20); only `physics_weight = 0` improves it (0.11), which is no
  longer a physics-informed model. Physics weight has no effect on
  KS/calibration/coverage, which are set by the copula calibration step. On the
  larger and more structured datasets (apple, date) the same priors help.
- **Per-group physics is real but not exploitable at these sample sizes.**
  Per-region fits on date show Water→mineral slopes differing 3–5× across
  UAE/Tunisia/Oman, but region-conditional physics (per-region edges plus a
  region-conditioned VAE) underperforms pooling (correlation distance
  0.53 → 0.76), and single-region subsetting is worse still (0.53 → 0.71) — a
  bias–variance outcome at n ≈ 7–64. Reproduce with
  `src.experiments.region_conditional` and `src.experiments.structure_floor`.

## Support-aware uncertainty (near/far transfer)

Same-model protocol: train one conditional-VAE ensemble on one group; NEAR =
held-out rows of that same group, FAR = a different group, from the same
models. Ensemble disagreement is read as epistemic uncertainty
(`results/_summary/multiseed_nearfar.csv`):

| Dataset | Disagreement FAR / NEAR | Widening | Coverage FAR / NEAR | Conformal FAR / NEAR |
|---|---|---|---|---|
| Mango (4 cultivars, individual-level) | 0.104 / 0.070 | +51 ± 22 % | 0.693 / 0.832 | 0.748 / 0.890 |
| Banana (region) | 0.118 / 0.068 | +73 ± 10 % | 0.988 / 0.999 | 0.832 / 0.917 |
| Date (region) | 0.110 / 0.055 | +105 ± 25 % | 0.784 / 0.826 | 0.853 / 0.913 |
| Safou (region) | 0.074 / 0.051 | +50 ± 29 % | 0.316 / 0.641 | 0.538 / 0.789 |

**Detection is reliable; calibration is only partial.** Disagreement widens for
an unseen group on every dataset and seed, so the model detects that it is
extrapolating. Raw coverage off-support is not calibrated: banana stays
over-conservative, while mango, date, and safou degrade (mango 0.83 → 0.69,
safou 0.64 → 0.32).

Split-conformal recalibration — fitting each feature's interval half-width on
the NEAR held-out reals and transferring that width off-support — corrects
in-region coverage toward 0.90 from both directions (banana 1.00 → 0.92,
date 0.83 → 0.91, mango 0.83 → 0.89) and closes roughly half the FAR gap
(mango 0.69 → 0.75, date 0.78 → 0.85, safou 0.32 → 0.54) without reaching
nominal. The residual is genuine distribution shift, plus insufficient
generated support at very small n. Implemented in `src/evaluation/loro.py`
(columns `coverage_*_conf`); standalone prototype in
`src.experiments.conformal_prototype`.

## Figures

`results/_summary/` — `fig1_leaderboard.png` (engines × datasets, correlation
distance and mean KS), `fig2_downstream.png` (TSTR vs the real ceiling),
`fig3_nearfar.png` (uncertainty widening). Regenerate with
`python -m src.summary_figures` after `src.multiseed`.

## Scope

This is a feasibility study, not a performance benchmark. The datasets are
modest and public; the nutrition demonstration is one crop and one nutrient
(mango Vitamin C); PI-VAE is the most consistent engine across the five
datasets rather than a universal winner — its fidelity advantage is largest
where real mechanistic structure exists, and where features are effectively
independent or n is very small, the simpler engines match it and its remaining
advantage is calibrated uncertainty. Far-support coverage is only partially
recalibratable.

Per-dataset reproduction: `make run CONFIG=<name>` and `make loro CONFIG=<name>`
for `apple_quality`, `banana_quality`, `biofood_date_region`,
`biofood_safou_region`, `mango_composition`; aggregate with
`python -m src.multiseed --loro`, `python -m src.ablation`, and
`python -m src.loss_table`.
