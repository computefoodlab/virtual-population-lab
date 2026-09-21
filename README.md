# Virtual Population Lab

Code for the feasibility study behind *Hybrid generative virtual fresh food
populations for a data-scarce world*. Given a real dataset of individual food
items, the pipeline generates one synthetic population per modelling engine and
evaluates which engine best preserves the real population's marginal
distributions, joint dependence, uncertainty calibration, and downstream
usefulness.

Everything dataset-specific (feature columns, assumed structural graph, imposed
conservation laws, loss weights, whether the source data is pre-scaled) lives in
a single config module, so pointing the pipeline at a new dataset means writing a
new config rather than touching modelling code.

Per-dataset auto-generated reports: `reports/<config_name>/REPORT.md`.
Cross-dataset tables used in the paper: [results/_summary/](results/_summary/).

## The five engines

Engine keys used in code, filenames, and result CSVs map to the paper's
abbreviations as follows:

| Repo key | Paper | Module |
|---|---|---|
| `physics_mc` | PI-MC | [src/generators/physics_mc_generator.py](src/generators/physics_mc_generator.py) |
| `mcmc` | MCMC (prior art) | [src/generators/mcmc_generator.py](src/generators/mcmc_generator.py) |
| `regression` | RFR | [src/generators/regression_generator.py](src/generators/regression_generator.py) |
| `vae` | VAE | [src/generators/vae_generator.py](src/generators/vae_generator.py) |
| `hybrid_vae` | PI-VAE | [src/generators/hybrid_vae_generator.py](src/generators/hybrid_vae_generator.py) |

1. **Structural / physics-informed Monte Carlo (PI-MC)** — ancestral sampling
   over a supplied structural graph. Root variables are drawn from their real
   marginals; each (parent → child) edge is fit as a linear-Gaussian conditional
   on the training data. Forward sampling is exact, with no Markov chains.

2. **Gaussian-copula Monte Carlo (MCMC)** — the domain prior art for virtual
   food populations: correlated, non-Gaussian parameters are drawn by
   Monte-Carlo sampling from a fitted Gaussian copula (empirical marginals mapped
   through a rank-correlation matrix), reproducing marginals and linear rank
   dependence without a neural network.

3. **Random-forest regression (RFR)** — one `RandomForestRegressor` per feature,
   predicting it from the others on the training split. Synthetic rows are seeded
   from out-of-bag predictions (leakage-free without a separate held-out set)
   plus noise bootstrapped from the OOB residuals.

4. **VAE** — learns a latent joint distribution over all features and samples new
   rows from its prior. Trained with the standard ELBO (reconstruction plus KL to
   the standard normal, `beta` annealed over the first `kl_warmup_frac` of
   training) and a covariance-matching term (`cov_weight`).

5. **Physics-informed VAE (PI-VAE)** — the VAE above plus a structural/physics
   consistency term on the graph edges, a marginal-matching term, and an
   empirical-copula calibration post-step; where a governing physical law holds,
   a conservation/compositional constraint is imposed a priori (soft penalty plus
   a hard feasibility projection at generation). A conditional variant
   ([src/generators/mechanistic_cvae.py](src/generators/mechanistic_cvae.py))
   conditions generation on a covariate (e.g. storage temperature and duration)
   for the digital-twin decision. `physics_weight = 0` recovers a calibrated VAE;
   an empty graph makes the physics term zero.

Each generator module exposes a single `generate(...)` function with a
consistent calling convention.

## Datasets

Five of the six datasets are public; cleaned copies used by the pipeline are in
[data/](data/). The **citrus flagship** is the authors' own **unpublished**
individual-fruit cold-storage data — **not distributed in this repository;
available on request**. Its full descriptive statistics are given in the paper's
Supplementary Table S2, and running `citrus_exp6` expects the raw file at
`data/cleaned-citrus-exp6.csv`.

| Config | Source | n | Sampling unit | Features (transfer / TSTR group) |
|---|---|---|---|---|
| `citrus_exp6` (flagship) | Authors' unpublished cold-storage experiment (available on request) | 239 | individual fruit | rind fresh/dry mass, moisture loss, chilling injury, colour (Cultivar; storage condition for the digital-twin decision) |
| `tomato_nir` | [Ibáñez et al. 2019, Zenodo 10633732](https://zenodo.org/records/10633732) | 650 | individual fruit | SSC, malic/citric/glutamic acid, fructose, glucose (Varietal type) |
| `grape_berry` | [LowSugarBerry, FigShare 10.6084/m9.figshare.28308986](https://doi.org/10.6084/m9.figshare.28308986) | 1,536 | individual berry | glucose, fructose, malic/tartaric acid, K, Mg, Ca (Genotype) |
| `apple_samnegard` | [Samnegård et al., Zenodo 4989885 / Dryad 10.5061/dryad.nk9871p](https://doi.org/10.5061/dryad.nk9871p) | 255 | individual fruit | Brix, TA, firmness, dry matter, P/K/Ca/Mg (Storage category) |
| `mango_composition` | [Munawar 2019, Mendeley b9d6s7hr33](https://doi.org/10.17632/b9d6s7hr33.1) | 186 | individual fruit | Vitamin C, SSC, TA (Cultivar) |
| `biofood_safou_region` | [FAO/INFOODS BioFoodComp4.0 v4.0](https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/) | 41 | individual tree | Water, Fat, palmitic, stearic (Region) |

Datasets carry raw units; the pipeline fits its own `StandardScaler` on the
training split (see `IS_PRE_SCALED` in each config).

## Layout

```
src/
  configs/
    base.py                  # every config field, documented once, with defaults
    <dataset>.py             # imports base.py, overrides only dataset-specific fields
  generators/                # physics_mc, mcmc, regression, vae, hybrid_vae, mechanistic_cvae
  evaluation/
    evaluate.py              # correlation, PCA, KS, ECDF, energy distance, MMD, summary metrics
    coverage.py              # interval coverage + calibration error
    tstr.py                  # train-on-synthetic, test-on-real
    loro.py                  # same-model near/far transfer + conformal recalibration
  config_loader.py           # resolves a config module by name
  data_loading.py            # loads and cleans the source CSV for a config
  main.py                    # runs every engine, evaluates, writes the report
  run_loro.py                # standalone near/far transfer run
  check_generalization.py    # re-checks generated data against train vs test
  multiseed.py               # repeats fidelity across seeds, mean +/- s.d. (paper Table 1 / S6)
  paper_algorithms.py        # per-engine equations and the PI-VAE algorithm
  experiments/               # scripts that produce the supplementary tables (see its README)
results/<config_name>/
  synthetic_data/            # generated rows per engine
  figures/                   # correlation and marginal figures
  summary_metrics.csv        # correlation distance and mean KS per engine
  ks_marginals.csv           # per-feature KS statistic and p-value per engine
  coverage.csv               # coverage@90 and calibration error per engine
  tstr.csv                   # downstream accuracy per engine plus the real ceiling
  loro.csv                   # near/far transfer, per group plus a MEAN row
  generalization_gap.csv     # correlation distance vs train split and vs test split
results/_summary/            # cross-dataset tables used in the paper (multiseed, baselines,
                             #   ablation, conformal, transfer, drift)
reports/<config_name>/
  REPORT.md                  # regenerated by main.py on every run; do not hand-edit
```

Outputs are namespaced by config name, so results from different datasets never
overwrite each other.

## Running it

```
make setup                       # create venv, install requirements
make run CONFIG=<config_name>    # generation + evaluation + report for one dataset
make loro CONFIG=<config_name>   # near/far transfer (requires LORO_GROUP in the config)
make check-overfit CONFIG=<name> # re-check the last run's synthetic data, no retraining
make run-all                     # make run for every dataset
make loro-all                    # make loro for every dataset
make all                         # run-all, then loro-all
```

Multi-seed aggregation (the paper's Table 1 / S6) and the supplementary-table
scripts are separate entry points:

```
python -m src.multiseed --seeds 41 42 43 44 45 --loro   # fidelity + near/far, mean +/- s.d.
python -m src.experiments.deep_baselines                # CTGAN / TVAE / MCMC vs PI-VAE (Table S7)
python -m src.experiments.ablation                      # constraints vs calibration (Table S8)
python -m src.experiments.transfer_decision             # data-scarce cultivar transfer (Table S10)
python -m src.experiments.conformal_calibration         # in-distribution conformal (Table S13)
python -m src.experiments.drift_detection               # drift monitor (Table S19)
python -m src.paper_algorithms                          # equations + algorithm
```

`make run` logs the active config and each pipeline stage, prints a bordered
summary table (best value per metric marked), and rewrites
`reports/<config_name>/REPORT.md` from that run's numbers and figures. All runs
are CPU-feasible.

### Adding a new dataset

Write a new module in `src/configs/` (copy an existing one as a template):
`from src.configs.base import *`, then override at minimum `DATA_PATH`,
`FEATURES`, `ID_COLUMNS`, `LABEL_COLUMN`, `IS_PRE_SCALED`, `ROOT_VARIABLES`,
`CAUSAL_GRAPH` (the structural graph), and `DATA_SOURCE_URL`. Set `LORO_GROUP`
to enable the near/far transfer experiment and `RUN_TSTR` for the downstream
check; add `CONSTRAINTS` to impose a conservation law where one holds. Every
field's meaning is documented once in
[src/configs/base.py](src/configs/base.py). Run it with
`make run CONFIG=<your_module_name>`.

### Generalization check

Each engine's correlation distance is scored against the held-out test split.
The generalization check compares that same metric against the train split the
engine was fit on: a small gap is evidence of genuine generalization, while
train being much better than test would indicate fitting training-split noise.
Results land in `results/<config_name>/generalization_gap.csv` and the report's
"Generalization Check" section. `make check-overfit` re-runs only this check.

## Evaluation

| Axis | Metric | Where |
|---|---|---|
| Joint dependence | Euclidean distance between real and generated correlation matrices; energy distance; MMD | `summary_metrics.csv`, `results/_summary/` |
| Marginals | Two-sample KS statistic and p-value per feature | `ks_marginals.csv` |
| Uncertainty | Coverage of the central 90% interval on real held-out values; calibration error across levels 0.10–0.95 | `coverage.csv` |
| Downstream utility | Digital-twin decision (at-risk-fraction error); train-on-synthetic, test-on-real accuracy vs a real-data ceiling | `results/_summary/`, `tstr.csv` |
| Transfer | Ensemble disagreement, fidelity, and coverage for a held-out group, with and without conformal recalibration | `loro.csv`, `results/_summary/` |

## Terminology

The paper reserves "physics" / "mechanistic" for genuine physical laws imposed a
priori — the published citrus cold-storage kinetics and the mass-balance /
compositional conservation laws that hold in the real fruit. Data-fitted
parent–child relationships are "structural priors", not physics. (In the code the
graph field is still named `CAUSAL_GRAPH` for backward compatibility; it encodes
the structural graph.)
