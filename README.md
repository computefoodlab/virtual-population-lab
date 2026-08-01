# Virtual Population Lab

Code for the feasibility study behind *Hybrid generative virtual fresh food
populations for a data-scarce world*. Given a real dataset of individual food
items, the pipeline generates one synthetic population per modelling engine and
evaluates which engine best preserves the real population's marginal
distributions, joint dependence, uncertainty calibration, and downstream
usefulness.

Everything dataset-specific (feature columns, assumed causal graph, whether the
source data is pre-scaled, loss weights) lives in a single config module, so
pointing the pipeline at a new dataset means writing a new config rather than
touching modelling code.

Headline results: [insights.md](insights.md). Per-dataset auto-generated
reports: `reports/<config_name>/REPORT.md`.

## The four engines

Engine keys used in code, filenames, and result CSVs map to the paper's
abbreviations as follows:

| Repo key | Paper | Module |
|---|---|---|
| `physics_mc` | PI-MC | [src/generators/physics_mc_generator.py](src/generators/physics_mc_generator.py) |
| `regression` | RFR | [src/generators/regression_generator.py](src/generators/regression_generator.py) |
| `vae` | VAE | [src/generators/vae_generator.py](src/generators/vae_generator.py) |
| `hybrid_vae` | PI-VAE | [src/generators/hybrid_vae_generator.py](src/generators/hybrid_vae_generator.py) |

1. **Physics-informed Monte Carlo** — ancestral sampling over a supplied causal
   graph. Root variables are drawn from their real marginals; each
   (parent, child) edge is fit as a linear regression on the training data, so
   slopes, intercepts, and residual noise are estimated rather than assumed.
   Every conditional is Gaussian, so plain forward sampling is exact and no
   MCMC is required.

2. **Regression** — one `RandomForestRegressor` per feature, predicting it from
   the others on the training split. Synthetic rows are seeded from out-of-bag
   predictions (each row predicted only by the trees that did not bag it, so it
   is leakage-free without a separate held-out set) plus noise bootstrapped from
   the OOB residuals.

3. **VAE** — learns a latent joint distribution over all features and samples
   new rows from its prior. Trained with the standard ELBO (reconstruction plus
   KL to the standard normal, `beta` annealed from 0 over the first
   `kl_warmup_frac` of training) and a covariance-matching term (`cov_weight`).

4. **Physics-informed VAE (PI-VAE)** — the VAE above plus a physics-consistency
   term on the fitted causal edges, a marginal-matching term, and an
   empirical-copula calibration post-step. `physics_weight=0` recovers the plain
   VAE; an empty causal graph makes the physics term zero.

Each generator module exposes a single `generate(...)` function with a
consistent calling convention.

## Layout

```
src/
  configs/
    base.py                  # every config field, documented once, with defaults
    <dataset>.py             # imports base.py, overrides only dataset-specific fields
  generators/                # physics_mc, regression, vae, hybrid_vae, conditional_vae
  evaluation/
    evaluate.py              # correlation, PCA, KS, ECDF, volcano, summary metrics
    coverage.py              # interval coverage + calibration error
    tstr.py                  # train-on-synthetic, test-on-real
    loro.py                  # same-model near/far transfer + conformal recalibration
  config_loader.py           # resolves a config module by name
  data_loading.py            # loads and cleans the source CSV for a config
  main.py                    # runs every engine, evaluates, writes the report
  run_loro.py                # standalone near/far transfer run
  check_generalization.py    # re-checks generated data against train vs test
  multiseed.py               # repeats everything across seeds, mean +/- s.d.
  ablation.py                # PI-VAE loss-term ablation
  summary_figures.py         # cross-dataset figures
  loss_table.py              # loss-term and hyperparameter tables
  paper_algorithms.py        # per-engine equations and the PI-VAE algorithm
  results_table.py           # consolidates every dataset's result CSVs
  experiments/               # supplementary analyses (see its README)
results/<config_name>/
  synthetic_data/            # generated rows per engine
  figures/                   # correlation, marginal, and transfer figures
  summary_metrics.csv        # correlation distance and mean KS per engine
  ks_marginals.csv           # per-feature KS statistic and p-value per engine
  coverage.csv               # coverage@90 and calibration error per engine
  tstr.csv                   # downstream accuracy per engine plus the real ceiling
  loro.csv                   # near/far transfer, per group plus a MEAN row
  generalization_gap.csv     # correlation distance vs train split and vs test split
results/_summary/            # cross-dataset tables, figures, and LaTeX fragments
reports/<config_name>/
  REPORT.md                  # regenerated by main.py on every run; do not hand-edit
```

Outputs are namespaced by config name, so results from different datasets never
overwrite each other. Per-dataset figures are regenerable and excluded from the
repository.

## Running it

```
make setup                       # create venv, install requirements
make run CONFIG=<config_name>    # generation + evaluation + report for one dataset
make loro CONFIG=<config_name>   # near/far transfer (requires LORO_GROUP in the config)
make check-overfit CONFIG=<name> # re-check the last run's synthetic data, no retraining
make run-all                     # make run for every finalized dataset
make loro-all                    # make loro for every finalized dataset
make figures                     # cross-dataset summary figures
make all                         # run-all, then loro-all, then figures
```

Multi-seed aggregation and the supplementary tables are separate entry points:

```
python -m src.multiseed --seeds 41 42 43 44 45 --loro   # mean +/- s.d. tables
python -m src.ablation                                  # PI-VAE loss-term ablation
python -m src.loss_table                                # loss terms + hyperparameters
python -m src.paper_algorithms                          # equations + algorithm
python -m src.summary_figures                           # figures 1-3
```

`make run` logs the active config and each pipeline stage, prints a bordered
summary table (best value per metric marked), and rewrites
`reports/<config_name>/REPORT.md` from that run's numbers and figures. All runs
are CPU-feasible.

### Adding a new dataset

Write a new module in `src/configs/` (copy an existing one as a template):
`from src.configs.base import *`, then override at minimum `DATA_PATH`,
`FEATURES`, `ID_COLUMNS`, `LABEL_COLUMN`, `IS_PRE_SCALED`, `ROOT_VARIABLES`,
`CAUSAL_GRAPH`, and `DATA_SOURCE_URL`. Set `LORO_GROUP` to enable the near/far
transfer experiment and `RUN_TSTR` to enable the downstream check. Every field's
meaning is documented once in [src/configs/base.py](src/configs/base.py), which
also carries defaults for pipeline mechanics and loss weights. Run it with
`make run CONFIG=<your_module_name>` — the config is selected at runtime, so no
other file needs editing.

### Generalization check

Each engine's correlation distance is scored against the held-out test split.
The generalization check compares that same metric against the train split the
engine was fit on: a small gap is evidence of genuine generalization, while
train being much better than test would indicate fitting training-split noise.
Results land in `results/<config_name>/generalization_gap.csv` and the report's
"Generalization Check" section. `make check-overfit` re-runs only this check
against already-generated synthetic data.

## Evaluation

| Axis | Metric | Where |
|---|---|---|
| Joint dependence | Euclidean distance between real and generated correlation matrices | `summary_metrics.csv` |
| Marginals | Two-sample KS statistic and p-value per feature | `ks_marginals.csv` |
| Uncertainty | Coverage of the central 90% interval on real held-out values; calibration error across levels 0.10–0.95 | `coverage.csv` |
| Downstream utility | Train-on-synthetic, test-on-real accuracy vs a real-data ceiling | `tstr.csv` |
| Transfer | Ensemble disagreement, fidelity, and coverage for a held-out group, with and without conformal recalibration | `loro.csv` |

---

# Datasets

All five datasets are public. Cleaned copies used by the pipeline are in
[data/](data/).

| Config | Source | Rows | Features |
|---|---|---|---|
| `apple_quality` | [Kaggle — Apple Quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) | 4,000 | Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity (+ Quality label) |
| `banana_quality` | [Kaggle — Banana Quality Dataset](https://www.kaggle.com/datasets/mrmars1010/banana-quality-dataset) | 1,000 | ripeness_index, sugar_content_brix, firmness_kgf, length_cm, weight_g (+ quality_category label, region group) |
| `biofood_date_region` | [FAO/INFOODS](https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/) | 77 | Water, Potassium, Iron, Calcium, Magnesium (Region group) |
| `biofood_safou_region` | [FAO/INFOODS](https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/) | 41 | Water, Fat, Palmitic, Stearic (Region group) |
| `mango_composition` | [Mendeley Data b9d6s7hr33](https://data.mendeley.com/datasets/b9d6s7hr33/1) | 186 | VitaminC, TA, SSC (Cultivar group) |

## Apple Quality

```
@misc{nidula_elgiriyewithana_2024,
    title={Apple Quality},
    url={https://www.kaggle.com/dsv/7384155},
    DOI={10.34740/KAGGLE/DSV/7384155},
    publisher={Kaggle},
    author={Nidula Elgiriyewithana},
    year={2024}
}
```

| Feature | Description |
|---|---|
| Size | Physical size of apple |
| Weight | Apple mass |
| Sweetness | Sugar-related attribute |
| Crunchiness | Texture property |
| Juiciness | Moisture-related property |
| Ripeness | Maturity indicator |
| Acidity | Chemical characteristic |
| Quality | Overall quality label (excluded from generation, categorical) |

Values in this CSV are already standardized (mean 0, std 1) — see
`IS_PRE_SCALED` in [src/configs/apple_quality.py](src/configs/apple_quality.py).
The other four datasets carry raw units, and the pipeline fits its own scaler on
the training split.
