# Supplementary analysis scripts

Analysis and ablation scripts behind the supplementary discussion, separate from
the main pipeline (`make run` / `make loro` / `python -m src.multiseed`). Run each
from the repository root.

| Script | What it shows |
|---|---|
| `python -m src.experiments.oracle_independent` | On independent-Gaussian data, PI-MC sits on the oracle floor and PI-VAE ties it — banana and mango being co-best on fidelity is the ceiling the data allows, not a shortfall. |
| `python -m src.experiments.structure_floor` | The banana correlation-distance noise floor (real train-vs-test ≈ 0.28), plus per-group OLS slope and correlation heterogeneity for date, mango, and safou. |
| `python -m src.experiments.region_conditional` | Region-conditional physics and single-region subsetting both underperform the pooled PI-VAE — a bias–variance outcome at small per-group n. |
| `python -m src.experiments.banana_latent_sweep` | No latent dimension recovers a banana correlation win; the metric is pinned at the noise floor. |
| `python -m src.experiments.conformal_prototype` | Standalone prototype of the conformal recalibration that ships in `src/evaluation/loro.py` (columns `coverage_*_conf`): fixes in-region coverage and closes about half the off-support gap. |

All scripts are CPU-feasible; the ones that train ensembles (`region_conditional`,
`conformal_prototype`, `banana_latent_sweep`) take a few minutes.
