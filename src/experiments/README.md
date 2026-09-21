# Supplementary analysis scripts

Analysis scripts that produce the supplementary tables, separate from the main
pipeline (`make run` / `make loro` / `python -m src.multiseed`). Run each from
the repository root. All are CPU-feasible; the ones that train ensembles take a
few minutes.

| Script | Produces | Supplement |
|---|---|---|
| `python -m src.experiments.deep_baselines` | `results/_summary/deep_baselines.csv` — correlation distance for CTGAN, TVAE, MCMC and PI-VAE. | Table S7 |
| `python -m src.experiments.ablation` | `results/_summary/component_ablation.csv` — isolates the imposed conservation physics vs the marginal calibration (VAE / +conservation / +calibration / both) on the conservation-law datasets, with the conservation-law violation rate; plus `results/_summary/calibration_sweep.csv` — VAE vs VAE+calibration across all six datasets. The data-fitted edge term is off throughout (`VAE_PHYSICS_WEIGHT = 0`). | Table S8 |
| `python -m src.experiments.citrus_digital_twin` | `results/_summary/citrus_digital_twin.csv` — citrus chilling-injury at-risk-fraction decision (leave-one-condition-out): single value vs bootstrap vs conditional VFP, per CI tolerance. Needs the citrus raw file (on request). | Table S9 |
| `python -m src.experiments.transfer_decision` | `results/_summary/transfer_decision.csv` — data-scarce cultivar-transfer decision (representative, bootstraps, empirical-Bayes, shrinkage-MVN, VFP). | Table S10 |
| `python -m src.experiments.citrus_fewshot_conformal` | `results/_summary/citrus_fewshot_conformal.csv` — off-support coverage@90 recovered by few-shot conformal recalibration (citrus leave-one-condition-out). Needs the citrus raw file (on request). | Table S14 |
| `python -m src.experiments.conformal_calibration` | `results/_summary/conformal_calibration.csv` — in-distribution coverage before and after split-conformal recalibration. | Table S13 |
| `python -m src.experiments.nutrient_variability` | `results/_summary/nutrient_variability.csv` — individual-level nutrient variability (vitamin / minerals / fatty acids); single-average vs VFP marginal KS. | Table S17 |
| `python -m src.experiments.drift_detection` | drift scores against a reference population (distribution-distance monitor). | Table S19 |

The two citrus experiments (`citrus_digital_twin`, `citrus_fewshot_conformal`)
condition a VAE on storage temperature and duration and impose the rind
conservation constraints. `citrus_digital_twin` additionally conditions on the
published chilling-injury damage integral Omega(T, t) (Onwude et al. 2024) as a
physics-informed input. They expect the citrus raw file at
`data/cleaned-citrus-exp6.csv`, which is the authors' unpublished data (available
on request), so they are not runnable from the public checkout alone.
