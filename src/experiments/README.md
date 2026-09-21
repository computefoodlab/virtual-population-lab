# Supplementary analysis scripts

Analysis scripts that produce the supplementary tables, separate from the main
pipeline (`make run` / `make loro` / `python -m src.multiseed`). Run each from
the repository root. All are CPU-feasible; the ones that train ensembles take a
few minutes.

| Script | Produces | Supplement |
|---|---|---|
| `python -m src.experiments.deep_baselines` | `results/_summary/deep_baselines.csv` — correlation distance for CTGAN, TVAE, MCMC and PI-VAE. | Table S7 |
| `python -m src.experiments.ablation` | `results/_summary/component_ablation.csv` — VAE / +calibration / PI-VAE±calibration on the datasets with genuine multivariate structure. | Table S8 |
| `python -m src.experiments.transfer_decision` | `results/_summary/transfer_decision.csv` — data-scarce cultivar-transfer decision (representative, bootstraps, empirical-Bayes, shrinkage-MVN, VFP). | Table S10 |
| `python -m src.experiments.conformal_calibration` | `results/_summary/conformal_calibration.csv` — in-distribution coverage before and after split-conformal recalibration. | Table S13 |
| `python -m src.experiments.drift_detection` | drift scores against a reference population (distribution-distance monitor). | Table S19 |
