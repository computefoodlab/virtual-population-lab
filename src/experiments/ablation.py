"""
Component ablation (Reviewer comment 5 / Daniel's request): separate how much of
the PI-VAE's performance comes from the generative model, the mechanistic
constraints, and the empirical-quantile calibration -- independently.

Four cells, all on the same VAE substrate (hybrid_vae with knobs toggled):
  1. VAE               : no constraints, no calibration      (physics=0, cov=0, cal off)
  2. VAE + calibration : + empirical-quantile post-processing (physics=0, cov=0, cal on)
  3. PI-VAE - calib    : + mechanistic/structure constraints  (physics>0, cov>0, cal off)
  4. PI-VAE + calib    : full model                           (physics>0, cov>0, cal on)

Reported per dataset (mean over seeds): correlation distance, mean KS, energy
distance (fidelity) and coverage@90 + calibration error (uncertainty). This shows
what calibration does to *calibration* vs what the constraints do to *fidelity*.

Run:  python -m src.experiments.ablation
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.evaluation.coverage import coverage_metrics
from src.multiseed import energy_distance
import pandas as pd

OUT = "results/_summary"
SEEDS = [41, 42, 43, 44, 45]
DATASETS = ["citrus_exp6", "tomato_nir", "grape_berry", "apple_samnegard"]
CELLS = [
    ("VAE",            0.0, 0.0, False),
    ("VAE+calib",      0.0, 0.0, True),
    ("PI-VAE-calib",   None, None, False),   # None -> use cfg physics/cov weights
    ("PI-VAE+calib",   None, None, True),
]


def gen(x, cfg, sc, seed, cov_w, phys_w, calib, calib_ref=None):
    np.random.seed(seed); torch.manual_seed(seed)
    return hv.generate(
        x, cfg.FEATURES, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM,
        epochs=cfg.VAE_EPOCHS, beta=cfg.VAE_BETA, n_samples=1000, scaler=sc,
        use_minibatch=cfg.VAE_USE_MINIBATCH, batch_size=cfg.VAE_BATCH_SIZE,
        cov_weight=cov_w, physics_weight=phys_w, marginal_weight=cfg.VAE_MARGINAL_WEIGHT,
        prior_type=cfg.VAE_PRIOR_TYPE, constrain_generated=cfg.VAE_CONSTRAIN_GENERATED,
        patience=cfg.VAE_PATIENCE, hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
        free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=calib, calib_reference=calib_ref)


def run(name):
    cfg = load_config(name); F = cfg.FEATURES; df = load_data(cfg)
    acc = {c[0]: {"corr": [], "ks": [], "energy": [], "cov90": [], "cal": []} for c in CELLS}
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        for label, cov_w, phys_w, calib in CELLS:
            cw = cfg.VAE_COV_WEIGHT if cov_w is None else cov_w
            pw = cfg.VAE_PHYSICS_WEIGHT if phys_w is None else phys_w
            g = gen(x, cfg, sc, seed, cw, pw, calib, calib_ref=tr[F].values)
            cov = coverage_metrics({"g": g}, te, F).iloc[0]
            acc[label]["corr"].append(correlation_euclidean_dist(te, g, F))
            acc[label]["ks"].append(marginal_ks_table(te, {"g": g}, F)["ks_stat"].mean())
            acc[label]["energy"].append(energy_distance(te, g, F, seed=seed))
            acc[label]["cov90"].append(float(cov["coverage_at_90"]))
            acc[label]["cal"].append(float(cov["calibration_error"]))
    print(f"\n=== {name} ({len(SEEDS)} seeds) — lower better except cov@90 (target 0.90) ===")
    print(f"{'cell':<15}{'corr':>9}{'mean_ks':>9}{'energy':>9}{'cov@90':>9}{'calib_err':>11}")
    rows = []
    metric_key = {"corr": "corr_dist", "ks": "mean_ks", "energy": "energy_dist",
                  "cov90": "coverage_at_90", "cal": "calib_err"}
    for label in [c[0] for c in CELLS]:
        a = acc[label]
        print(f"{label:<15}{np.mean(a['corr']):>9.3f}{np.mean(a['ks']):>9.3f}"
              f"{np.mean(a['energy']):>9.3f}{np.mean(a['cov90']):>9.3f}{np.mean(a['cal']):>11.3f}")
        for k, mk in metric_key.items():
            rows.append({"dataset": name, "cell": label, "metric": mk,
                         "mean": float(np.mean(a[k])), "std": float(np.std(a[k]))})
    return rows


def main():
    import os
    allrows = []
    for name in DATASETS:
        allrows += run(name)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(allrows).to_csv(f"{OUT}/component_ablation.csv", index=False)
    print("\n(VAE->+calib isolates the quantile post-processing; VAE->PI-VAE isolates the "
          "mechanistic/structure constraints; both stacked = full model.)")
    print(f"wrote: {OUT}/component_ablation.csv")


if __name__ == "__main__":
    main()
