"""
Component ablation: isolate what the imposed PHYSICS (the conservation /
mass-balance constraint) and the empirical-quantile CALIBRATION each contribute
to the PI-VAE, and demonstrate the biological-plausibility benefit the physics
provides. Shown on the datasets that carry a genuine conservation law
(citrus, tomato, safou).

All cells share the same VAE substrate (covariance matching on; the data-fitted
edge term is off, VAE_PHYSICS_WEIGHT = 0). Only two knobs are toggled:

  1. VAE                : no conservation, no calibration
  2. VAE + conservation : + the physical conservation constraint (soft penalty +
                          hard feasibility projection at generation)
  3. VAE + calibration  : + empirical-quantile marginal calibration
  4. PI-VAE (both)       : conservation + calibration (the full model)

Reported per dataset (mean over seeds): correlation distance, mean KS, energy
distance (fidelity), calibration error, and the conservation-LAW VIOLATION RATE
in the generated population. The violation rate is the concrete physics benefit:
the conservation cell drives it to 0%, while a purely data-driven cell does not.

Run:  python -m src.experiments.ablation
(citrus needs the raw file at data/cleaned-citrus-exp6.csv - on request.)
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import numpy as np, torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv
from src.generators.hybrid_vae_generator import _build_constraints
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.evaluation.coverage import coverage_metrics
from src.multiseed import energy_distance

OUT = "results/_summary"
SEEDS = [41, 42, 43, 44, 45]
DATASETS = ["citrus_exp6", "tomato_nir", "biofood_safou_region"]   # have a conservation law
CELLS = [
    ("VAE",              False, False),
    ("VAE+conservation", True,  False),
    ("VAE+calibration",  False, True),
    ("PI-VAE (both)",    True,  True),
]


def violation_rate(gen_df, constraints, F):
    """Fraction of generated rows that violate the conservation law A x <= b."""
    if not constraints:
        return np.nan
    A, b = _build_constraints(constraints, F)
    X = gen_df[F].values
    viol = (X @ A.T - b) > 1e-6
    return float(viol.any(axis=1).mean())


def gen(x, cfg, sc, seed, use_cons, calib, calib_ref):
    np.random.seed(seed); torch.manual_seed(seed)
    cons = getattr(cfg, "CONSTRAINTS", None) if use_cons else None
    cw = getattr(cfg, "CONSTRAINT_WEIGHT", 0.0) if use_cons else 0.0
    return hv.generate(
        x, cfg.FEATURES, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM,
        epochs=cfg.VAE_EPOCHS, beta=cfg.VAE_BETA, n_samples=1000, scaler=sc,
        use_minibatch=cfg.VAE_USE_MINIBATCH, batch_size=cfg.VAE_BATCH_SIZE,
        cov_weight=cfg.VAE_COV_WEIGHT, physics_weight=0.0,   # edge term off (new architecture)
        marginal_weight=cfg.VAE_MARGINAL_WEIGHT, prior_type=cfg.VAE_PRIOR_TYPE,
        constrain_generated=cfg.VAE_CONSTRAIN_GENERATED, patience=cfg.VAE_PATIENCE,
        hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT, free_bits=cfg.VAE_FREE_BITS,
        constraints=cons, constraint_weight=cw,
        calibrate_marginals=calib, calib_reference=calib_ref)


def run(name):
    cfg = load_config(name); F = cfg.FEATURES; df = load_data(cfg)
    cons_def = getattr(cfg, "CONSTRAINTS", None)
    acc = {c[0]: {"corr": [], "ks": [], "energy": [], "cal": [], "viol": []} for c in CELLS}
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        for label, use_cons, calib in CELLS:
            g = gen(x, cfg, sc, seed, use_cons, calib, calib_ref=tr[F].values)
            cov = coverage_metrics({"g": g}, te, F).iloc[0]
            acc[label]["corr"].append(correlation_euclidean_dist(te, g, F))
            acc[label]["ks"].append(marginal_ks_table(te, {"g": g}, F)["ks_stat"].mean())
            acc[label]["energy"].append(energy_distance(te, g, F, seed=seed))
            acc[label]["cal"].append(float(cov["calibration_error"]))
            acc[label]["viol"].append(violation_rate(g, cons_def, F))
    print(f"\n=== {name} ({len(SEEDS)} seeds) — lower better; viol% = conservation-law violations ===")
    print(f"{'cell':<18}{'corr':>9}{'mean_ks':>9}{'energy':>9}{'calib_err':>11}{'viol%':>8}")
    rows = []
    mk = {"corr": "corr_dist", "ks": "mean_ks", "energy": "energy_dist",
          "cal": "calib_err", "viol": "violation_rate"}
    for label in [c[0] for c in CELLS]:
        a = acc[label]
        print(f"{label:<18}{np.mean(a['corr']):>9.3f}{np.mean(a['ks']):>9.3f}"
              f"{np.mean(a['energy']):>9.3f}{np.mean(a['cal']):>11.3f}{np.nanmean(a['viol'])*100:>7.1f}%")
        for k, m in mk.items():
            rows.append({"dataset": name, "cell": label, "metric": m,
                         "mean": float(np.nanmean(a[k])), "std": float(np.nanstd(a[k]))})
    return rows


CALIB_DATASETS = ["citrus_exp6", "tomato_nir", "grape_berry", "apple_samnegard",
                  "mango_composition", "biofood_safou_region"]


def calib_sweep(name):
    """Calibration-only comparison (VAE vs VAE+calibration) on every dataset, to
    show the marginal-fit benefit broadly (conservation off, edge term off, so only
    calibration is toggled)."""
    cfg = load_config(name); F = cfg.FEATURES; df = load_data(cfg)
    out = {False: {"corr": [], "ks": []}, True: {"corr": [], "ks": []}}
    for seed in SEEDS:
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        for calib in (False, True):
            g = gen(x, cfg, sc, seed, use_cons=False, calib=calib, calib_ref=tr[F].values)
            out[calib]["corr"].append(correlation_euclidean_dist(te, g, F))
            out[calib]["ks"].append(marginal_ks_table(te, {"g": g}, F)["ks_stat"].mean())
    rows = []
    for calib in (False, True):
        lbl = "VAE+calibration" if calib else "VAE"
        rows.append({"dataset": name, "cell": lbl,
                     "corr_dist": round(float(np.mean(out[calib]["corr"])), 4),
                     "mean_ks": round(float(np.mean(out[calib]["ks"])), 4)})
    return rows


def main():
    allrows = []
    for name in DATASETS:
        allrows += run(name)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(allrows).to_csv(f"{OUT}/component_ablation.csv", index=False)
    print("\n(VAE->+conservation isolates the imposed physics — note the violation rate "
          "dropping to 0%; VAE->+calibration isolates the marginal calibration; the edge "
          "term is off throughout.)")
    print(f"wrote: {OUT}/component_ablation.csv")

    # Calibration-only sweep across all six datasets (marginal-fit benefit).
    print("\n=== calibration sweep (all six datasets): VAE vs VAE+calibration ===")
    print(f"{'dataset':<24}{'KS (VAE)':>10}{'KS (+calib)':>13}")
    crows = []
    for name in CALIB_DATASETS:
        r = calib_sweep(name)
        vae, cal = r[0], r[1]
        print(f"{name:<24}{vae['mean_ks']:>10.3f}{cal['mean_ks']:>13.3f}")
        crows += r
    pd.DataFrame(crows).to_csv(f"{OUT}/calibration_sweep.csv", index=False)
    print(f"wrote: {OUT}/calibration_sweep.csv")


if __name__ == "__main__":
    main()
