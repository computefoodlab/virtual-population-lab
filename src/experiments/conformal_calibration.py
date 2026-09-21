"""
In-distribution calibration via split-conformal recalibration (Reviewer 1 #4).

The plain PI-VAE per-feature interval can under-cover on held-out real data. This
experiment measures coverage@90 and calibration error for the PI-VAE population
BEFORE and AFTER split-conformal recalibration: the real held-out set is split
into a calibration half (sets the conformal widening q) and an evaluation half
(where coverage is measured), so calibration never sees the evaluation points.

The honest, expected finding: conformal recovers nominal in-distribution
coverage@90 wherever n is adequate (citrus flagship, date, mango, tomato, apple),
and fails only at very small n (safou, n=41), where the calibration split is too
small for a stable finite-sample quantile. This is IN-DISTRIBUTION calibration;
off-support/transfer calibration remains open (Reviewer 1 #3).

Run:  python -m src.experiments.conformal_calibration
      python -m src.experiments.conformal_calibration citrus_exp6 --seeds 41 42 43 44 45
Writes: results/_summary/conformal_calibration.csv
"""
import sys, os; sys.path.insert(0, os.getcwd())
import argparse
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config
from src.data_loading import load_data
from src.main import generate_population
from src.evaluation.coverage import coverage_metrics, conformal_coverage_metrics

DATASETS = ["citrus_exp6", "tomato_nir", "grape_berry", "apple_samnegard",
            "mango_composition", "biofood_safou_region"]
SEEDS = [41, 42, 43, 44, 45]
OUT = "results/_summary"
ENGINE = "hybrid_vae"


def run(name, seeds):
    cfg = load_config(name)
    F = cfg.FEATURES
    df = load_data(cfg)
    base90, con90, basece, converr, ncal = [], [], [], [], []
    for s in seeds:
        cfg.RANDOM_SEED = s
        np.random.seed(s); torch.manual_seed(s)
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=s)
        cal, ev = train_test_split(te, test_size=0.5, random_state=s)
        scaler = None if cfg.IS_PRE_SCALED else StandardScaler().fit(tr[F])
        torch.manual_seed(s)
        gen = {ENGINE: generate_population(tr, cfg, scaler, quiet=True)[ENGINE]}
        b = coverage_metrics(gen, ev, F).set_index("method")
        c = conformal_coverage_metrics(gen, cal, ev, F).set_index("method")
        base90.append(float(b.loc[ENGINE, "coverage_at_90"]))
        con90.append(float(c.loc[ENGINE, "coverage_at_90"]))
        basece.append(float(b.loc[ENGINE, "calibration_error"]))
        converr.append(float(c.loc[ENGINE, "calibration_error"]))
        ncal.append(len(cal))

    def ms(v): return float(np.mean(v)), float(np.std(v))
    rows = []
    for metric, base, conf in [("coverage_at_90", base90, con90),
                               ("calibration_error", basece, converr)]:
        bm, bs = ms(base); cm, cs = ms(conf)
        rows.append({"dataset": name, "metric": metric,
                     "base_mean": bm, "base_std": bs,
                     "conf_mean": cm, "conf_std": cs})
    print(f"{name:<22} n_cal~{int(np.mean(ncal)):<4} "
          f"cov@90 {np.mean(base90):.3f} -> {np.mean(con90):.3f}   "
          f"cal.err {np.mean(basece):.3f} -> {np.mean(converr):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    configs = args.configs if args.configs else DATASETS
    os.makedirs(OUT, exist_ok=True)
    print(f"Conformal in-distribution calibration (PI-VAE, {len(args.seeds)} seeds)\n")
    allrows = []
    for ds in configs:
        allrows += run(ds, args.seeds)
    pd.DataFrame(allrows).to_csv(f"{OUT}/conformal_calibration.csv", index=False)
    print(f"\nwrote: {OUT}/conformal_calibration.csv")


if __name__ == "__main__":
    main()
