"""
Individual-level nutrient variability across nutrient classes.

Shows that individual food items vary substantially in nutrient content, and that
the VFP reproduces the nutrient DISTRIBUTION where a single average value cannot -
across a vitamin (mango Vitamin C), minerals (grape K/Mg/Ca) and fatty acids
(safou fat/palmitic/stearic). For each class it reports the real individual spread
(fold range and coefficient of variation) and the marginal KS of a single mean
value vs the VFP, both against held-out real fruit (lower KS = closer).

Run:  python -m src.experiments.nutrient_variability
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv

CASES = [
    ("mango_composition", "vitamin (Vitamin C)", ["VitaminC"]),
    ("grape_berry", "minerals (K, Mg, Ca)", ["Potassium", "Magnesium", "Calcium"]),
    ("biofood_safou_region", "fatty acids (fat, palmitic, stearic)", ["Fat", "Palmitic", "Stearic"]),
]
SEEDS = [41, 42, 43]
OUT = "results/_summary"


def run():
    records = []
    print(f"\n=== Individual-level nutrient variability ({len(SEEDS)} seeds) ===\n")
    print(f"{'commodity / nutrient class':<44}{'real spread':>18}{'KS avg':>9}{'KS VFP':>9}")
    for name, label, nutr in CASES:
        cfg = load_config(name)
        F = cfg.FEATURES
        df = load_data(cfg)

        folds, cvs = [], []
        for c in nutr:
            x = df[c][df[c] > 0]
            folds.append(x.max() / x.min())
            cvs.append(x.std() / x.mean())

        ks_avg, ks_vfp = [], []
        for s in SEEDS:
            tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=s)
            sc = StandardScaler().fit(tr[F])
            np.random.seed(s); torch.manual_seed(s)
            g = hv.generate(
                sc.transform(tr[F]), F, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM,
                epochs=cfg.VAE_EPOCHS, beta=cfg.VAE_BETA, n_samples=1500, scaler=sc,
                use_minibatch=cfg.VAE_USE_MINIBATCH, batch_size=cfg.VAE_BATCH_SIZE,
                cov_weight=cfg.VAE_COV_WEIGHT, physics_weight=cfg.VAE_PHYSICS_WEIGHT,
                marginal_weight=cfg.VAE_MARGINAL_WEIGHT, prior_type=cfg.VAE_PRIOR_TYPE,
                constrain_generated=cfg.VAE_CONSTRAIN_GENERATED, patience=cfg.VAE_PATIENCE,
                hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
                free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=cfg.VAE_CALIBRATE_MARGINALS)
            gd = pd.DataFrame(g, columns=F)
            a = [ks_2samp(np.full(len(te), tr[c].mean()), te[c]).statistic for c in nutr]
            v = [ks_2samp(gd[c], te[c]).statistic for c in nutr]
            ks_avg.append(np.mean(a)); ks_vfp.append(np.mean(v))

        spread = f"{np.mean(folds):.1f}x, CV {np.mean(cvs):.2f}"
        tag = f"{name} ({label})"
        ka, kv = float(np.mean(ks_avg)), float(np.mean(ks_vfp))
        print(f"{tag:<44}{spread:>18}{ka:>9.3f}{kv:>9.3f}")
        records.append({"commodity": name, "nutrient_class": label,
                        "fold": round(float(np.mean(folds)), 2), "cv": round(float(np.mean(cvs)), 3),
                        "ks_single_average": round(ka, 4), "ks_vfp": round(kv, 4)})

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(records).to_csv(f"{OUT}/nutrient_variability.csv", index=False)
    print("\n(real fold = max/min across individual fruit; KS avg = a single mean value vs "
          "real; KS VFP = generated distribution vs real; lower KS is better.)")
    print(f"wrote: {OUT}/nutrient_variability.csv")


if __name__ == "__main__":
    run()
