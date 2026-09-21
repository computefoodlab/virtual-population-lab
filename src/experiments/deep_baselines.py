"""
Modern deep tabular generators (CTGAN, TVAE) vs PI-VAE and the MCMC prior-art.

Answers Reviewer 1 #10: are stronger modern generative baselines missing? We add
CTGAN and TVAE (Xu et al., 2019) and compare them on the same fidelity metrics
(correlation distance, mean KS, energy distance, MMD) against real held-out data.
The expected, honest finding is that GAN/VAE-style deep generators need more data
than these datasets provide and overfit or underperform at small n, while
remaining competitive on the largest set (apple). Multi-seed mean +/- std.

Run:  python -m src.experiments.deep_baselines
Writes: results/_summary/deep_baselines.csv
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv, mcmc_generator
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.multiseed import energy_distance, mmd_rbf

from ctgan import CTGAN, TVAE

DATASETS = ["citrus_exp6", "tomato_nir", "grape_berry", "apple_samnegard",
            "mango_composition", "biofood_safou_region"]
SEEDS = [41, 42, 43]
EPOCHS = 300
N = 1000
OUT = "results/_summary"


def metrics(test_df, gen_df, F, seed):
    ks = marginal_ks_table(test_df, {"g": gen_df}, F)["ks_stat"].mean()
    return {
        "corr_dist": correlation_euclidean_dist(test_df, gen_df, F),
        "mean_ks": float(ks),
        "energy_dist": energy_distance(test_df, gen_df, F, seed=seed),
        "mmd": mmd_rbf(test_df, gen_df, F, seed=seed),
    }


def run(name):
    cfg = load_config(name)
    F = cfg.FEATURES
    df = load_data(cfg)
    rows = []
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)
        te_f = te[F].reset_index(drop=True)

        # CTGAN
        m = CTGAN(epochs=EPOCHS, verbose=False)
        m.fit(tr[F], discrete_columns=[])
        rows.append({"engine": "CTGAN", **metrics(te_f, m.sample(N)[F], F, seed)})

        # TVAE
        m = TVAE(epochs=EPOCHS)
        m.fit(tr[F], discrete_columns=[])
        rows.append({"engine": "TVAE", **metrics(te_f, m.sample(N)[F], F, seed)})

        # MCMC prior-art
        np.random.seed(seed)
        rows.append({"engine": "MCMC", **metrics(
            te_f, mcmc_generator.generate(tr, F, n_samples=N), F, seed)})

        # PI-VAE (respect IS_PRE_SCALED like the main pipeline)
        sc = None if cfg.IS_PRE_SCALED else StandardScaler().fit(tr[F])
        x = tr[F].values if sc is None else sc.transform(tr[F])
        np.random.seed(seed); torch.manual_seed(seed)
        g = hv.generate(x, F, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM,
                        epochs=cfg.VAE_EPOCHS, beta=cfg.VAE_BETA, n_samples=N, scaler=sc,
                        use_minibatch=cfg.VAE_USE_MINIBATCH, batch_size=cfg.VAE_BATCH_SIZE,
                        cov_weight=cfg.VAE_COV_WEIGHT, physics_weight=cfg.VAE_PHYSICS_WEIGHT,
                        marginal_weight=cfg.VAE_MARGINAL_WEIGHT, prior_type=cfg.VAE_PRIOR_TYPE,
                        constrain_generated=cfg.VAE_CONSTRAIN_GENERATED, patience=cfg.VAE_PATIENCE,
                        hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
                        free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=cfg.VAE_CALIBRATE_MARGINALS)
        rows.append({"engine": "PI-VAE", **metrics(te_f, g, F, seed)})

    d = pd.DataFrame(rows)
    print(f"\n=== {name}  (n_train={int(len(df)*(1-cfg.TEST_SIZE))}) ===")
    print(f"{'engine':<10}{'corr_dist':>18}{'mean_ks':>16}{'energy':>16}{'mmd':>16}")
    out = []
    for e in ["CTGAN", "TVAE", "MCMC", "PI-VAE"]:
        s = d[d.engine == e]
        cells = {}
        for mkey in ["corr_dist", "mean_ks", "energy_dist", "mmd"]:
            cells[mkey] = (s[mkey].mean(), s[mkey].std(ddof=1) if len(s) > 1 else 0.0)
            out.append({"dataset": name, "engine": e, "metric": mkey,
                        "mean": cells[mkey][0], "std": cells[mkey][1]})
        print(f"{e:<10}" + "".join(f"{cells[k][0]:>10.3f}±{cells[k][1]:.3f}"
              for k in ["corr_dist", "mean_ks", "energy_dist", "mmd"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    allout = []
    for ds in DATASETS:
        allout += run(ds)
    pd.DataFrame(allout).to_csv(f"{OUT}/deep_baselines.csv", index=False)
    print(f"\n(mean±std over {len(SEEDS)} seeds; lower is better for all four metrics)")
    print(f"wrote: {OUT}/deep_baselines.csv")


if __name__ == "__main__":
    main()
