"""
No lever recovers a banana correlation win.

Sweeps the Physics-VAE latent dimension on banana (near-independent features)
across 10 seeds and compares to the regression baseline. Confirms correlation
distance is pinned at the sampling-noise floor: no latent dim systematically
beats the baseline, and KS/calibration/coverage are unmoved (set by calibration).

Run from repo root:  python -m src.experiments.banana_latent_sweep
"""
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, torch, logging
logging.disable(logging.CRITICAL)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv, regression_generator as rg
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.evaluation.coverage import coverage_metrics

SEEDS = list(range(41, 51))
LATENTS = [2, 3, 4, 5, 6, 8]


def met(g, te, F):
    ks = marginal_ks_table(te, {"g": g}, F)["ks_stat"].mean()
    cov = coverage_metrics({"g": g}, te, F).iloc[0]
    return (correlation_euclidean_dist(te, g, F), float(ks),
            float(cov["calibration_error"]), float(cov["coverage_at_90"]))


def main():
    cfg = load_config("banana_quality"); F = cfg.FEATURES; df = load_data(cfg)
    base, acc = [], {L: [] for L in LATENTS}
    for s in SEEDS:
        np.random.seed(s); tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=s)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        base.append(correlation_euclidean_dist(te, rg.generate(tr, F, n_samples=1000), F))
        for L in LATENTS:
            torch.manual_seed(s)
            g = hv.generate(x, F, cfg.CAUSAL_GRAPH, latent_dim=L, epochs=cfg.VAE_EPOCHS, beta=cfg.VAE_BETA,
                            n_samples=1000, scaler=sc, use_minibatch=cfg.VAE_USE_MINIBATCH,
                            batch_size=cfg.VAE_BATCH_SIZE, cov_weight=cfg.VAE_COV_WEIGHT,
                            physics_weight=cfg.VAE_PHYSICS_WEIGHT, marginal_weight=cfg.VAE_MARGINAL_WEIGHT,
                            prior_type=cfg.VAE_PRIOR_TYPE, constrain_generated=cfg.VAE_CONSTRAIN_GENERATED,
                            patience=cfg.VAE_PATIENCE, hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
                            free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=cfg.VAE_CALIBRATE_MARGINALS)
            acc[L].append(met(g, te, F))
    print(f"Regression baseline corr: {np.mean(base):.3f} ± {np.std(base, ddof=1):.3f}\n")
    print(f"{'latent':>7}{'corr':>16}{'KS':>16}{'calib':>16}{'cov90':>16}{'beats base':>14}")
    for L in LATENTS:
        v = np.array(acc[L]); f = lambda i: f"{v[:, i].mean():.3f}±{v[:, i].std(ddof=1):.3f}"
        wins = int((v[:, 0] < np.array(base)).sum())
        print(f"{L:>7}{f(0):>16}{f(1):>16}{f(2):>16}{f(3):>16}{f'{wins}/{len(SEEDS)}':>14}")


if __name__ == "__main__":
    main()
