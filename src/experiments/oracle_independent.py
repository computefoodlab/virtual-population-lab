"""
Oracle analysis on independent-Gaussian data.

Shows that on data with no exploitable dependence structure, Physics-MC (which
with an empty graph reduces to an independent-Gaussian sampler) sits on the
oracle floor and Physics-VAE ties it — i.e. banana and mango being co-best on
fidelity is the ceiling the data allows, not a shortfall.

Run from repo root:  python -m src.experiments.oracle_independent
"""
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, torch, pandas as pd, logging
logging.disable(logging.CRITICAL)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.generators import physics_mc_generator as pm, regression_generator as rg
from src.generators import vae_generator as vg, hybrid_vae_generator as hv
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.evaluation.coverage import coverage_metrics

F = ["f1", "f2", "f3"]
GROUPS = {"A": 18, "B": 85, "C": 29, "D": 54}          # mango cultivar sizes
MEANS = {"A": [56, 500, 9.5], "B": [35, 450, 9.0], "C": [50, 520, 9.8], "D": [47, 480, 9.3]}
SDS = [10.0, 90.0, 1.0]                                 # per-feature sd; features INDEPENDENT


def make_real(seed):
    rng = np.random.RandomState(seed); rows = []
    for g, n in GROUPS.items():
        for _ in range(n):
            rows.append([rng.normal(MEANS[g][j], SDS[j]) for j in range(3)] + [g])
    return pd.DataFrame(rows, columns=F + ["Group"])


def oracle(train_df, n, seed):
    """Sample from the TRUE generating distribution (perfect DGP knowledge)."""
    rng = np.random.RandomState(seed + 999)
    counts = (train_df["Group"].value_counts(normalize=True) * n).round().astype(int)
    rows = []
    for g, c in counts.items():
        for _ in range(c):
            rows.append([rng.normal(MEANS[g][j], SDS[j]) for j in range(3)])
    return pd.DataFrame(rows, columns=F)


def met(g, te):
    ks = marginal_ks_table(te, {"g": g}, F)["ks_stat"].mean()
    cov = coverage_metrics({"g": g}, te, F).iloc[0]
    return (correlation_euclidean_dist(te, g, F), float(ks), float(cov["calibration_error"]))


def main():
    acc = {k: [] for k in ["ORACLE (floor)", "physics_mc", "regression", "vae", "hybrid_vae"]}
    for s in [41, 42, 43]:
        np.random.seed(s); torch.manual_seed(s)
        df = make_real(s)
        tr, te = train_test_split(df, test_size=0.3, random_state=s)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        acc["ORACLE (floor)"].append(met(oracle(tr, 1000, s), te))
        np.random.seed(s); acc["physics_mc"].append(met(pm.generate(tr, F, [], F, n_samples=1000), te))
        np.random.seed(s); acc["regression"].append(met(rg.generate(tr, F, n_samples=1000), te))
        torch.manual_seed(s); acc["vae"].append(met(vg.generate(
            x, F, latent_dim=3, epochs=2000, beta=1.0, n_samples=1000, scaler=sc, use_minibatch=False,
            batch_size=128, cov_weight=1.0, patience=300, hidden_dim=64, dropout=0.0, free_bits=1.0), te))
        torch.manual_seed(s); acc["hybrid_vae"].append(met(hv.generate(
            x, F, [], latent_dim=3, epochs=2000, beta=1.0, n_samples=1000, scaler=sc, use_minibatch=False,
            batch_size=128, cov_weight=1.0, physics_weight=1.0, marginal_weight=1.0, prior_type="standard",
            constrain_generated=True, patience=300, hidden_dim=64, dropout=0.0, free_bits=1.0,
            calibrate_marginals=True), te))
    print("Independent near-Gaussian data (3 feat, 4 groups, zero cross-corr). 3 seeds.")
    print(f"{'generator':<18}{'corr':>14}{'KS':>14}{'calib':>14}")
    for k, v in acc.items():
        v = np.array(v); f = lambda i: f"{v[:, i].mean():.3f}±{v[:, i].std(ddof=1):.3f}"
        print(f"{k:<18}{f(0):>14}{f(1):>14}{f(2):>14}")


if __name__ == "__main__":
    main()
