"""
Region-conditional physics underperforms pooling.

Tests whether the subpopulation heterogeneity (per-region or per-cultivar slope
differences) can be exploited: fit physics edges per group and condition the VAE
on group, versus the pooled Physics-VAE. Pooling wins at these per-group sample
sizes — a bias-variance outcome.

Compares, on the pooled held-out test set:
  pooled-PVAE   the shipped PI-VAE (unconditional, one pooled edge set)   [baseline]
  region-VAE    CVAE conditioned on group, per-group calibration, NO physics
  region-PVAE   CVAE conditioned on group + PER-GROUP physics + per-group calibration

Run from repo root:  python -m src.experiments.region_conditional
"""
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, torch, logging
logging.disable(logging.CRITICAL)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config_loader import load_config
from src.data_loading import load_data
from src.generators.conditional_vae_generator import CVAE
from src.generators.hybrid_vae_generator import _fit_edges, _physics_loss, _calibrate_marginals
from src.generators import hybrid_vae_generator as hv
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.evaluation.coverage import coverage_metrics

SEEDS = [41, 42, 43]
MIN_REGION_ROWS = 8   # per-group edges only if the group has >= this many train rows; else pooled


def _onehot(i, k):
    v = np.zeros(k, np.float32); v[i] = 1.0; return v


def train_region_pvae(x, region_idx, k, edges_by_region, cfg, physics_weight, seed):
    torch.manual_seed(seed)
    cond = np.stack([_onehot(r, k) for r in region_idx])
    xt, ct = torch.FloatTensor(x), torch.FloatTensor(cond)
    ridx = torch.LongTensor(region_idx)
    model = CVAE(x.shape[1], k, cfg.LATENT_DIM, cfg.VAE_HIDDEN_DIM)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    warmup = max(1, int(cfg.VAE_EPOCHS * 0.3))
    best, since = float("inf"), 0
    for ep in range(cfg.VAE_EPOCHS):
        beta = cfg.VAE_BETA * min(1.0, (ep + 1) / warmup)
        recon, mu, logvar = model(xt, ct)
        recon_loss = ((recon - xt) ** 2).mean()
        kl = torch.clamp(0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar), min=cfg.VAE_FREE_BITS).sum(1).mean()
        phys = recon.new_zeros(())
        if physics_weight > 0:
            cnt = 0
            for r, edges in edges_by_region.items():
                m = ridx == r
                if m.sum() >= 3 and edges:
                    phys = phys + _physics_loss(recon[m], edges); cnt += 1
            phys = phys / max(cnt, 1)
        loss = recon_loss + beta * kl + physics_weight * phys
        opt.zero_grad(); loss.backward(); opt.step()
        if ep >= warmup:
            if loss.item() < best - 1e-4:
                best, since = loss.item(), 0
            else:
                since += 1
                if since >= cfg.VAE_PATIENCE:
                    break
    return model


def generate_region(model, k, region_frac, region_scaled, scaler, latent_dim, n_total=1000):
    model.eval(); out = []
    with torch.no_grad():
        for r, frac in region_frac.items():
            nr = max(2, int(round(frac * n_total)))
            z = torch.randn(nr, latent_dim)
            c = torch.FloatTensor(np.tile(_onehot(r, k), (nr, 1)))
            g = model.decoder(torch.cat([z, c], dim=1)).numpy()
            if r in region_scaled and len(region_scaled[r]) >= 5:
                g = _calibrate_marginals(g, region_scaled[r])
            out.append(g)
    return scaler.inverse_transform(np.vstack(out))


def metrics(gen, te, F):
    import pandas as pd
    gdf = pd.DataFrame(gen, columns=F)
    ks = marginal_ks_table(te, {"g": gdf}, F)["ks_stat"].mean()
    cov = coverage_metrics({"g": gdf}, te, F).iloc[0]
    return (correlation_euclidean_dist(te, gdf, F), float(ks),
            float(cov["calibration_error"]), float(cov["coverage_at_90"]))


def run(name):
    cfg = load_config(name); F = cfg.FEATURES; G = cfg.LORO_GROUP
    df = load_data(cfg); regions = sorted(df[G].unique()); k = len(regions)
    rmap = {r: i for i, r in enumerate(regions)}
    print(f"\n===== {name}  regions={regions} =====")
    res = {"pooled-PVAE": [], "region-VAE": [], "region-PVAE": []}
    for seed in SEEDS:
        np.random.seed(seed)
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)
        sc = StandardScaler().fit(tr[F]); xtr = sc.transform(tr[F])
        ridx = np.array([rmap[r] for r in tr[G]])
        pooled_edges = _fit_edges(xtr.astype(np.float32), F, cfg.CAUSAL_GRAPH)
        edges_by_region, region_scaled, region_frac = {}, {}, {}
        for r, i in rmap.items():
            rows = xtr[ridx == i]; region_scaled[i] = rows; region_frac[i] = (ridx == i).mean()
            edges_by_region[i] = (_fit_edges(rows.astype(np.float32), F, cfg.CAUSAL_GRAPH)
                                  if len(rows) >= MIN_REGION_ROWS else pooled_edges)
        torch.manual_seed(seed)
        g0 = hv.generate(xtr, F, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM, epochs=cfg.VAE_EPOCHS,
                         beta=cfg.VAE_BETA, n_samples=1000, scaler=sc, use_minibatch=cfg.VAE_USE_MINIBATCH,
                         batch_size=cfg.VAE_BATCH_SIZE, cov_weight=cfg.VAE_COV_WEIGHT,
                         physics_weight=cfg.VAE_PHYSICS_WEIGHT, marginal_weight=cfg.VAE_MARGINAL_WEIGHT,
                         prior_type=cfg.VAE_PRIOR_TYPE, constrain_generated=cfg.VAE_CONSTRAIN_GENERATED,
                         patience=cfg.VAE_PATIENCE, hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
                         free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=True)
        res["pooled-PVAE"].append(metrics(g0.values, te, F))
        for label, pw in [("region-VAE", 0.0), ("region-PVAE", cfg.VAE_PHYSICS_WEIGHT)]:
            m = train_region_pvae(xtr, ridx, k, edges_by_region, cfg, pw, seed)
            res[label].append(metrics(generate_region(m, k, region_frac, region_scaled, sc, cfg.LATENT_DIM), te, F))
    print(f"{'engine':<14}{'corr':>16}{'KS':>16}{'calib':>16}{'cov90':>16}")
    for lab, vals in res.items():
        v = np.array(vals); f = lambda i: f"{v[:, i].mean():.3f}±{v[:, i].std(ddof=1):.3f}"
        print(f"{lab:<14}{f(0):>16}{f(1):>16}{f(2):>16}{f(3):>16}")


def single_region_subset(cfg_name="biofood_date_region", region="UAE", seeds=(41, 42, 43, 44, 45)):
    """Single-region subsetting: does restricting to the largest region beat
    pooling? Runs the shipped Physics-VAE on that region only and reports
    correlation distance."""
    cfg = load_config(cfg_name); F = cfg.FEATURES; G = cfg.LORO_GROUP
    df = load_data(cfg); df = df[df[G] == region].reset_index(drop=True)
    corr = []
    for s in seeds:
        np.random.seed(s)
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=s)
        sc = StandardScaler().fit(tr[F]); x = sc.transform(tr[F])
        torch.manual_seed(s)
        g = hv.generate(x, F, cfg.CAUSAL_GRAPH, latent_dim=cfg.LATENT_DIM, epochs=cfg.VAE_EPOCHS,
                        beta=cfg.VAE_BETA, n_samples=1000, scaler=sc, use_minibatch=cfg.VAE_USE_MINIBATCH,
                        batch_size=cfg.VAE_BATCH_SIZE, cov_weight=cfg.VAE_COV_WEIGHT,
                        physics_weight=cfg.VAE_PHYSICS_WEIGHT, marginal_weight=cfg.VAE_MARGINAL_WEIGHT,
                        prior_type=cfg.VAE_PRIOR_TYPE, constrain_generated=cfg.VAE_CONSTRAIN_GENERATED,
                        patience=cfg.VAE_PATIENCE, hidden_dim=cfg.VAE_HIDDEN_DIM, dropout=cfg.VAE_DROPOUT,
                        free_bits=cfg.VAE_FREE_BITS, calibrate_marginals=True)
        corr.append(correlation_euclidean_dist(te, g, F))
    print(f"\n{cfg_name} {region}-only (n={len(df)}): Physics-VAE corr dist = "
          f"{np.mean(corr):.3f} ± {np.std(corr, ddof=1):.3f}  "
          f"(compare the pooled Physics-VAE row printed above)")


def main():
    for ds in ["biofood_date_region", "biofood_safou_region"]:
        run(ds)
    single_region_subset("biofood_date_region", "UAE")


if __name__ == "__main__":
    main()
