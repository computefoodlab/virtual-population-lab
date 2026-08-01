"""
Correlation noise floor and subpopulation heterogeneity.

Reports:
  (1) the correlation-distance noise floor on banana (real train-vs-test and
      two random halves) — shows independent-feature datasets are pinned at the
      floor;
  (2) per-group OLS slope and correlation heterogeneity for date (regions) and
      mango (cultivars) — shows why a single pooled causal edge is a compromise.

Run from repo root:  python -m src.experiments.structure_floor
"""
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, logging
logging.disable(logging.CRITICAL)
from sklearn.model_selection import train_test_split
from src.config_loader import load_config
from src.data_loading import load_data
from src.evaluation.evaluate import correlation_euclidean_dist


def noise_floor(cfg_name="banana_quality", seeds=range(41, 51)):
    cfg = load_config(cfg_name); F = cfg.FEATURES; df = load_data(cfg)
    tt, hh = [], []
    for s in seeds:
        tr, te = train_test_split(df, test_size=cfg.TEST_SIZE, random_state=s)
        tt.append(correlation_euclidean_dist(tr, te, F))
        h1, h2 = train_test_split(df, test_size=0.5, random_state=s)
        hh.append(correlation_euclidean_dist(h1, h2, F))
    print(f"\n[{cfg_name}] correlation-distance noise floor:")
    print(f"  real train vs real test : {np.mean(tt):.3f} ± {np.std(tt, ddof=1):.3f}  "
          f"(range {min(tt):.3f}-{max(tt):.3f})")
    print(f"  two random halves        : {np.mean(hh):.3f} ± {np.std(hh, ddof=1):.3f}")


def edge_heterogeneity(cfg_name, group_col=None):
    cfg = load_config(cfg_name); F = cfg.FEATURES
    G = group_col or cfg.LORO_GROUP
    df = load_data(cfg); regions = sorted(df[G].unique())
    print(f"\n[{cfg_name}] per-{G} correlation / slope heterogeneity (n per group: "
          f"{dict(df[G].value_counts())}):")
    # causal-graph edges if present, else all feature pairs
    pairs = cfg.CAUSAL_GRAPH or [(a, b) for i, a in enumerate(F) for b in F[i + 1:]]
    hdr = "edge/pair".ljust(20) + "pooled".rjust(9) + "".join(str(r).rjust(11) for r in regions)
    print("  " + hdr)
    for a, b in pairs:
        def rs(d):
            if len(d) < 3 or d[a].std() < 1e-9:
                return "n/a"
            r = np.corrcoef(d[a], d[b])[0, 1]
            sl = np.cov(d[a], d[b], ddof=0)[0, 1] / np.var(d[a])
            return f"{r:+.2f}/{sl:+.2f}"
        row = f"{a}->{b}".ljust(20) + rs(df).rjust(9) + "".join(rs(df[df[G] == r]).rjust(11) for r in regions)
        print("  " + row)


def main():
    noise_floor("banana_quality")
    edge_heterogeneity("biofood_date_region")   # region slopes differ 3-5x, Iron flips sign
    edge_heterogeneity("mango_composition")     # cultivar corrs weak + sign-inconsistent
    edge_heterogeneity("biofood_safou_region")  # near-deterministic, shared mechanism


if __name__ == "__main__":
    main()
