"""
Physics-VAE loss-term ablation. Starting from the full model, each term is
switched off one at a time and the fidelity and calibration metrics are
recomputed across multiple seeds (mean +/- std), isolating what each term
contributes.

Variants (one change from Full):
  Full            all terms on, with copula calibration
  - physics       lambda_phys = 0   (no causal-edge constraint)
  - covariance    lambda_cov  = 0   (no covariance matching)
  - marginal      lambda_marg = 0   (no 1-D Wasserstein marginal term)
  - calibration   copula post-step off

Metrics vs the real held-out test split: correlation distance, mean KS,
calibration error, coverage@90. (TSTR is omitted here — it needs per-class
re-generation and the distributional/calibration metrics already isolate each
term's role; the main results table carries TSTR.)

Usage:
  python -m src.ablation                                   # 5 datasets, seeds 41-43
  python -m src.ablation biofood_safou_region --seeds 41 42 43

Writes results/_summary/: ablation.csv, ablation.md, ablation.tex
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config_loader import load_config
from src.data_loading import load_data
from src.generators import hybrid_vae_generator as hv
from src.evaluation.coverage import coverage_metrics
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")
OUT = "results/_summary"

DATASETS = {
    "apple_quality": "Apple",
    "banana_quality": "Banana",
    "biofood_date_region": "Date",
    "biofood_safou_region": "Safou",
    "mango_composition": "Mango",
}
# key, label, override dict applied on top of the config's full settings
VARIANTS = [
    ("full", "Full", {}),
    ("no_phys", "$-$ physics", {"physics_weight": 0.0}),
    ("no_cov", "$-$ covariance", {"cov_weight": 0.0}),
    ("no_marg", "$-$ marginal", {"marginal_weight": 0.0}),
    ("no_calib", "$-$ calibration", {"calibrate_marginals": False}),
]
METRICS = [("corr_dist", "Corr. dist."), ("mean_ks", "Mean KS"),
           ("calib_err", "Calib. err."), ("coverage_at_90", "Coverage@90")]


def _base_kwargs(config):
    return dict(
        latent_dim=config.LATENT_DIM, epochs=config.VAE_EPOCHS, beta=config.VAE_BETA,
        n_samples=config.N_SAMPLES, use_minibatch=config.VAE_USE_MINIBATCH,
        batch_size=config.VAE_BATCH_SIZE, cov_weight=config.VAE_COV_WEIGHT,
        physics_weight=config.VAE_PHYSICS_WEIGHT, marginal_weight=config.VAE_MARGINAL_WEIGHT,
        prior_type=config.VAE_PRIOR_TYPE, constrain_generated=config.VAE_CONSTRAIN_GENERATED,
        patience=config.VAE_PATIENCE, hidden_dim=config.VAE_HIDDEN_DIM,
        dropout=config.VAE_DROPOUT, free_bits=config.VAE_FREE_BITS,
        calibrate_marginals=config.VAE_CALIBRATE_MARGINALS,
    )


def _metrics(gen_df, test_df, features):
    ks = marginal_ks_table(test_df, {"g": gen_df}, features)["ks_stat"].mean()
    cov = coverage_metrics({"g": gen_df}, test_df, features).iloc[0]
    return {
        "corr_dist": correlation_euclidean_dist(test_df, gen_df, features),
        "mean_ks": float(ks),
        "calib_err": float(cov["calibration_error"]),
        "coverage_at_90": float(cov["coverage_at_90"]),
    }


def run(configs, seeds):
    long_rows = []
    for cfg_name in configs:
        config = load_config(cfg_name)
        empty_graph = len(config.CAUSAL_GRAPH) == 0
        log.info("=== ablation %s : %d seeds ===", cfg_name, len(seeds))
        acc = {}  # (variant, metric) -> [values]
        for seed in seeds:
            np.random.seed(seed)
            df = load_data(config)
            train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed)
            scaler = None if config.IS_PRE_SCALED else StandardScaler().fit(train_df[config.FEATURES])
            x = train_df[config.FEATURES].values if scaler is None else scaler.transform(train_df[config.FEATURES])
            for key, _, override in VARIANTS:
                # Ablating physics on an empty graph is a no-op; skip (reported as n/a).
                if key == "no_phys" and empty_graph:
                    continue
                kwargs = _base_kwargs(config)
                kwargs.update(override)
                torch.manual_seed(seed)
                gen = hv.generate(x, config.FEATURES, config.CAUSAL_GRAPH, scaler=scaler, **kwargs)
                m = _metrics(gen, test_df, config.FEATURES)
                for mk, mv in m.items():
                    acc.setdefault((key, mk), []).append(mv)
            log.info("  [%s] seed %d done", cfg_name, seed)
        for (variant, metric), vals in acc.items():
            v = np.asarray(vals)
            long_rows.append({
                "dataset": DATASETS.get(cfg_name, cfg_name), "config": cfg_name,
                "variant": variant, "metric": metric,
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                "n": len(v),
            })
    return pd.DataFrame(long_rows)


def _cell(df, cfg, variant, metric):
    r = df[(df.config == cfg) & (df.variant == variant) & (df.metric == metric)]
    if r.empty:
        return None
    return r["mean"].iloc[0], r["std"].iloc[0]


def write_md(df, seeds, path):
    L = [f"# Physics-VAE loss-term ablation (mean ± std over {len(seeds)} seeds: {seeds})", "",
         "Each variant changes ONE thing from Full. Lower is better for correlation distance, "
         "mean KS, and calibration error; coverage@90 targets the nominal 0.90. Δ = change vs Full "
         "(positive = worse fidelity / calibration when the term is removed).", "",
         "Notes: `− physics` is a no-op on Banana/Mango (empty causal graph). Safou's Full already "
         "uses λ_cov=0, so `− covariance` is a no-op there: covariance matching worsens Safou "
         "correlation (+0.08 to include it), because a 4×4 covariance estimated from ~28 rows is too "
         "noisy a target once the near-deterministic physics already pins the structure.", ""]
    for cfg, label in DATASETS.items():
        if not (df.config == cfg).any():
            continue
        L += [f"## {label}", "",
              "| Variant | " + " | ".join(m[1] for m in METRICS) + " |",
              "|" + "---|" * (len(METRICS) + 1)]
        full = {mk: _cell(df, cfg, "full", mk) for mk, _ in METRICS}
        for key, vlabel, _ in VARIANTS:
            cells = []
            for mk, _ in METRICS:
                c = _cell(df, cfg, key, mk)
                if c is None:
                    cells.append("n/a")
                    continue
                s = f"{c[0]:.3f} ± {c[1]:.3f}"
                if key != "full" and full[mk] is not None and mk != "coverage_at_90":
                    d = c[0] - full[mk][0]
                    s += f"  (Δ{d:+.3f})"
                cells.append(s)
            L.append(f"| {vlabel.replace('$-$','−')} | " + " | ".join(cells) + " |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def write_tex(df, seeds, path):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Physics-VAE loss-term ablation, mean $\pm$ std over "
         rf"{len(seeds)} random seeds. Each row removes one term from the Full model. "
         r"Lower is better for correlation distance, mean KS, and calibration error; coverage@90 "
         r"targets 0.90. `$-$ physics' is not applicable to Banana/Mango (empty causal graph). "
         r"Safou's Full already sets $\lambda_{\mathrm{cov}}{=}0$ (so `$-$ covariance' is a no-op "
         r"there): covariance matching \emph{worsens} correlation on Safou ($+0.08$ to include it) "
         r"because a $4{\times}4$ covariance estimated from ${\sim}28$ rows is too noisy a target "
         r"once the near-deterministic physics already pins the structure.}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{ll cccc}", r"\toprule",
         r"Dataset & Variant & Corr.\ dist.\ $\downarrow$ & Mean KS $\downarrow$ & "
         r"Calib.\ err.\ $\downarrow$ & Coverage@90 \\", r"\midrule"]
    for cfg, label in DATASETS.items():
        if not (df.config == cfg).any():
            continue
        first = True
        for key, vlabel, _ in VARIANTS:
            cells = []
            for mk, _ in METRICS:
                c = _cell(df, cfg, key, mk)
                cells.append("n/a" if c is None else f"{c[0]:.3f} $\\pm$ {c[1]:.3f}")
            ds = label if first else ""
            first = False
            L.append(f"{ds} & {vlabel} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    args = ap.parse_args()
    configs = args.configs or list(DATASETS)
    os.makedirs(OUT, exist_ok=True)

    df = run(configs, args.seeds)
    df.to_csv(f"{OUT}/ablation.csv", index=False)
    write_md(df, args.seeds, f"{OUT}/ablation.md")
    write_tex(df, args.seeds, f"{OUT}/ablation.tex")
    log.info("wrote %s/ablation.csv, ablation.md, ablation.tex", OUT)
    print("\n" + open(f"{OUT}/ablation.md").read())


if __name__ == "__main__":
    main()
