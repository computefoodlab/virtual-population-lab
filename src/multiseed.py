"""
Repeat the full evaluation across multiple random seeds and report every headline
metric as mean +/- standard deviation.

For each (dataset, seed) it reruns exactly the main-pipeline generation
(`src.main.generate_population`, so there is no metric-code drift) and recomputes
fidelity (correlation distance, mean KS), uncertainty calibration (coverage@90,
calibration error), and downstream utility (TSTR accuracy). With --loro it also
reruns the same-model near/far transfer per seed.

Usage:
  python -m src.multiseed                                  # 5 datasets, seeds 41-45
  python -m src.multiseed biofood_safou_region --seeds 41 42
  python -m src.multiseed --loro --loro-seeds 41 42 43

Writes to results/_summary/:
  multiseed_metrics.csv   — long form: dataset, engine, metric, mean, std, n, values
  multiseed_summary.md    — wide human table, every cell "mean +/- std"
  table1_multiseed.tex    — main results table with mean +/- std (best mean bold)
  multiseed_nearfar.csv   — near/far mean +/- std per dataset  (only with --loro)
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# src.main loads a config at import time from sys.argv[1]; neutralize argv over
# the import so it just loads the default, then restore it for our own parsing.
_argv = sys.argv
sys.argv = [_argv[0]]
from src.main import generate_population, ENGINE_MODULES  # noqa: E402
sys.argv = _argv

from src.config_loader import load_config  # noqa: E402
from src.data_loading import load_data  # noqa: E402
from src.evaluation.coverage import coverage_metrics  # noqa: E402
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table  # noqa: E402
from src.evaluation.tstr import run_tstr  # noqa: E402
from src.evaluation import loro as loro_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")

OUT = "results/_summary"

DEFAULT_CONFIGS = ["apple_quality", "banana_quality", "biofood_date_region",
                   "biofood_safou_region", "mango_composition"]
DATASET_LABEL = {
    "apple_quality": "Apple (quality)",
    "banana_quality": "Banana (quality)",
    "biofood_date_region": "Date (nutrients)",
    "biofood_safou_region": "Safou (nutrients)",
    "mango_composition": "Mango (Vit C)",
}
ENGINES = ["physics_mc", "regression", "vae", "hybrid_vae"]
ENGINE_NAME = {"physics_mc": "Physics-MC", "regression": "Regression",
               "vae": "VAE", "hybrid_vae": "Physics-VAE"}
# metric key -> (display, "lower is better"?)
METRICS = {
    "corr_dist": ("Corr. dist.", True),
    "mean_ks": ("Mean KS", True),
    "calib_err": ("Calib. err.", True),
    "coverage_at_90": ("Coverage@90", None),   # target is nominal 0.90, not min/max
    "tstr_acc": ("TSTR acc.", False),
}


def _one_seed(config, seed):
    """Run the generation + evaluation once at `seed`; return a list of
    {engine, metric, value} records (plus the real TSTR ceiling)."""
    config.RANDOM_SEED = seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = load_data(config)
    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed)
    scaler = None if config.IS_PRE_SCALED else StandardScaler().fit(train_df[config.FEATURES])

    torch.manual_seed(seed)
    generated = generate_population(train_df, config, scaler, quiet=True)

    ks = marginal_ks_table(test_df, generated, config.FEATURES)
    cov = coverage_metrics(generated, test_df, config.FEATURES).set_index("method")

    recs = []
    for e in ENGINES:
        if e not in generated:
            continue
        recs.append({"engine": e, "metric": "corr_dist",
                     "value": correlation_euclidean_dist(test_df, generated[e], config.FEATURES)})
        recs.append({"engine": e, "metric": "mean_ks",
                     "value": float(ks.loc[ks.method == e, "ks_stat"].mean())})
        recs.append({"engine": e, "metric": "coverage_at_90", "value": float(cov.loc[e, "coverage_at_90"])})
        recs.append({"engine": e, "metric": "calib_err", "value": float(cov.loc[e, "calibration_error"])})

    # TSTR — stratified conditional generation, exactly as the main pipeline does it.
    if getattr(config, "RUN_TSTR", False) and config.LABEL_COLUMN and config.LABEL_COLUMN in df.columns:
        classes = sorted(train_df[config.LABEL_COLUMN].dropna().unique())
        if len(classes) >= 2:
            labeled = {k: [] for k in ENGINE_MODULES}
            for cls in classes:
                cls_df = train_df[train_df[config.LABEL_COLUMN] == cls]
                torch.manual_seed(seed)
                pop = generate_population(cls_df, config, scaler, n_samples=len(cls_df), quiet=True)
                for k, g in pop.items():
                    g = g.copy()
                    g[config.LABEL_COLUMN] = cls
                    labeled[k].append(g)
            labeled = {k: pd.concat(v, ignore_index=True) for k, v in labeled.items() if v}
            tstr = run_tstr(labeled, train_df, test_df, config.FEATURES, config.LABEL_COLUMN, seed=seed)
            tstr = tstr.set_index("method")
            for e in ENGINES:
                if e in tstr.index:
                    recs.append({"engine": e, "metric": "tstr_acc", "value": float(tstr.loc[e, "accuracy"])})
            if "real" in tstr.index:
                recs.append({"engine": "real", "metric": "tstr_acc", "value": float(tstr.loc["real", "accuracy"])})
    return recs


def _agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def run_metrics(configs, seeds):
    long_rows = []
    for cfg in configs:
        config = load_config(cfg)
        log.info("=== %s : %d seeds %s ===", cfg, len(seeds), seeds)
        acc = {}  # (engine, metric) -> [values]
        for seed in seeds:
            log.info("  [%s] seed %d ...", cfg, seed)
            for r in _one_seed(config, seed):
                acc.setdefault((r["engine"], r["metric"]), []).append(r["value"])
        for (engine, metric), vals in acc.items():
            a = _agg(vals)
            long_rows.append({"dataset": DATASET_LABEL.get(cfg, cfg), "config": cfg,
                              "engine": engine, "metric": metric,
                              "mean": round(a["mean"], 4), "std": round(a["std"], 4), "n": a["n"],
                              "values": ";".join(f"{x:.4f}" for x in vals)})
    return pd.DataFrame(long_rows)


def run_nearfar(configs, seeds):
    rows = []
    for cfg in configs:
        config = load_config(cfg)
        group = getattr(config, "LORO_GROUP", "")
        if not group:
            continue
        log.info("=== near/far %s (%s) : %d seeds ===", cfg, group, len(seeds))
        df = load_data(config)
        acc = {}
        for seed in seeds:
            log.info("  [near/far %s] seed %d ...", cfg, seed)
            loro_mod.SEED = seed  # varies the region split AND the ensemble init (seed+100+k)
            np.random.seed(seed)
            torch.manual_seed(seed)
            res = loro_mod.leave_one_group_out(df, config.FEATURES, group)
            if res.empty:
                continue
            m = res[res["group"] == "MEAN"]
            if m.empty:
                continue
            m = m.iloc[0]
            for col in ("disagreement_far", "disagreement_near", "coverage_far",
                        "coverage_near", "coverage_far_conf", "coverage_near_conf",
                        "corr_far", "corr_near"):
                acc.setdefault(col, []).append(float(m[col]))
        if not acc:
            continue
        row = {"dataset": DATASET_LABEL.get(cfg, cfg), "config": cfg, "n_seeds": len(acc["disagreement_far"])}
        for col, vals in acc.items():
            a = _agg(vals)
            row[f"{col}_mean"] = round(a["mean"], 4)
            row[f"{col}_std"] = round(a["std"], 4)
        df_far, df_near = acc["disagreement_far"], acc["disagreement_near"]
        widen = [100 * (f / n - 1) for f, n in zip(df_far, df_near)]
        row["widening_pct_mean"] = round(float(np.mean(widen)))
        row["widening_pct_std"] = round(float(np.std(widen, ddof=1)) if len(widen) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------- output formatting -------------------------

def _cell(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def write_summary_md(long_df, seeds, path):
    piv = long_df.pivot_table(index=["dataset", "engine"], columns="metric",
                              values=["mean", "std"], aggfunc="first")
    lines = [f"# Multi-seed results (mean ± std over {len(seeds)} seeds: {seeds})", "",
             "Lower is better for correlation distance, mean KS, and calibration error; "
             "higher for TSTR accuracy; coverage@90 targets the nominal 0.90.", ""]
    cols = [m for m in METRICS if ("mean", m) in piv.columns]
    header = "| Dataset | Engine | " + " | ".join(METRICS[m][0] for m in cols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(cols) + 2))
    last_ds = None
    for cfg in DEFAULT_CONFIGS:
        label = DATASET_LABEL[cfg]
        order = ENGINES + ["real"]
        sub = long_df[long_df["config"] == cfg]
        for engine in order:
            if not ((sub["engine"] == engine).any()):
                continue
            ds_cell = label if label != last_ds else ""
            last_ds = label
            name = "Real (ceiling)" if engine == "real" else ENGINE_NAME.get(engine, engine)
            cells = []
            for m in cols:
                r = sub[(sub.engine == engine) & (sub.metric == m)]
                cells.append(_cell(r["mean"].iloc[0], r["std"].iloc[0]) if not r.empty else "--")
            lines.append(f"| {ds_cell} | {name} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def write_table1_tex(long_df, seeds, path):
    def get(cfg, engine, metric):
        r = long_df[(long_df.config == cfg) & (long_df.engine == engine) & (long_df.metric == metric)]
        return (r["mean"].iloc[0], r["std"].iloc[0]) if not r.empty else (np.nan, np.nan)

    L = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Fidelity, uncertainty calibration, and downstream utility across the five "
        rf"datasets, reported as mean $\pm$ standard deviation over {len(seeds)} random seeds. "
        r"Lower is better for correlation distance, mean KS, and calibration error; higher for "
        r"TSTR accuracy. Best mean per column within a dataset is \textbf{bold}. TSTR ceiling = a "
        r"classifier trained on real data.}",
        r"\label{tab:main}",
        r"\begin{tabular}{ll ccccc}", r"\toprule",
        r"Dataset & Engine & Corr.\ dist.\ $\downarrow$ & Mean KS $\downarrow$ & "
        r"Calib.\ err.\ $\downarrow$ & Coverage@90 & TSTR acc.\ $\uparrow$ \\", r"\midrule",
    ]
    col_metrics = ["corr_dist", "mean_ks", "calib_err", "coverage_at_90", "tstr_acc"]
    for cfg in DEFAULT_CONFIGS:
        label = DATASET_LABEL[cfg]
        best = {}
        for m in col_metrics:
            lower = METRICS[m][1]
            means = {e: get(cfg, e, m)[0] for e in ENGINES}
            means = {e: v for e, v in means.items() if not np.isnan(v)}
            if not means or lower is None:
                best[m] = None
            else:
                best[m] = min(means, key=means.get) if lower else max(means, key=means.get)
        for i, e in enumerate(ENGINES):
            cells = []
            for m in col_metrics:
                mean, std = get(cfg, e, m)
                if np.isnan(mean):
                    cells.append("--"); continue
                s = f"{mean:.3f} $\\pm$ {std:.3f}"
                cells.append(f"\\textbf{{{s}}}" if best[m] == e else s)
            name = label if i == 0 else ""
            L.append(f"{name} & {ENGINE_NAME[e]} & " + " & ".join(cells) + r" \\")
        rmean, rstd = get(cfg, "real", "tstr_acc")
        if not np.isnan(rmean):
            L.append(r" & \textit{Real (ceiling)} & -- & -- & -- & -- & "
                     rf"\textit{{{rmean:.3f} $\pm$ {rstd:.3f}}} \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--loro", action="store_true", help="also rerun near/far transfer per seed")
    ap.add_argument("--loro-seeds", nargs="+", type=int, default=None,
                    help="seeds for near/far (defaults to --seeds; near/far is heavy)")
    args = ap.parse_args()
    configs = args.configs or DEFAULT_CONFIGS
    os.makedirs(OUT, exist_ok=True)

    long_df = run_metrics(configs, args.seeds)
    long_df.to_csv(f"{OUT}/multiseed_metrics.csv", index=False)
    write_summary_md(long_df, args.seeds, f"{OUT}/multiseed_summary.md")
    write_table1_tex(long_df, args.seeds, f"{OUT}/table1_multiseed.tex")
    log.info("wrote %s/multiseed_metrics.csv, multiseed_summary.md, table1_multiseed.tex", OUT)

    if args.loro:
        nf = run_nearfar(configs, args.loro_seeds or args.seeds)
        if not nf.empty:
            nf.to_csv(f"{OUT}/multiseed_nearfar.csv", index=False)
            log.info("wrote %s/multiseed_nearfar.csv", OUT)

    print("\n" + open(f"{OUT}/multiseed_summary.md").read())


if __name__ == "__main__":
    main()
