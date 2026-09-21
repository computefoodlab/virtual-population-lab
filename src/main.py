import inspect
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_loader import DEFAULT_CONFIG, load_config
from src.data_loading import load_data
from src.evaluation.coverage import HEADLINE_LEVEL, coverage_metrics, save_calibration_plot
from src.evaluation.evaluate import display_name, generalization_gap, metric_display_name, run_evaluation
from src.evaluation.tstr import run_tstr
from src.generators import (
    hybrid_vae_generator,
    mcmc_generator,
    physics_mc_generator,
    regression_generator,
    vae_generator,
)

# Engine key -> its generator module, in the order they appear everywhere
# (summary table, report, figures). The module is used for the report's
# methodology section (pulled from each generate() docstring).
ENGINE_MODULES = {
    "physics_mc": physics_mc_generator,
    "mcmc": mcmc_generator,
    "regression": regression_generator,
    "vae": vae_generator,
    "hybrid_vae": hybrid_vae_generator,
    # Optional uncalibrated PI-VAE ablation, present only when a config sets
    # VAE_REPORT_UNCALIBRATED.
    "hybrid_vae_raw": hybrid_vae_generator,
}


def generate_population(source_df, config, scaler, n_samples=None, quiet=False):
    """
    Generate one synthetic population per engine from `source_df`, returning
    {engine_key: DataFrame}. Used both for the main run (source = full train)
    and for TSTR (source = one label class at a time), so the two paths can't
    drift. `scaler` is fit once on the full train and shared, so per-class
    generation stays in the same feature space; None when the data is already
    scaled. `quiet` silences per-engine/epoch logging for the many-call TSTR
    path.
    """
    n_samples = config.N_SAMPLES if n_samples is None else n_samples
    fx = config.FEATURES
    vae_input = source_df[fx].values if scaler is None else scaler.transform(source_df[fx])

    gen_log = logging.getLogger("vp-lab")
    prev_level = gen_log.level
    if quiet:
        gen_log.setLevel(logging.WARNING)
    # FIXED_EDGES lets a config impose a mechanistic/literature slope on specific
    # causal-graph edges (given in source units) instead of fitting it from data,
    # e.g. grape Glucose->Fructose = 1.0 (invertase 1:1 stoichiometry). All other
    # edges stay data-fitted. Only affects the PI-VAE physics term.
    fixed_edges = getattr(config, "FIXED_EDGES", None)
    edge_constants = None
    if fixed_edges:
        fixed_slopes = {(e["parent"], e["child"]): e["slope"] for e in fixed_edges}
        feature_scale = scaler.scale_ if scaler is not None else None
        edge_constants = hybrid_vae_generator.build_edge_constants(
            vae_input, fx, config.CAUSAL_GRAPH, fixed_slopes, feature_scale=feature_scale,
        )
    # calibrate_marginals is passed per-call below so the uncalibrated ablation
    # can toggle only that one flag.
    pi_vae_kwargs = dict(
        latent_dim=config.LATENT_DIM, epochs=config.VAE_EPOCHS, beta=config.VAE_BETA,
        n_samples=n_samples, scaler=scaler,
        use_minibatch=config.VAE_USE_MINIBATCH, batch_size=config.VAE_BATCH_SIZE,
        cov_weight=config.VAE_COV_WEIGHT, physics_weight=config.VAE_PHYSICS_WEIGHT,
        marginal_weight=config.VAE_MARGINAL_WEIGHT, prior_type=config.VAE_PRIOR_TYPE,
        constrain_generated=config.VAE_CONSTRAIN_GENERATED,
        patience=config.VAE_PATIENCE, hidden_dim=config.VAE_HIDDEN_DIM,
        dropout=config.VAE_DROPOUT, free_bits=config.VAE_FREE_BITS,
        edge_constants=edge_constants,
        constraints=getattr(config, "CONSTRAINTS", None),
        constraint_weight=getattr(config, "CONSTRAINT_WEIGHT", 0.0),
        # Raw source-unit training values: the marginal-calibration reference, so
        # the empirical inverse-CDF maps onto the exact observed values (exact
        # zeros / discrete levels preserved), which inverse-transforming a scaled
        # copy would perturb.
        calib_reference=source_df[fx].values,
    )
    # The uncalibrated PI-VAE is reported alongside the calibrated one only in
    # the main (verbose) run, and only when calibration is on.
    report_raw = (
        config.VAE_CALIBRATE_MARGINALS
        and getattr(config, "VAE_REPORT_UNCALIBRATED", False)
        and not quiet
    )
    # Seed every engine from the same RNG state, for both numpy (physics-MC,
    # regression) and torch (VAE, PI-VAE), so each engine's output is independent
    # of the order the engines happen to run in.
    def _seed():
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)

    try:
        _seed()
        out = {"physics_mc": physics_mc_generator.generate(
            source_df, fx, config.CAUSAL_GRAPH, config.ROOT_VARIABLES, n_samples=n_samples,
            constraints=getattr(config, "CONSTRAINTS", None),
        )}
        _seed()
        # Correlation-preserving MCMC prior-art baseline (Onwude 2022 / Hertog 2009).
        out["mcmc"] = mcmc_generator.generate(source_df, fx, n_samples=n_samples)
        _seed()
        out["regression"] = regression_generator.generate(source_df, fx, n_samples=n_samples)
        _seed()
        out["vae"] = vae_generator.generate(
            vae_input, fx,
            latent_dim=config.LATENT_DIM, epochs=config.VAE_EPOCHS, beta=config.VAE_BETA,
            n_samples=n_samples, scaler=scaler,
            use_minibatch=config.VAE_USE_MINIBATCH, batch_size=config.VAE_BATCH_SIZE,
            cov_weight=config.VAE_COV_WEIGHT, patience=config.VAE_PATIENCE,
            hidden_dim=config.VAE_HIDDEN_DIM, dropout=config.VAE_DROPOUT,
            free_bits=config.VAE_FREE_BITS,
        )
        _seed()
        out["hybrid_vae"] = hybrid_vae_generator.generate(
            vae_input, fx, config.CAUSAL_GRAPH,
            calibrate_marginals=config.VAE_CALIBRATE_MARGINALS, **pi_vae_kwargs,
        )
        if report_raw:
            # Same seed as the calibrated PI-VAE above, so the two rows differ
            # only by the calibration post-step.
            _seed()
            out["hybrid_vae_raw"] = hybrid_vae_generator.generate(
                vae_input, fx, config.CAUSAL_GRAPH,
                calibrate_marginals=False, **pi_vae_kwargs,
            )
    finally:
        gen_log.setLevel(prev_level)
    return out

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")

CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
config = load_config(CONFIG_NAME)

# Namespaced by config so results/reports from different datasets never
# clobber each other.
RESULTS_DIR = f"results/{CONFIG_NAME}"
REPORT_PATH = f"reports/{CONFIG_NAME}/REPORT.md"


def log_config():
    log.info("Active config: %s", CONFIG_NAME)
    for key in dir(config):
        if key.isupper():
            log.info("  %s = %r", key, getattr(config, key))


BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s):
    return len(ANSI_RE.sub("", s))


def print_summary(summary):
    """Prints `summary` as a bordered ASCII table, with the best (lowest) value
    per metric column marked with '*' and, in an actual terminal, colored."""
    metrics = [c for c in summary.columns if c != "method"]
    color = sys.stdout.isatty()

    headers = ["Engine"] + [metric_display_name(m) for m in metrics]
    rows = []
    for _, row in summary.iterrows():
        cells = [display_name(row["method"])]
        for metric in metrics:
            is_best = row[metric] == summary[metric].min()
            text = f"{'*' if is_best else ' '}{row[metric]:.4f}"
            if is_best and color:
                text = f"{BOLD_GREEN}{text}{RESET}"
            cells.append(text)
        rows.append(cells)

    widths = [
        max(_visible_len(h), *(_visible_len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def format_row(cells):
        padded = []
        for i, (c, w) in enumerate(zip(cells, widths)):
            fill = " " * (w - _visible_len(c))
            padded.append(c + fill if i == 0 else fill + c)  # text left, numbers right
        return "| " + " | ".join(padded) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    print(sep)
    print(format_row(headers))
    print(sep)
    for r in rows:
        print(format_row(r))
    print(sep)
    print("* = best (lowest) for that metric")


def write_report(df, train_df, test_df, summary, ks_table, gap_df, std_table, engines,
                 tstr_df=None, coverage_df=None):
    """
    Rewrites reports/REPORT.md from this run's actual data and outputs.
    `engines`: dict of {key: (module, generated_df)} — methodology text comes
    straight from each module's generate() docstring instead of being
    duplicated here, so it can't drift from the code.
    """
    metrics = [c for c in summary.columns if c != "method"]
    lines = [
        "# Virtual Population Lab — Report",
        "",
        f"_Auto-generated on {datetime.now():%Y-%m-%d %H:%M} from `{CONFIG_NAME}`. "
        f"Re-run `make run CONFIG={CONFIG_NAME}` to refresh._",
        "",
        "## Objective",
        "",
        "Compare modeling engines on how well each generates a synthetic population "
        "that preserves the real population's feature distributions, correlations, "
        "and uncertainty.",
        "",
        "## Data",
        "",
        f"- Source file: `{config.DATA_PATH}`",
    ]

    if config.DATA_SOURCE_URL:
        lines.append(f"- Source dataset: {config.DATA_SOURCE_URL}")

    lines += [
        f"- Features ({len(config.FEATURES)}): {', '.join(config.FEATURES)}",
        f"- Rows after cleaning: {len(df)} — split into {len(train_df)} train / {len(test_df)} test "
        f"(`TEST_SIZE={config.TEST_SIZE}`)",
        "",
        "## Methodology",
        "",
    ]

    seen_modules = set()
    for key, (module, _) in engines.items():
        if module in seen_modules:  # e.g. hybrid_vae_raw shares the PI-VAE module
            continue
        seen_modules.add(module)
        doc = inspect.getdoc(module.generate) or ""
        lines += [f"**{display_name(key)}** (`{module.__name__}.generate`)", "", doc, ""]

    lines += ["## Results", ""]
    lines.append("| Engine | " + " | ".join(metric_display_name(m) for m in metrics) + " |")
    lines.append("|" + "---|" * (len(metrics) + 1))
    for _, row in summary.iterrows():
        cells = [display_name(row["method"])]
        for metric in metrics:
            text = f"{row[metric]:.4f}"
            cells.append(f"**{text}**" if row[metric] == summary[metric].min() else text)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "(Lower is better for both metrics; bold = best.)", "", "## Findings", ""]

    for metric in metrics:
        best_row = summary.loc[summary[metric].idxmin()]
        lines.append(f"- Best **{metric_display_name(metric)}**: {display_name(best_row['method'])} ({best_row[metric]:.4f})")

    lines += [
        "",
        "Per-feature marginal fit (two-sample KS test, real vs. generated; "
        "lower ks_stat / higher p_value = closer):",
        "",
    ]
    for key in engines:
        sub = ks_table[ks_table.method == key]
        significant = int((sub.p_value < 0.05).sum())
        lines.append(f"- **{display_name(key)}**: {significant}/{len(sub)} features "
                      "statistically distinguishable from real (p < 0.05)")

    ks_pivot = ks_table.pivot(index="feature", columns="method", values="ks_stat")
    ks_pivot = ks_pivot.reindex(index=config.FEATURES, columns=list(engines))
    lines += [
        "",
        "## Per-Feature KS Statistic",
        "",
        "Lower = closer to real; bold = best per feature.",
        "",
    ]
    lines.append("| Feature | " + " | ".join(display_name(k) for k in engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))
    for feature in config.FEATURES:
        row = ks_pivot.loc[feature]
        best_key = row.idxmin()
        cells = [f"**{row[k]:.4f}**" if k == best_key else f"{row[k]:.4f}" for k in engines]
        lines.append(f"| {feature} | " + " | ".join(cells) + " |")

    std_pivot = std_table.pivot(index="feature", columns="method", values="generated_std")
    std_pivot = std_pivot.reindex(index=config.FEATURES, columns=list(engines))
    ratio_pivot = std_table.pivot(index="feature", columns="method", values="ratio_to_real")
    ratio_pivot = ratio_pivot.reindex(index=config.FEATURES, columns=list(engines))
    real_std = std_table.drop_duplicates("feature").set_index("feature")["real_std"]

    lines += [
        "",
        "## Feature Spread Comparison",
        "",
        f"**{metric_display_name('correlation_euclidean_dist')}** is scale-invariant — Pearson correlation "
        "ignores absolute spread, so a method can score well on it while its population is uniformly "
        "shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as "
        "a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is "
        "narrower than real).",
        "",
        "| Feature | Real Std | " + " | ".join(display_name(k) for k in engines) + " |",
        "|---|---|" + "---|" * len(engines),
    ]
    for feature in config.FEATURES:
        cells = [
            f"{std_pivot.loc[feature, k]:.3f} ({ratio_pivot.loc[feature, k]:.2f}x)"
            for k in engines
        ]
        lines.append(f"| {feature} | {real_std[feature]:.3f} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Generalization Check",
        "",
        "Correlation distance for each engine's synthetic data against the train split it was fit on "
        "vs. the held-out test split it's actually scored on. A small gap is evidence results generalize "
        "rather than matching training-set noise a real (unseen) population wouldn't share.",
        "",
        "| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |",
        "|---|---|---|---|",
    ]
    for _, row in gap_df.iterrows():
        lines.append(
            f"| {display_name(row['method'])} | {row['train_correlation_dist']:.4f} | "
            f"{row['test_correlation_dist']:.4f} | {row['gap']:.4f} |"
        )

    if tstr_df is not None:
        lines += [
            "",
            "## Downstream Utility (TSTR)",
            "",
            f"Train-on-Synthetic, Test-on-Real for the `{config.LABEL_COLUMN}` label: a "
            "RandomForest is trained on each engine's synthetic population (labels produced by "
            "generating each class separately) and scored on the real held-out test set. "
            "**Real (TRTR)** — a classifier trained on real data — is the ceiling; the closer an "
            "engine gets to it, the more genuinely useful its synthetic population is. This "
            "rewards preserving the feature-label joint structure, not just the marginals.",
            "",
            "| Trained on | Accuracy | ROC-AUC |",
            "|---|---|---|",
        ]
        for _, row in tstr_df.iterrows():
            label = "**Real (TRTR ceiling)**" if row["method"] == "real" else display_name(row["method"])
            auc = "n/a" if pd.isna(row["roc_auc"]) else f"{row['roc_auc']:.4f}"
            lines.append(f"| {label} | {row['accuracy']:.4f} | {auc} |")
        lines.append("")
        lines.append("(Higher is better; closer to the Real ceiling = more useful synthetic data.)")

    if coverage_df is not None:
        lines += [
            "",
            "## Uncertainty Calibration (Coverage)",
            "",
            f"Tests the *calibrated uncertainty* claim directly. For each feature, the central "
            f"{int(HEADLINE_LEVEL * 100)}% interval of each engine's generated population is formed, "
            "and we measure the fraction of real held-out values that fall inside it (averaged over "
            f"features). Well-calibrated ⇒ coverage ≈ the nominal {HEADLINE_LEVEL:.2f}; **below** = "
            "over-confident (intervals too narrow), **above** = intervals too wide. `calibration_error` "
            "is the mean absolute gap between empirical and nominal coverage across interval levels "
            "from 0.10 to 0.95 (lower = better calibrated across the whole range).",
            "",
            f"| Engine | Coverage @ {int(HEADLINE_LEVEL * 100)}% (nominal {HEADLINE_LEVEL:.2f}) | Calibration Error |",
            "|---|---|---|",
        ]
        best_cal = coverage_df["calibration_error"].min()
        for _, row in coverage_df.iterrows():
            cal = f"{row['calibration_error']:.4f}"
            cal = f"**{cal}**" if row["calibration_error"] == best_cal else cal
            lines.append(f"| {display_name(row['method'])} | {row['coverage_at_90']:.3f} | {cal} |")
        lines += [
            "",
            "(Coverage closest to nominal and lowest calibration error = best-calibrated uncertainty.)",
            "",
            f"![Uncertainty calibration reliability curve]"
            f"(../../{RESULTS_DIR}/figures/marginals/coverage_calibration.png)",
        ]

    # REPORT.md lives at reports/<config>/REPORT.md, two directories deep,
    # so figures under results/<config>/... need ../../ to get back to root.
    rel_results = f"../../{RESULTS_DIR}"

    # Near/far transfer — pulled from a prior `make loro` run if one exists (LORO
    # runs separately and far heavier, so its results may or may not be present).
    loro_path = f"{RESULTS_DIR}/loro.csv"
    if getattr(config, "LORO_GROUP", "") and os.path.exists(loro_path):
        lo = pd.read_csv(loro_path)
        if not lo.empty:
            has_conf = "coverage_far_conf" in lo.columns
            lines += [
                "",
                "## Support-Aware Uncertainty (Near/Far Transfer)",
                "",
                f"Same-model transfer over `{config.LORO_GROUP}`: one conditional-VAE ensemble is "
                "trained per region, then queried for the **held-out same region (NEAR)** vs a "
                "**different region (FAR)**. A model that 'knows what it doesn't know' has ensemble "
                "**disagreement** that widens for the unseen region (FAR > NEAR) while fidelity and "
                "coverage degrade. `coverage-conformal` recalibrates each interval's width on the "
                "NEAR held-out reals (target 0.90) and transfers that width off-support — it fixes "
                "NEAR coverage and partially closes the FAR gap (the residual is genuine "
                "distribution shift). (From the latest `make loro` run.)",
                "",
                "| Train region | disagreement FAR / NEAR | coverage FAR / NEAR |"
                + (" coverage-conformal FAR / NEAR |" if has_conf else "") + " corr FAR / NEAR |",
                "|---|---|---|" + ("---|" if has_conf else "") + "---|",
            ]
            for _, r in lo.iterrows():
                g = "**MEAN**" if r["group"] == "MEAN" else str(r["group"])
                conf_cell = (f" {r['coverage_far_conf']:.3f} / {r['coverage_near_conf']:.3f} |"
                             if has_conf else "")
                lines.append(
                    f"| {g} | {r['disagreement_far']:.4f} / {r['disagreement_near']:.4f} | "
                    f"{r['coverage_far']:.3f} / {r['coverage_near']:.3f} |"
                    + conf_cell +
                    f" {r['corr_far']:.3f} / {r['corr_near']:.3f} |"
                )
            mrow = lo[lo["group"] == "MEAN"]
            if not mrow.empty:
                mr = mrow.iloc[0]
                pct = 100 * (mr["disagreement_far"] / mr["disagreement_near"] - 1)
                verdict = "widens off-support" if pct > 5 else ("narrows" if pct < -5 else "no clear change")
                lines += [
                    "",
                    f"Mean ensemble disagreement is **{pct:+.0f}%** for unseen vs held-out same regions — {verdict}.",
                    "",
                    f"![Near/far transfer]({rel_results}/figures/loro/loro_{config.LORO_GROUP}.png)",
                ]

    lines += ["", "## Figures", "", "### Joint variability (correlation)", "",
              "**Correlation matrices**", "",
              f"![Real correlation matrix]({rel_results}/figures/correlation/correlation_real.png)"]
    for key in engines:
        lines.append(f"![{display_name(key)} correlation matrix]"
                      f"({rel_results}/figures/correlation/correlation_{key}.png)")
    lines += [
        "",
        "**PCA projection — real vs. each method, plus all combined**",
        "",
        "![PCA projection of real vs generated populations]"
        f"({rel_results}/figures/correlation/pca_real_vs_generated.png)",
        "",
        "**PCA projection — each population shown individually, no overlay**",
        "",
        f"![PCA projection of each population individually]({rel_results}/figures/correlation/pca_individual.png)",
        "",
        "### Marginal distributions (KS)",
        "",
        "**KS statistic by feature and method**",
        "",
        f"![KS statistic heatmap]({rel_results}/figures/marginals/ks_heatmap.png)",
        "",
        "**Empirical CDFs — real vs. generated, per feature**",
        "",
        f"![ECDF overlay per feature]({rel_results}/figures/marginals/ecdf_overlay.png)",
        "",
        "**P-value by feature and method**",
        "",
        f"![P-value heatmap]({rel_results}/figures/marginals/pvalue_heatmap.png)",
        "",
        "**Volcano plot — effect size vs. significance**",
        "",
        f"![Volcano plot]({rel_results}/figures/marginals/volcano_plot.png)",
        "",
        "---",
        "",
        f"Raw data: `{RESULTS_DIR}/summary_metrics.csv`, `{RESULTS_DIR}/ks_marginals.csv`, "
        f"`{RESULTS_DIR}/generalization_gap.csv`, `{RESULTS_DIR}/synthetic_data/`.",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    log_config()

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    log.info("Loading data from %s", config.DATA_PATH)
    df = load_data(config)
    log.info("Loaded %d clean rows", len(df))

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    log.info("Split into %d train / %d test rows", len(train_df), len(test_df))

    if config.IS_PRE_SCALED:
        vae_scaler = None
    else:
        vae_scaler = StandardScaler().fit(train_df[config.FEATURES])

    log.info("Generating synthetic populations (physics-MC, regression, VAE, PI-VAE)...")
    generated = generate_population(train_df, config, vae_scaler)
    for key, gen_df in generated.items():
        log.info("  %-30s generated %d rows", display_name(key), len(gen_df))

    engines = {key: (ENGINE_MODULES[key], gen) for key, gen in generated.items()}

    synth_dir = f"{RESULTS_DIR}/synthetic_data"
    os.makedirs(synth_dir, exist_ok=True)
    # Clear stale engine CSVs first — the engine set can change between runs
    # (e.g. toggling VAE_REPORT_UNCALIBRATED adds/removes hybrid_vae_raw), and a
    # leftover file would misrepresent what this run actually produced.
    for stale in os.listdir(synth_dir):
        if stale.endswith(".csv"):
            os.remove(os.path.join(synth_dir, stale))
    for name, gen_df in generated.items():
        gen_df.to_csv(f"{synth_dir}/{name}.csv", index=False)
    log.info("Saved synthetic data to %s/", synth_dir)

    log.info("Running evaluation...")
    summary, ks_table, std_table = run_evaluation(
        test_df, generated, config.FEATURES,
        figures_dir=f"{RESULTS_DIR}/figures",
        results_dir=RESULTS_DIR,
    )
    log.info("Saved figures and metrics to %s/", RESULTS_DIR)

    print_summary(summary)

    log.info(
        "Checking generalization — correlation distance against the train split each engine "
        "was fit on vs. the held-out test split it's actually scored on (small gap = generalizes; "
        "train much better than test = overfitting signal)..."
    )
    gap_df = generalization_gap(train_df, test_df, generated, config.FEATURES)
    gap_df.to_csv(f"{RESULTS_DIR}/generalization_gap.csv", index=False)
    for _, row in gap_df.iterrows():
        log.info(
            "  %-30s corr_dist: train=%.4f  test=%.4f  gap=%.4f",
            display_name(row["method"]), row["train_correlation_dist"],
            row["test_correlation_dist"], row["gap"],
        )

    log.info(
        "Checking feature spread — each engine's per-feature std dev as a ratio of real's, "
        "averaged across all features (1.00 = matches real's spread exactly; well below 1.00 = "
        "generated population is narrower than real; this is independent of %s, "
        "which is scale-invariant and can't see this on its own)...",
        metric_display_name("correlation_euclidean_dist"),
    )
    mean_std_ratio = std_table.groupby("method")["ratio_to_real"].mean()
    for name in engines:
        log.info("  %-30s mean std ratio=%.3f", display_name(name), mean_std_ratio[name])

    log.info(
        "Checking uncertainty calibration — fraction of real held-out values falling inside each "
        "engine's generated central interval, vs the nominal rate (coverage near nominal = "
        "calibrated; below = over-confident/too narrow; above = too wide)..."
    )
    coverage_df = coverage_metrics(generated, test_df, config.FEATURES)
    coverage_df.to_csv(f"{RESULTS_DIR}/coverage.csv", index=False)
    save_calibration_plot(
        generated, test_df, config.FEATURES,
        os.path.join(f"{RESULTS_DIR}/figures/marginals", "coverage_calibration.png"),
    )
    for _, row in coverage_df.iterrows():
        log.info(
            "  %-30s coverage@%d%%=%.3f (nominal %.2f)  calibration_error=%.4f",
            display_name(row["method"]), int(HEADLINE_LEVEL * 100),
            row["coverage_at_90"], HEADLINE_LEVEL, row["calibration_error"],
        )

    tstr_df = run_tstr_step(df, train_df, test_df, vae_scaler)

    write_report(df, train_df, test_df, summary, ks_table, gap_df, std_table, engines,
                 tstr_df, coverage_df)
    log.info("Wrote %s", REPORT_PATH)


def run_tstr_step(df, train_df, test_df, vae_scaler):
    """
    Downstream-utility check: build a labeled synthetic training set per engine
    by stratified conditional generation (each class generated from its own
    real rows), train a classifier on it, and score it against real held-out
    rows (see src.evaluation.tstr). Runs only when config.RUN_TSTR is set and
    the dataset has a usable categorical label with >=2 classes; returns the
    results DataFrame, or None when skipped.
    """
    if not getattr(config, "RUN_TSTR", False) or not config.LABEL_COLUMN:
        return None
    if config.LABEL_COLUMN not in df.columns:
        log.info("TSTR skipped — label column %r not in the data.", config.LABEL_COLUMN)
        return None

    classes = sorted(train_df[config.LABEL_COLUMN].dropna().unique())
    if len(classes) < 2:
        log.info("TSTR skipped — label %r has fewer than 2 classes.", config.LABEL_COLUMN)
        return None

    log.info(
        "Running TSTR (train-on-synthetic, test-on-real) on label %r (%d classes) — "
        "generating each class separately per engine...",
        config.LABEL_COLUMN, len(classes),
    )

    labeled = {key: [] for key in ENGINE_MODULES}
    for cls in classes:
        class_df = train_df[train_df[config.LABEL_COLUMN] == cls]
        class_pop = generate_population(class_df, config, vae_scaler, n_samples=len(class_df), quiet=True)
        for key, gen_df in class_pop.items():
            tagged = gen_df.copy()
            tagged[config.LABEL_COLUMN] = cls
            labeled[key].append(tagged)
    # Skip engines that produced no frames (e.g. the optional hybrid_vae_raw
    # ablation, which generate_population only emits in the main verbose run).
    labeled = {key: pd.concat(frames, ignore_index=True)
               for key, frames in labeled.items() if frames}

    tstr_df = run_tstr(labeled, train_df, test_df, config.FEATURES, config.LABEL_COLUMN,
                       seed=config.RANDOM_SEED)
    tstr_df.to_csv(f"{RESULTS_DIR}/tstr.csv", index=False)

    for _, row in tstr_df.iterrows():
        label = "Real (TRTR ceiling)" if row["method"] == "real" else display_name(row["method"])
        auc = "  n/a" if pd.isna(row["roc_auc"]) else f"{row['roc_auc']:.3f}"
        log.info("  %-30s accuracy=%.3f  roc_auc=%s", label, row["accuracy"], auc)

    return tstr_df


if __name__ == "__main__":
    main()
