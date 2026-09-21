import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FIGURE_DPI = 400

# Okabe-Ito palette: colorblind-safe, real vs. up to 7 generated methods.
# Ordered for maximum contrast between the first few slots, which are the ones
# in view with the four shipped engines.
REAL_COLOR = "#333333"
GENERATED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

# Alpha needs to drop as more semi-transparent colors stack on top of each
# other, or overlaps alpha-blend into muddy intermediate colors. Applies
# only to the *generated* colors in the combined panel (real vs. all) —
# "Real" itself always uses SCATTER_ALPHA_PAIR so it looks the same everywhere.
SCATTER_ALPHA_PAIR = 0.5
SCATTER_ALPHA_COMBINED = 0.32
SCATTER_SIZE = 22

# Human-readable labels for known method keys, used in chart titles, legends,
# the printed summary, and CSV values/headers via _with_display_names(). File
# paths (e.g. correlation_physics_mc.png) keep the raw keys — those are
# stable identifiers other code paths still filter/build on.
DISPLAY_NAMES = {
    "physics_mc": "Structural Monte Carlo",
    "regression": "Regression",
    "vae": "Variational Autoencoder",
    "hybrid_vae": "Physics-Informed VAE",
    "hybrid_vae_raw": "Physics-Informed VAE (uncalibrated)",
}


def display_name(name):
    """Falls back to Title Case for any key not in DISPLAY_NAMES (e.g. a future engine)."""
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# Human-readable labels for metric/column names, used for display everywhere
# they appear — chart text, the printed table, and CSV headers.
METRIC_DISPLAY_NAMES = {
    "correlation_euclidean_dist": "Correlation Distance (Euclidean)",
    "mean_ks_stat": "Mean KS Statistic",
    "ks_stat": "KS Statistic",
    "p_value": "P-Value",
    "train_correlation_dist": "Correlation Dist. (vs. Train)",
    "test_correlation_dist": "Correlation Dist. (vs. Test)",
    "gap": "Train/Test Gap",
    "real_std": "Real Std Dev",
    "generated_std": "Generated Std Dev",
    "ratio_to_real": "Ratio to Real",
}


def metric_display_name(name):
    return METRIC_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _with_display_names(frame):
    """Returns a copy of `frame` with the method column and all headers
    rendered human-readable — for CSVs meant to be opened and read directly."""
    frame = frame.assign(method=frame["method"].map(display_name))
    return frame.rename(columns=lambda c: "Engine" if c == "method" else metric_display_name(c))


def save_correlation_heatmap(data, features, title, path):
    plt.figure(figsize=(8, 6))

    sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()


def save_pca_scatter(real_df, generated, features, path):
    """
    generated: dict of {method_name: dataframe}. One panel per method
    (real vs. that method alone, so its fit is visible without other
    methods' points cluttering the view) plus one combined panel (real vs.
    all methods overlaid), all in a single grid image with shared axis
    limits so panels are directly comparable at a glance.
    """
    scaler = StandardScaler().fit(real_df[features])
    pca = PCA(2).fit(scaler.transform(real_df[features]))
    real_proj = pca.transform(scaler.transform(real_df[features]))

    colors = dict(zip(generated, GENERATED_COLORS))
    projections = {name: pca.transform(scaler.transform(fake_df[features])) for name, fake_df in generated.items()}

    all_points = np.vstack([real_proj] + list(projections.values()))
    pad = 0.05 * (all_points.max() - all_points.min())
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    n_panels = len(generated) + 1  # one per method, plus one combined
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, (name, proj) in zip(axes, projections.items()):
        ax.scatter(real_proj[:, 0], real_proj[:, 1], s=SCATTER_SIZE, alpha=SCATTER_ALPHA_PAIR,
                   label="Real", color=REAL_COLOR)
        ax.scatter(proj[:, 0], proj[:, 1], s=SCATTER_SIZE, alpha=SCATTER_ALPHA_PAIR,
                   label=display_name(name), color=colors[name])
        ax.set_title(f"Real vs. {display_name(name)}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # "Real" keeps the same alpha as the individual panels; only the generated
    # colors drop to a lower alpha here, since they are the ones that stack.
    combined_ax = axes[len(projections)]
    combined_ax.scatter(real_proj[:, 0], real_proj[:, 1], s=SCATTER_SIZE, alpha=SCATTER_ALPHA_PAIR,
                         label="Real", color=REAL_COLOR)
    for name, proj in projections.items():
        combined_ax.scatter(proj[:, 0], proj[:, 1], s=SCATTER_SIZE, alpha=SCATTER_ALPHA_COMBINED,
                             label=display_name(name), color=colors[name])
    combined_ax.set_title("Real vs. All (Combined)")
    combined_ax.set_xlim(xlim)
    combined_ax.set_ylim(ylim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    explained = pca.explained_variance_ratio_
    for ax in axes[:n_panels]:
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")

    handles, labels = combined_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        f"Real vs. Generated Populations — PCA Projection of {len(features)} Features\n"
        "(components fit on real data only)",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def save_pca_individual_panels(real_df, generated, features, path):
    """
    One panel per population — real, then each generated method — each
    showing only that population's own cloud, no overlay with any other.
    Since each panel is a single color, there's no alpha-blending between
    colors to worry about at all. Same PCA basis and shared axis limits as
    save_pca_scatter, so shapes are directly comparable side by side.
    """
    scaler = StandardScaler().fit(real_df[features])
    pca = PCA(2).fit(scaler.transform(real_df[features]))
    real_proj = pca.transform(scaler.transform(real_df[features]))

    colors = dict(zip(generated, GENERATED_COLORS))
    projections = {name: pca.transform(scaler.transform(fake_df[features])) for name, fake_df in generated.items()}

    all_points = np.vstack([real_proj] + list(projections.values()))
    pad = 0.05 * (all_points.max() - all_points.min())
    xlim = (all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ylim = (all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    panels = [("Real", real_proj, REAL_COLOR)]
    panels += [(display_name(name), proj, colors[name]) for name, proj in projections.items()]

    ncols = int(np.ceil(np.sqrt(len(panels))))
    nrows = int(np.ceil(len(panels) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.flatten()

    explained = pca.explained_variance_ratio_
    for ax, (label, proj, color) in zip(axes, panels):
        ax.scatter(proj[:, 0], proj[:, 1], s=SCATTER_SIZE, alpha=SCATTER_ALPHA_PAIR, color=color)
        ax.set_title(label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"Individual Populations — PCA Projection of {len(features)} Features\n"
        "(components fit on real data only; each panel shown alone, no overlay)",
        y=0.95,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def save_ks_heatmap(ks_table, path):
    """
    Features x methods grid, colored by ks_stat. Sequential (not diverging)
    colormap since ks_stat is a magnitude — 0 is "identical," there's no
    meaningful sign — unlike the correlation heatmaps' coolwarm.
    """
    pivot = ks_table.pivot(index="feature", columns="method", values="ks_stat")
    pivot = pivot.rename(columns=display_name)

    plt.figure(figsize=(max(7, 2 * len(pivot.columns)), max(4, 0.6 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1)

    plt.title("KS Statistic by Feature and Method\n(lower = generated marginal closer to real)", fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def save_ecdf_overlay(real_df, generated, features, path):
    """
    One subplot per feature: real vs. each method's empirical CDF (a sorted-
    values staircase from 0 to 1). This is the literal picture behind
    ks_stat — the biggest vertical gap between two curves in a subplot IS
    that feature's ks_stat for that method.
    """
    ncols = min(3, len(features))
    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        real_sorted = np.sort(real_df[feature])
        ax.step(real_sorted, np.arange(1, len(real_sorted) + 1) / len(real_sorted),
                where="post", label="Real", color=REAL_COLOR, linewidth=2)

        for color, (name, fake_df) in zip(GENERATED_COLORS, generated.items()):
            fake_sorted = np.sort(fake_df[feature])
            ax.step(fake_sorted, np.arange(1, len(fake_sorted) + 1) / len(fake_sorted),
                    where="post", label=display_name(name), color=color, linewidth=1.5, alpha=0.85)

        ax.set_title(feature)
        ax.set_ylabel("Cumulative probability")

    for ax in axes[len(features):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Empirical CDFs — Real vs. Generated, per Feature", y=1.08)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def _fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def save_pvalue_heatmap(ks_table, path):
    """
    Features x methods grid, colored by -log10(p_value): raw p-values here
    span ~80 orders of magnitude (0.8 down to 1e-75), so a linear scale on
    p itself would just render as all-zero: -log10 makes it readable, and
    each cell is still annotated with the actual p-value. Taller bar/darker
    cell = more statistically significant difference from real.
    """
    pivot = ks_table.pivot(index="feature", columns="method", values="p_value")
    pivot = pivot.rename(columns=display_name)
    neg_log_p = -np.log10(pivot.clip(lower=1e-300))
    labels = pivot.map(_fmt_p)

    plt.figure(figsize=(max(7, 2 * len(pivot.columns)), max(4, 0.6 * len(pivot))))
    sns.heatmap(neg_log_p, annot=labels, fmt="", cmap="Reds")

    plt.title("P-Value by Feature and Method (-log10 scale; cell text = raw p)\n"
               "(darker = more statistically significant difference from real)", fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def save_volcano_plot(ks_table, path):
    """
    ks_stat (effect size) vs. -log10(p_value) (significance) — one point per
    (feature, method). Combines both KS numbers in a single view instead of two
    separate heatmaps. Points are not individually labeled (features x methods
    cluster too tightly for readable text) since the exact per-feature values are
    already in the KS and p-value heatmaps; this plot shows the overall
    per-method pattern.
    """
    plt.figure(figsize=(8, 6))

    methods = list(dict.fromkeys(ks_table["method"]))  # preserve first-seen order
    for color, method in zip(GENERATED_COLORS, methods):
        sub = ks_table[ks_table.method == method]
        neg_log_p = -np.log10(sub["p_value"].clip(lower=1e-300))
        plt.scatter(sub["ks_stat"], neg_log_p, label=display_name(method), color=color, s=70, alpha=0.85)

    plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1, label="p = 0.05")
    plt.xlabel("KS statistic (effect size)")
    plt.ylabel("-log10(p-value) (significance)")
    plt.legend()
    plt.title("Volcano Plot — KS Statistic vs. Significance, per Feature and Method")
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()


def correlation_euclidean_dist(real_df, fake_df, features):
    """
    Euclidean distance between the real and generated feature correlation
    matrices, each flattened to a vector first: sqrt(sum of squared
    per-cell differences). Not an average — comparable across methods on
    this dataset (same matrix size), but not across datasets with a
    different feature count.

    Any correlation cell that is undefined — a feature with zero variance in
    either population (a constant column) has NaN correlations — is treated as
    0 (no linear relationship). Without this, a single degenerate/collapsed
    feature (e.g. within a small held-out subgroup, or from a mode-collapsed
    generator) would make the whole distance NaN.
    """
    real_corr = np.nan_to_num(real_df[features].corr().values.flatten())
    fake_corr = np.nan_to_num(fake_df[features].corr().values.flatten())

    return euclidean(real_corr, fake_corr)


def generalization_gap(train_df, test_df, generated, features):
    """
    For each method, compares correlation_euclidean_dist against the split
    it was fit on (train_df) vs. the held-out split it's actually scored
    on (test_df). A small gap is evidence the fit generalizes rather than
    matching training-set noise a real (unseen) population wouldn't share —
    a large one, especially with train notably better than test, would be a
    genuine overfitting signal.
    """
    rows = []

    for name, fake_df in generated.items():
        train_dist = correlation_euclidean_dist(train_df, fake_df, features)
        test_dist = correlation_euclidean_dist(test_df, fake_df, features)
        rows.append({
            "method": name,
            "train_correlation_dist": train_dist,
            "test_correlation_dist": test_dist,
            "gap": abs(train_dist - test_dist),
        })

    return pd.DataFrame(rows)


def marginal_ks_table(real_df, generated, features):
    """
    Two-sample Kolmogorov-Smirnov statistic per feature per method: how well
    each generator's marginal distribution matches the real one (lower = closer).
    """
    rows = []

    for name, fake_df in generated.items():
        for feature in features:
            stat, p_value = ks_2samp(real_df[feature], fake_df[feature])
            rows.append({"method": name, "feature": feature, "ks_stat": stat, "p_value": p_value})

    return pd.DataFrame(rows)


def feature_std_comparison(real_df, generated, features):
    """
    Per-feature standard deviation, real vs. each method, plus each
    method's std as a fraction of real's. correlation_euclidean_dist is
    scale-invariant (Pearson correlation ignores absolute spread), so two
    methods can score identically on it while one's population is
    uniformly shrunk relative to real — this catches that directly, and
    explains cases where a method visually looks closer to real (matching
    spread) despite scoring worse on correlation distance, or vice versa.
    """
    real_std = real_df[features].std()
    rows = []

    for name, fake_df in generated.items():
        fake_std = fake_df[features].std()
        for feature in features:
            rows.append({
                "method": name,
                "feature": feature,
                "real_std": real_std[feature],
                "generated_std": fake_std[feature],
                "ratio_to_real": fake_std[feature] / real_std[feature],
            })

    return pd.DataFrame(rows)


def run_evaluation(real_df, generated, features, figures_dir, results_dir):
    """
    generated: dict of {method_name: dataframe}. Saves joint-variability
    figures (correlation heatmaps, PCA scatter) to figures_dir/correlation/
    and marginal-fit figures (KS heatmap, ECDF overlay) to
    figures_dir/marginals/, plus a per-feature KS table, a per-feature std
    comparison, and a per-method correlation_euclidean_dist summary to
    results_dir. Returns (summary, ks_table, std_table) as DataFrames.
    """
    correlation_dir = os.path.join(figures_dir, "correlation")
    marginals_dir = os.path.join(figures_dir, "marginals")
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(marginals_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    save_correlation_heatmap(
        real_df, features,
        f"Feature Correlation Matrix — Real Data (n={len(real_df)})",
        os.path.join(correlation_dir, "correlation_real.png"),
    )

    for name, fake_df in generated.items():
        save_correlation_heatmap(
            fake_df, features,
            f"Feature Correlation Matrix — {display_name(name)} (Synthetic, n={len(fake_df)})",
            os.path.join(correlation_dir, f"correlation_{name}.png"),
        )

    save_pca_scatter(real_df, generated, features, os.path.join(correlation_dir, "pca_real_vs_generated.png"))
    save_pca_individual_panels(real_df, generated, features, os.path.join(correlation_dir, "pca_individual.png"))

    # Both DataFrames keep raw method keys and column names in memory, since
    # callers filter by them; only the CSV written to disk gets display names.
    ks_table = marginal_ks_table(real_df, generated, features)
    _with_display_names(ks_table).to_csv(os.path.join(results_dir, "ks_marginals.csv"), index=False)

    save_ks_heatmap(ks_table, os.path.join(marginals_dir, "ks_heatmap.png"))
    save_pvalue_heatmap(ks_table, os.path.join(marginals_dir, "pvalue_heatmap.png"))
    save_ecdf_overlay(real_df, generated, features, os.path.join(marginals_dir, "ecdf_overlay.png"))
    save_volcano_plot(ks_table, os.path.join(marginals_dir, "volcano_plot.png"))

    std_table = feature_std_comparison(real_df, generated, features)
    _with_display_names(std_table).to_csv(os.path.join(results_dir, "feature_std_comparison.csv"), index=False)

    summary = pd.DataFrame([
        {
            "method": name,
            "correlation_euclidean_dist": correlation_euclidean_dist(real_df, fake_df, features),
            "mean_ks_stat": ks_table.loc[ks_table.method == name, "ks_stat"].mean(),
        }
        for name, fake_df in generated.items()
    ])
    _with_display_names(summary).to_csv(os.path.join(results_dir, "summary_metrics.csv"), index=False)

    return summary, ks_table, std_table
