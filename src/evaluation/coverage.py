"""
Uncertainty calibration via coverage.

A useful generator produces calibrated uncertainty, not just realistic point
values, which the correlation and KS metrics do not test. This does: for each
feature, take the generated population's
central interval at a nominal level (e.g. the 5th-95th percentile for 90%) and
measure what fraction of the REAL held-out values actually fall inside. A
well-calibrated generator covers real values at the nominal rate; one whose
spread is too narrow (over-confident) under-covers, one too wide over-covers.

This is a *marginal* (per-feature) calibration check — it works identically for
every engine because they all produce a population. Conditional coverage
(intervals that widen out-of-support) needs conditional generation and is left
for the transfer experiments.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.evaluate import GENERATED_COLORS, display_name

FIGURE_DPI = 400
HEADLINE_LEVEL = 0.90
# Grid of nominal central-interval levels for the aggregate calibration error
# and the reliability curve.
LEVELS = np.round(np.arange(0.1, 0.951, 0.05), 2)


def _central_coverage(generated_values, real_values, level):
    """Fraction of real_values inside the generated central interval at `level`."""
    lo = np.quantile(generated_values, (1 - level) / 2)
    hi = np.quantile(generated_values, (1 + level) / 2)
    return float(np.mean((real_values >= lo) & (real_values <= hi)))


def coverage_metrics(generated, real_df, features, levels=LEVELS):
    """
    Per engine: mean coverage at the headline level (nominal HEADLINE_LEVEL, so
    closer to that value = better) and a calibration error — the mean absolute
    gap between empirical and nominal coverage across `levels` and features
    (lower = better calibrated). Returns a DataFrame.
    """
    rows = []
    for name, gen_df in generated.items():
        headline, cal_errors = [], []
        for feature in features:
            gen_values = gen_df[feature].values
            real_values = real_df[feature].values
            headline.append(_central_coverage(gen_values, real_values, HEADLINE_LEVEL))
            cal_errors.append(np.mean([
                abs(_central_coverage(gen_values, real_values, level) - level)
                for level in levels
            ]))
        rows.append({
            "method": name,
            "coverage_at_90": np.mean(headline),
            "calibration_error": np.mean(cal_errors),
        })
    return pd.DataFrame(rows)


def save_calibration_plot(generated, real_df, features, path, levels=LEVELS):
    """
    Reliability curve: nominal coverage (x) vs mean empirical coverage over
    features (y), one line per engine, with the diagonal = perfect calibration.
    A line below the diagonal = under-coverage (over-confident, intervals too
    narrow); above = over-coverage (intervals too wide).
    """
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.5, label="Perfect calibration")

    for color, (name, gen_df) in zip(GENERATED_COLORS, generated.items()):
        empirical = [
            np.mean([_central_coverage(gen_df[f].values, real_df[f].values, level) for f in features])
            for level in levels
        ]
        plt.plot(levels, empirical, marker="o", markersize=4, color=color,
                 linewidth=1.8, label=display_name(name), alpha=0.9)

    plt.xlabel("Nominal coverage (central interval level)")
    plt.ylabel("Empirical coverage on real held-out data (mean over features)")
    plt.title("Uncertainty Calibration — Reliability Curve\n"
              "(on the diagonal = calibrated; below = over-confident; above = too wide)")
    plt.legend(loc="upper left")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
