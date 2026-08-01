"""
Standalone leave-one-group-out (LORO) transfer experiment. Kept out of the
main pipeline because it trains a conditional-VAE ensemble per group in two
regimes (far/near) and is far heavier than a normal run.

Usage: make loro CONFIG=<name>   (requires LORO_GROUP set in that config)
"""
import logging
import os
import sys

import numpy as np
import torch

from src.config_loader import DEFAULT_CONFIG, load_config
from src.data_loading import load_data
from src.evaluation.loro import leave_one_group_out, save_loro_plot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_name)

    group_col = getattr(config, "LORO_GROUP", "")
    if not group_col:
        log.info(
            "No LORO_GROUP set in config %r — skipping near/far transfer "
            "(nothing to hold out).", config_name,
        )
        return

    results_dir = f"results/{config_name}"
    os.makedirs(f"{results_dir}/figures/loro", exist_ok=True)

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    df = load_data(config)
    groups = sorted(df[group_col].unique())
    log.info("Same-model near/far transfer on '%s' — %d groups: %s", group_col, len(groups), groups)
    log.info("Per train-region: train one CVAE ensemble on a split of that region, then query "
             "NEAR = held-out same-region, FAR = each other region (same models); this is slow...")

    loro_df = leave_one_group_out(df, config.FEATURES, group_col)
    if loro_df.empty:
        log.info("No LORO groups large enough — nothing to report for %s.", config_name)
        return
    loro_df.to_csv(f"{results_dir}/loro.csv", index=False)

    fig_path = f"{results_dir}/figures/loro/loro_{group_col}.png"
    save_loro_plot(loro_df, group_col, fig_path)

    log.info("Near/far results (FAR = different region/unseen, NEAR = held-out same region):")
    for _, r in loro_df.iterrows():
        log.info(
            "  %-10s corr far/near=%.3f/%.3f  KS=%.3f/%.3f  cov90=%.3f/%.3f  "
            "cov90-conformal=%.3f/%.3f  disagree=%.4f/%.4f",
            str(r["group"]), r["corr_far"], r["corr_near"], r["ks_far"], r["ks_near"],
            r["coverage_far"], r["coverage_near"], r["coverage_far_conf"], r["coverage_near_conf"],
            r["disagreement_far"], r["disagreement_near"],
        )

    mean = loro_df[loro_df["group"] == "MEAN"].iloc[0]
    pct = 100 * (mean["disagreement_far"] / mean["disagreement_near"] - 1)
    if pct > 5:
        verdict = f"{pct:.0f}% higher — uncertainty widens off-support (as expected for genuinely novel groups)"
    elif pct < -5:
        verdict = (f"{abs(pct):.0f}% LOWER — uncertainty does NOT widen here; the held-out groups are "
                   "too similar to training to count as real extrapolation")
    else:
        verdict = f"~unchanged ({pct:+.0f}%) — no clear widening"
    log.info(
        "Headline: mean ensemble disagreement is %.4f for UNSEEN groups vs %.4f for seen — %s.",
        mean["disagreement_far"], mean["disagreement_near"], verdict,
    )
    log.info("Saved %s and %s", f"{results_dir}/loro.csv", fig_path)


if __name__ == "__main__":
    main()
