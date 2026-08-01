"""
Standalone diagnostic: for each engine's already-generated synthetic data
in results/<config>/synthetic_data/, compares correlation_euclidean_dist
against the train split it was fit on vs. the held-out test split it's
actually scored on. A small gap is evidence results generalize rather than
matching training-set noise. Doesn't re-run generation, so it's a fast
recheck after `make run` without paying that cost again.

Usage: make check-overfit CONFIG=<name>
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_loader import DEFAULT_CONFIG, load_config
from src.data_loading import load_data
from src.evaluation.evaluate import display_name, generalization_gap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_name)

    results_dir = f"results/{config_name}"
    synthetic_dir = f"{results_dir}/synthetic_data"

    if not os.path.isdir(synthetic_dir):
        raise SystemExit(
            f"No synthetic data at {synthetic_dir}/ — run `make run CONFIG={config_name}` first."
        )

    np.random.seed(config.RANDOM_SEED)

    df = load_data(config)
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )

    generated = {
        filename[:-4]: pd.read_csv(os.path.join(synthetic_dir, filename))
        for filename in sorted(os.listdir(synthetic_dir))
        if filename.endswith(".csv")
    }
    if not generated:
        raise SystemExit(f"No CSVs found in {synthetic_dir}/.")

    gap_df = generalization_gap(train_df, test_df, generated, config.FEATURES)
    gap_path = f"{results_dir}/generalization_gap.csv"
    gap_df.to_csv(gap_path, index=False)

    log.info(
        "Generalization check for '%s' (train=%d rows, test=%d rows):",
        config_name, len(train_df), len(test_df),
    )
    for _, row in gap_df.iterrows():
        log.info(
            "  %-30s train=%.4f  test=%.4f  gap=%.4f",
            display_name(row["method"]), row["train_correlation_dist"],
            row["test_correlation_dist"], row["gap"],
        )
    log.info("Saved %s", gap_path)


if __name__ == "__main__":
    main()
