"""
Apple quality dataset config. Field meanings are documented once in
src/configs/base.py — this file only overrides what's specific to this
dataset. Run with `make run CONFIG=apple_quality`.
"""
from src.configs.base import *  # noqa: F401,F403 — re-exported as this module's own config values

DATA_PATH = "data/cleaned-scaled-apple-quality.csv"

# Elgiriyewithana, N. (2024). Apple Quality [Data set]. Kaggle.
# https://doi.org/10.34740/kaggle/dsv/7384155
DATA_SOURCE_URL = "https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality"

ID_COLUMNS = ["A_id"]
LABEL_COLUMN = "Quality"

FEATURES = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
]

ROOT_VARIABLES = ["Size", "Ripeness"]

CAUSAL_GRAPH = [
    ("Size", "Weight"),
    ("Ripeness", "Sweetness"),
    ("Ripeness", "Crunchiness"),
    ("Weight", "Juiciness"),
    ("Ripeness", "Acidity"),
]

IS_PRE_SCALED = True

# The causal chain has 7 independent noise sources (2 root Gaussians + 5
# residual terms), so base.py's default 4-dim latent under-fits this data.
LATENT_DIM = 7
VAE_EPOCHS = 4000

# Physics-informed VAE — marginal-matching weight, set where marginal fit
# and per-feature spread saturate without degrading joint structure.
VAE_MARGINAL_WEIGHT = 2.0
