"""
Banana quality dataset — reduced to the fruit-measurement features plus the
quality label. Run with `make run CONFIG=banana_quality`.

The categorical variety codes and the agronomy context (tree age, soil, rainfall,
altitude) are dropped: they are label-encoded or only weakly related to the
fruit's own properties. What remains is 5 near-independent fruit measurements
plus the quality label. No reliable causal graph exists here (candidate edges
such as ripeness->sugar have r ~ 0), so this is the no-physics case.

`region` (country codes 0-7) is kept only as the near/far grouping.
"""
from src.configs.base import *  # noqa: F401,F403 — re-exported as this module's own config values

DATA_PATH = "data/cleaned-banana-quality.csv"
DATA_SOURCE_URL = "https://www.kaggle.com/datasets/mrmars1010/banana-quality-dataset"

ID_COLUMNS = ["sample_id"]
# TSTR target, predicted from the fruit measurements (quality_score, its
# deterministic bucket source, is excluded so the task isn't trivial).
LABEL_COLUMN = "quality_category"
LORO_GROUP = "region"
RUN_TSTR = True

FEATURES = ["ripeness_index", "sugar_content_brix", "firmness_kgf", "length_cm", "weight_g"]

# No valid causal graph on this dataset (candidate edges have r~0), so every
# feature is a root and the physics engines fall back to marginal sampling.
ROOT_VARIABLES = list(FEATURES)
CAUSAL_GRAPH = []

IS_PRE_SCALED = False
# Floors KL cost per latent dim so the VAE can't fully collapse.
VAE_FREE_BITS = 5.0
LATENT_DIM = 6
