"""
Date (Phoenix dactylifera) mineral composition, grouped by region of origin —
one fruit subgrouped by growing region, so transfer runs from a data-rich region
to a data-scarce one within a single fruit.

  77 entries (complete-case on the 5 dense features Water + K + Fe + Ca + Mg),
  across three origins:
    - UAE     (n=64) — data-rich
    - Tunisia (n=10) — data-scarce (standardized centroid distance to UAE
                       = 3.1 sigma)
    - Oman    (n=3)  — too small to hold out; stays in training

Scope: only two regions are large enough to hold out (UAE, Tunisia), so this is a
single in/out split rather than a multi-distance gradient. This is a compiled
composition table, not sampled individuals, so it serves as a corroborating
case. The dilution physics holds within date (Water->Mg r = -0.77, ->K -0.75),
so the physics engines are exercised.

Run: make run  CONFIG=biofood_date_region   (comparison on the pooled date data)
     make loro CONFIG=biofood_date_region   (leave-one-region-out transfer)
"""
from src.configs.base import *  # noqa: F401,F403 — re-exported as this module's own config values

DATA_PATH = "data/cleaned-biofood-date-region.csv"
DATA_SOURCE_URL = "https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/"

ID_COLUMNS = []
LABEL_COLUMN = "Region"

FEATURES = ["Water", "Potassium", "Iron", "Calcium", "Magnesium"]

# Water dilutes the per-100g mineral concentration — holds within date too.
ROOT_VARIABLES = ["Water"]
CAUSAL_GRAPH = [
    ("Water", "Potassium"),
    ("Water", "Iron"),
    ("Water", "Calcium"),
    ("Water", "Magnesium"),
]

IS_PRE_SCALED = False
LATENT_DIM = 3
VAE_FREE_BITS = 1.0
# Small n (53 train rows): cap epochs (shorter KL warmup) and add mild dropout so
# the deep engines do not overfit.
VAE_EPOCHS = 2000
VAE_DROPOUT = 0.1
RUN_TSTR = False

# Leave-one-region-out: UAE (data-rich) vs Tunisia (data-scarce); Oman is too
# small to hold out and stays in training.
LORO_GROUP = "Region"
RANDOM_SEED = 42