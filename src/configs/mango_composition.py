"""
Mango composition dataset (Mendeley b9d6s7hr33) — Vitamin C + acidity + sugar.

186 intact mangoes across 4 cultivars (Kent, Palmer, Kweni, Cengkir), each
with three measured composition attributes: Vitamin C (mg/100g), titratable
acidity TA (mg/100g), and soluble solids SSC (degrees Brix). (The source file
also has ~1,500 NIR bands, not used here.)

Scope: 3 features, near-independent (|r| < 0.2), so the correlation comparison is
not discriminative here and no mechanistic causal graph is asserted
(CAUSAL_GRAPH empty). The transfer experiment is leave-one-cultivar-out:
generate an unseen cultivar's Vitamin C / acidity / sugar profile from an
ensemble trained on the others, and measure fidelity and uncertainty near vs
far. All four cultivars have >=18 samples, so all can be held out.

Run: make run CONFIG=mango_composition    (comparison + calibration)
     make loro CONFIG=mango_composition   (leave-one-cultivar-out transfer)
"""
from src.configs.base import *  # noqa: F401,F403 — re-exported as this module's own config values

DATA_PATH = "data/cleaned-mango-composition.csv"
DATA_SOURCE_URL = "https://data.mendeley.com/datasets/b9d6s7hr33/1"

ID_COLUMNS = []
LABEL_COLUMN = "Cultivar"

FEATURES = ["VitaminC", "TA", "SSC"]

# Near-independent features -> no mechanistic structure.
ROOT_VARIABLES = list(FEATURES)
CAUSAL_GRAPH = []

IS_PRE_SCALED = False
LATENT_DIM = 3
VAE_FREE_BITS = 1.0
RUN_TSTR = False

# Leave-one-mango-cultivar-out: Kent, Palmer, Kweni, Cengkir (all >=18).
LORO_GROUP = "Cultivar"
