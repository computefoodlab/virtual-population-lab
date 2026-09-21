"""
Safou (Dacryodes edulis) lipid composition — individual-TREE composition samples
from primary African studies, compiled into FAO/INFOODS BioFoodComp4.0 (v4.0).
41 samples: Congo (Brazzaville, Boko; the database's own note reads "Sample N
derive from one individual tree", n=21) and Guinea (Trees A/B/C, n=20, where
"different samples represent different stages of maturity"). So this is genuine
individual-level data (per-tree), NOT a compiled aggregate table like the date
set — but the observed spread mixes between-tree, between-region (Congo/Guinea)
AND maturity-stage variation (Guinea samples span unripe->ripe, ~0.1%->22% fat),
not a single fixed-maturity population. n=41 is small, so fidelity/spread numbers
are INDICATIVE / proof-of-concept only. Near-deterministic mass-balance graph
(Water->Fat; Fat->fatty acids); the conservation laws hold. 4 features: fat,
palmitic, stearic, water.

Run: make run CONFIG=biofood_safou_region ; make loro CONFIG=biofood_safou_region
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-biofood-safou-region.csv"
# FAO/INFOODS Food Composition Database for Biodiversity v4.0 (BioFoodComp4.0),
# FAO, Rome — Dacryodes edulis entries (compiled, cross-region/study).
DATA_SOURCE_URL = "https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/"

ID_COLUMNS = []
LABEL_COLUMN = "Region"

FEATURES = ["Water", "Fat", "Palmitic", "Stearic"]

# Universal mass-balance chain: water displaces fat; the fatty acids are
# fractions of total fat. Holds in both regions.
ROOT_VARIABLES = ["Water"]
CAUSAL_GRAPH = [
    ("Water", "Fat"),
    ("Fat", "Palmitic"),
    ("Fat", "Stearic"),
]

# Imposed conservation laws (both hold in 100% of the real fruit): the measured
# fatty acids are fractions of total fat (Palmitic + Stearic <= Fat), and the
# proximate composition cannot exceed 100% (Water + Fat <= 100). Enforced a
# priori as hard constraints (soft penalty + feasibility projection at
# generation) - genuine mass balance, not fitted from the outcome.
CONSTRAINTS = [
    ({"Palmitic": 1.0, "Stearic": 1.0, "Fat": -1.0}, 0.0),
    ({"Water": 1.0, "Fat": 1.0}, 100.0),
]
CONSTRAINT_WEIGHT = 1.0

# The source values are raw nutrient concentrations on very different scales
# (Water ~72, Stearic ~0.26), so the pipeline fits its own StandardScaler for the
# VAEs; feeding raw values to a VAE at this scale spread makes it diverge.
IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
# No dropout: the near-deterministic physics needs full decoder capacity.
RUN_TSTR = False

# Calibration on: the lipid margins are near-symmetric, so the empirical-copula
# remap preserves correlations while removing the decoder's physically impossible
# tail (it otherwise emits negative fat in the lean-fruit region). Margins are
# therefore real rather than purely learned.
VAE_CALIBRATE_MARGINALS = True
VAE_PHYSICS_WEIGHT = 0.0

# Covariance-matching term off here. The 4 features are already pinned by the
# near-deterministic physics edges (r 0.76-0.99), and a 4x4 sample covariance
# estimated from only ~28 training rows is too noisy to be a useful target — it
# destabilizes the correlation structure the physics already fixes, with no
# effect on marginal fit or calibration (both set by the copula step). Kept on
# for every other dataset, where it is the main driver of joint structure.
VAE_COV_WEIGHT = 0.0

# Leave-one-region-out: Congo (oil-rich) vs Guinea (lean).
LORO_GROUP = "Region"
