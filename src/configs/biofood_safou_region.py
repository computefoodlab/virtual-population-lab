"""
Safou (Dacryodes edulis) lipid composition — individual-level nutrient data with
real growing regions. 41 individual African-pear fruits (one fruit per tree):
Congo (oil-rich, n=21) and Guinea (lean, n=20). Near-deterministic mass-balance
graph (Water->Fat r = -0.99; Fat->fatty acids). At this sample size the fidelity
numbers are indicative only. 4 features: 3 nutrients (fat, palmitic, stearic)
plus water.

Run: make run CONFIG=biofood_safou_region ; make loro CONFIG=biofood_safou_region
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-biofood-safou-region.csv"
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
VAE_PHYSICS_WEIGHT = 3.0

# Covariance-matching term off here. The 4 features are already pinned by the
# near-deterministic physics edges (r 0.76-0.99), and a 4x4 sample covariance
# estimated from only ~28 training rows is too noisy to be a useful target — it
# destabilizes the correlation structure the physics already fixes, with no
# effect on marginal fit or calibration (both set by the copula step). Kept on
# for every other dataset, where it is the main driver of joint structure.
VAE_COV_WEIGHT = 0.0

# Leave-one-region-out: Congo (oil-rich) vs Guinea (lean).
LORO_GROUP = "Region"
