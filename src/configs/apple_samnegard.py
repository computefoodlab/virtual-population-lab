"""
Apple (cv. Amorosa) individual-fruit quality + mineral composition — Samnegard
et al. (Zenodo 4989885 / Dryad doi:10.5061/dryad.nk9871p). REAL per-fruit data:
each row is one apple (unique AppleID), 255 apples across two Swedish orchards,
pollination treatments and cold-storage categories. This is a genuine individual-fruit
apple dataset with a documented protocol and physical units.

Eight measured per-fruit features: Brix (soluble solids), TA (titratable
acidity), firmness, dry-matter %, and the phloem-loaded minerals K, Ca, Mg, P
(ICP).

Physics:
- Phloem-mobile cation co-variation (Marschner): K, Mg and P are phloem-mobile
  and co-vary strongly (K-Mg r=0.87, K-P r=0.89, P-Mg r=0.83), while Ca is
  xylem-borne and near-independent -> the same mechanism as the date dataset.
- Dry-matter / soluble-solids mass balance: dry matter carries the sugars, so
  DryMatter tracks Brix (r=0.70).

Groups: cold-storage category (initial 90, early 40, late 24, final 108) -- a
single cultivar, so the meaningful group is the storage/ripening axis (which
shifts firmness, Brix and dry-matter), not cultivar/region. Pollination
Treatment is also present but barely changes composition (real-data TSTR ceiling
~0.53), whereas Store_cat is a genuine downstream signal (ceiling ~0.80).

Run: make run CONFIG=apple_samnegard ; make loro CONFIG=apple_samnegard
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-apple-samnegard.csv"
DATA_SOURCE_URL = "https://zenodo.org/records/4989885"

ID_COLUMNS = []
LABEL_COLUMN = "Store_cat"

FEATURES = ["Brix", "TA", "Firmness", "DryMatter", "K", "Ca", "Mg", "P"]

# Phloem cation co-loading (K drives Mg, P) + dry-matter/sugar mass balance.
# Ca is xylem-borne -> left as a root (near-independent of the phloem cations).
ROOT_VARIABLES = ["K", "Ca", "TA", "Firmness", "DryMatter"]
# Structural priors grounded in the literature (associational, not causal claims):
CAUSAL_GRAPH = [
    # K, P and Mg form a positively correlated mineral block in apple fruit
    # (K-P r=0.79, K-Mg r=0.74); K is the dominant phloem-mediated driver and
    # P/Mg are collinear with it -> shared-transport co-accumulation
    # (assoc. across 140 cultivars, ACS JAFC 2026, PMC13426305).
    ("K", "Mg"),          # r = +0.87 in this dataset
    ("K", "P"),           # r = +0.89 in this dataset
    # Dry matter is dominated by sugars+starch; harvest DM predicts post-storage
    # SSC at R^2=0.93 (SSC ~85% sugars) - a compositional relationship
    # (Palmer et al., Postharvest Biol. Technol. 2010, S0925521402002077).
    ("DryMatter", "Brix"),  # r = +0.70 in this dataset
]

IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
RUN_TSTR = True

LORO_GROUP = "Store_cat"
