"""
Grape single-berry composition (LowSugarBerry data paper, FigShare
doi:10.6084/m9.figshare.28308986; Table 6).

1,517 individual grape berries across 6 genotypes (Grenache, Merlot, Morrastel,
G5, G7, G14) over 22 ripening sampling dates, with HPLC/enzymatic composition:
glucose, fructose, malic acid, tartaric acid, and minerals K/Mg/Ca. Real,
non-uniform, and rich in genuine physics: glucose-fructose co-accumulation
(r=0.99), glucose-malic ripening trade-off (r=-0.81), and phloem-mobility
structure among minerals (K-Mg co-vary r=0.70, Ca decoupled r=0.41) — the same
mechanism as the date set.
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-grape-berry.csv"
DATA_SOURCE_URL = "https://doi.org/10.6084/m9.figshare.28308986"
ID_COLUMNS = []
LABEL_COLUMN = "Genotype"
FEATURES = ["Glucose","Fructose","MalicAcid","TartaricAcid","Potassium","Magnesium","Calcium"]
# Structural priors grounded in the literature (associational, not causal claims):
ROOT_VARIABLES = ["Glucose","MalicAcid","TartaricAcid","Potassium","Calcium"]
CAUSAL_GRAPH = [
    # Vacuolar invertase hydrolyses sucrose into equimolar glucose+fructose; the
    # Fru:Glc ratio approaches 1.0 at maturity, matching the 1:1 stoichiometry of
    # sucrolysis (Davies & Robinson, Plant Physiol. 1996; ScienceDirect S0981942899800047).
    ("Glucose", "Fructose"),
    # K and Mg are both phloem-mobile cations that co-accumulate through ripening,
    # K being the dominant phloem osmoticum -> shared-transport co-accumulation
    # (Rogiers et al., Front. Plant Sci. 2017, PMC5623721; Storey, VITIS mineral sinks).
    ("Potassium", "Magnesium"),
]

# Mechanistic constant imposed on the physics term (source units): invertase
# hydrolyses sucrose 1:1 into glucose+fructose (equal molar mass) -> raw slope 1.0.
# The data confirms it (fitted slope ~0.99, real ~1.01) so imposing it leaves
# fidelity unchanged while making the edge mechanism-derived, not self-fitted.
# The intercept and residual spread are still estimated from data.
FIXED_EDGES = [{"parent": "Glucose", "child": "Fructose", "slope": 1.0}]
IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
RUN_TSTR = True
LORO_GROUP = "Genotype"
