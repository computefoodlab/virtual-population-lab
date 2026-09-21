"""
Tomato taste-compound composition — individual whole-fruit data (Ibanez et al.
2019, J. Food Engineering 263:237-242; Zenodo 10633732, CC-BY-SA-4.0). Each of
650 tomatoes was NIR-scanned and then crushed for its OWN reference chemistry, so
every row is one individual fruit (not accession means). Six taste compounds:
soluble solids (SSC, Brix), fructose, glucose, citric, malic and glutamic acid.

Groups: 5 varietal types (ProcessingN 168, ProcessingE 180, Cherry 106,
MidSized 108, Landrace 88) that are moderately distinct in sugar level (cherry /
landrace sweet, processing / mid-sized diluted) -> a genuine leave-one-type-out
extrapolation task.

Physics:
- Soluble-solids mass balance: Brix approximates total dissolved solutes, so SSC
  tracks the sugars (SSC-glucose r=0.83, SSC-fructose r=0.78).
- Hexose-pool co-accumulation: fructose and glucose move together (r=0.93).

Run: make run CONFIG=tomato_nir ; make loro CONFIG=tomato_nir
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-tomato-nir.csv"
DATA_SOURCE_URL = "https://zenodo.org/records/10633732"

ID_COLUMNS = []
LABEL_COLUMN = "Type"

FEATURES = ["SSC", "Malic", "Citric", "Glutamic", "Fructose", "Glucose"]

# Hexose co-accumulation + Brix reflecting the sugar pool (soluble-solids balance).
ROOT_VARIABLES = ["Glucose", "Malic", "Citric", "Glutamic"]
# Structural priors grounded in the literature (associational, not causal claims):
CAUSAL_GRAPH = [
    # Glucose and fructose are the dominant hexoses, both released by sucrose
    # cleavage, so they co-vary (r = +0.93). Zhao et al., BMC Plant Biol. 2022
    # (s12870-022-03685-8); sugars are ~65% of tomato TSS.
    ("Glucose", "Fructose"),
    # Brix/TSS is a refractometric measure of dissolved solids; sugars form the
    # major fraction (~65%), so Brix tracks hexoses (r = +0.83) - a definitional/
    # compositional link (acids also contribute). Shammai et al., Front. Genet.
    # 2021 (fgene.2021.714942).
    ("Glucose", "SSC"),
]

# Imposed compositional law: the dissolved sugars are part of the soluble solids,
# so glucose + fructose (g/kg) cannot exceed the total soluble solids SSC (Brix,
# g/100g): (Glucose + Fructose) <= 10 * SSC. Holds in 99.7% of the real fruit (the
# factor 10 is the g/kg -> g/100g unit conversion). Enforced a priori as a hard
# constraint (soft penalty + feasibility projection at generation).
CONSTRAINTS = [({"Glucose": 1.0, "Fructose": 1.0, "SSC": -10.0}, 0.0)]
CONSTRAINT_WEIGHT = 1.0

IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
RUN_TSTR = True

LORO_GROUP = "Type"
