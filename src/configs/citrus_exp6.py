"""
Citrus orange cold-storage experiment (authors' own unpublished data; part of the
group's citrus research program, but NOT one of the published Onwude digital-twin
datasets) - the individual-fruit flagship. Exp 6, 2019 season: 239 individual oranges (cvs.
Midknight, Nova) across storage temperature (-0.6/2/7 C) x duration (12-42 d),
each fruit measured for the dense per-fruit block: rind fresh weight, rind dry
weight, moisture loss, chilling injury (%), and colour. Unlike the other citrus
files (tree-replicate or compiled), this one is genuinely per individual fruit.

Physics: mass-balance conservation (rind fresh >= dry). The mechanistic
chilling-injury / respiration kinetics (Onwude et al. 2024, 2025) are imposed in
the dedicated PI-VFP experiment; here the structural edge is used for the
five-engine fidelity comparison.

Run: make run CONFIG=citrus_exp6 ; python -m src.multiseed citrus_exp6
"""
from src.configs.base import *  # noqa: F401,F403

DATA_PATH = "data/cleaned-citrus-exp6.csv"
# Authors' own unpublished individual-fruit cold-storage experiment (Exp 6, time x
# temp, 2019 season); available on request. NOT one of the published Onwude citrus
# digital-twin datasets (those are Exp 1 per-replicate CI aggregates and Valencia
# tree-averages). The published papers are cited only for the mechanistic EQUATIONS
# used as the physics constraint (Sci. Rep. 2024, 14:14437), not for this data.
DATA_SOURCE_URL = "authors' unpublished individual-fruit experiment (available on request)"

ID_COLUMNS = []
LABEL_COLUMN = "Cultivar"

FEATURES = ["RindFresh", "RindDry", "MoistureLoss", "ChillingInjury", "Colour"]

# mass-balance structural edge: rind dry is a fraction of rind fresh weight.
ROOT_VARIABLES = ["RindFresh", "MoistureLoss", "ChillingInjury", "Colour"]
CAUSAL_GRAPH = [("RindFresh", "RindDry")]

# Imposed conservation law (mass balance): dry mass cannot exceed fresh mass,
# RindDry <= RindFresh. Holds in 100% of the real fruit; enforced a priori as a
# hard constraint (soft penalty during training + feasibility projection at
# generation), making this a physics-informed constraint, not a fitted relation.
CONSTRAINTS = [({"RindDry": 1.0, "RindFresh": -1.0}, 0.0)]
CONSTRAINT_WEIGHT = 1.0

IS_PRE_SCALED = False
LATENT_DIM = 5
VAE_FREE_BITS = 1.0
RUN_TSTR = True

LORO_GROUP = "Cultivar"
