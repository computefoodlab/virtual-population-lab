"""
Shared dataset loading/cleaning, given a resolved config. Kept separate
from main.py so other entry points (check_generalization.py, ...) can
reuse it without importing main.py itself.
"""
import pandas as pd


def load_data(config):
    df = pd.read_csv(config.DATA_PATH)
    df = df.drop(columns=[c for c in config.ID_COLUMNS if c in df.columns])

    # Coerce first: a stray non-numeric row (e.g. a footer/citation row baked
    # into the CSV) would otherwise silently flip a feature column's dtype
    # to string, and dropna alone wouldn't catch it since it isn't NaN yet.
    df[config.FEATURES] = df[config.FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=config.FEATURES)

    return df
