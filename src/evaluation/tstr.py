"""
TSTR (Train on Synthetic, Test on Real) — a downstream-utility check.

Distributional metrics (KS, correlation distance) ask "does the synthetic
population look like the real one?". TSTR asks the harder, more practical
question: "is it good enough to *use*?" A classifier is trained only on
synthetic rows and then evaluated on real held-out rows; the closer it gets to
a classifier trained on real data (the TRTR ceiling), the more genuinely useful
the synthetic population is. This rewards preserving the feature->label
structure — i.e. the joint dependence — not just the marginals.

Labeled synthetic data is produced by stratified conditional generation (each
engine generates each label class separately, from that class's real rows, in
main.py); this module only scores it.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def _score(name, x_train, y_train, x_test, y_test, positive_class, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(x_test))

    roc_auc = np.nan  # AUC is only well-defined for a binary target
    if positive_class is not None and positive_class in list(clf.classes_):
        proba = clf.predict_proba(x_test)[:, list(clf.classes_).index(positive_class)]
        roc_auc = roc_auc_score((y_test == positive_class).astype(int), proba)

    return {"method": name, "accuracy": accuracy, "roc_auc": roc_auc}


def run_tstr(labeled_synthetic, train_df, test_df, features, label_col, seed=42):
    """
    labeled_synthetic: dict {engine_key: DataFrame with `features` + `label_col`}.
    Trains a RandomForest on each engine's synthetic data (and on real data, as
    the TRTR ceiling) and scores it on the real `test_df`. Returns a DataFrame
    with columns method / accuracy / roc_auc, real first.
    """
    classes = sorted(train_df[label_col].dropna().unique())
    positive_class = classes[-1] if len(classes) == 2 else None

    x_test = test_df[features].values
    y_test = test_df[label_col].values

    rows = [_score(
        "real", train_df[features].values, train_df[label_col].values,
        x_test, y_test, positive_class, seed,
    )]
    for key, gen_df in labeled_synthetic.items():
        rows.append(_score(
            key, gen_df[features].values, gen_df[label_col].values,
            x_test, y_test, positive_class, seed,
        ))

    return pd.DataFrame(rows)
