import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def generate(train_df, features, n_samples=1000):
    """
    For each feature, fits one RandomForestRegressor (on train_df) predicting it
    from the other features. Synthetic rows are built from a base row's
    prediction plus noise, so outputs vary instead of collapsing to a
    deterministic point estimate.

    Both the base rows and the residual noise come from out-of-bag (OOB)
    predictions rather than a held-out split. A random forest bags each tree on a
    bootstrap sample, so roughly a third of the trees never saw any given
    training row; `oob_prediction_` averages only those trees, giving a
    leakage-free prediction for every training row. That lets this generator seed
    itself entirely from train_df, never touching the evaluation set, with no
    data sacrificed to a separate base split. It also makes the residuals
    out-of-sample rather than in-sample, so the bootstrapped noise reflects
    predictive uncertainty instead of understating it.
    """
    train_df = train_df.reset_index(drop=True)

    oob_predictions = {}
    residuals = {}

    for target in features:
        inputs = [f for f in features if f != target]

        model = RandomForestRegressor(n_estimators=100, oob_score=True, bootstrap=True)
        model.fit(train_df[inputs], train_df[target])

        oob_predictions[target] = model.oob_prediction_
        residuals[target] = train_df[target].values - model.oob_prediction_

    base_idx = np.random.randint(0, len(train_df), size=n_samples)

    synthetic = {}
    for target in features:
        point_estimate = oob_predictions[target][base_idx]
        noise = np.random.choice(residuals[target], size=n_samples, replace=True)
        synthetic[target] = point_estimate + noise

    return pd.DataFrame(synthetic)[features]
