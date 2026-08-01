import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def _fit_linear(df, x_col, y_col):
    """Fit y_col ~ x_col on real data and return (slope, intercept, residual_std)."""
    x = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression().fit(x, y)
    residual_std = (y - model.predict(x)).std()

    return model.coef_[0], model.intercept_, residual_std


def generate(df, features, causal_graph, root_variables, n_samples=1000):
    """
    Generates synthetic rows via forward Monte Carlo sampling over a
    caller-supplied causal graph:

    - Each variable in `root_variables` is drawn from its own real marginal
      distribution (assumed Gaussian).
    - Each (parent, child) edge in `causal_graph` draws `child` from a
      linear relationship fit on the real data, plus residual noise sampled
      at that fit's real residual std.

    Every conditional here is a Gaussian we can sample directly, so plain
    ancestral Monte Carlo sampling (this function) is exact — there's no
    intractable distribution to approximate, so no need for MCMC.

    The causal structure and roots are dataset knowledge supplied by the caller
    (see src/configs/) — this function has no dataset-specific assumptions baked
    in, so it works unchanged for a different set of features and relationships.
    """
    generated = {}

    for root in root_variables:
        generated[root] = np.random.normal(
            df[root].mean(),
            df[root].std(),
            n_samples
        )

    for parent, child in causal_graph:
        if parent not in generated:
            raise ValueError(
                f"Parent '{parent}' for child '{child}' hasn't been generated yet — "
                "check that it's in root_variables or appears earlier in causal_graph."
            )

        coef, intercept, resid_std = _fit_linear(df, parent, child)

        generated[child] = (
            coef * generated[parent]
            + intercept
            + np.random.normal(0, resid_std, n_samples)
        )

    missing = set(features) - set(generated)
    if missing:
        raise ValueError(
            f"No generation rule for {missing} — add to root_variables or causal_graph."
        )

    return pd.DataFrame(generated)[features]
