import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.generators.hybrid_vae_generator import _build_constraints, _project_constraints

log = logging.getLogger("vp-lab.generators")


def _fit_linear(df, x_col, y_col):
    """Fit y_col ~ x_col on real data and return (slope, intercept, residual_std)."""
    x = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression().fit(x, y)
    residual_std = (y - model.predict(x)).std()

    return model.coef_[0], model.intercept_, residual_std


def generate(df, features, causal_graph, root_variables, n_samples=1000,
             constraints=None):
    """
    Generates synthetic rows via forward Monte Carlo sampling over a
    caller-supplied structural graph:

    - Each variable in `root_variables` is drawn from its own real marginal
      distribution (assumed Gaussian).
    - Each (parent, child) edge in `causal_graph` draws `child` from a
      linear relationship fit on the real data, plus residual noise sampled
      at that fit's real residual std.

    Every conditional here is a Gaussian we can sample directly, so plain
    ancestral Monte Carlo sampling (this function) is exact — there's no
    intractable distribution to approximate, so no need for MCMC.

    This is the Structural Monte Carlo (SMC) baseline: its parent-child edges are
    data-fitted structural priors, not physics. Where the caller supplies
    `constraints` — genuine conservation / mass-balance laws  sum_i w_i x_i <= bound
    in source units, not fitted from data — the generated population is projected
    onto the feasible region with the same hard feasibility projection the PI-VAE
    uses, so those physical laws hold exactly (0% violations). The term
    physics-informed is reserved for the PI-VAE; here the conservation projection is
    an added constraint on a structural baseline. The structural graph, roots and
    constraints are all dataset knowledge supplied by the caller (see src/configs/),
    so the function works unchanged for a different set of features and
    relationships.
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

    result = pd.DataFrame(generated)[features]

    if constraints:
        # Hard feasibility projection in source units: enforces the conservation
        # / mass-balance laws a priori so they hold exactly in the returned
        # population (0% violations), rather than only holding on average through
        # the fitted edges. Same projection the PI-VAE applies at generation.
        A_src, b_src = _build_constraints(constraints, features)
        projected = _project_constraints(result.values, A_src, b_src)
        result = pd.DataFrame(projected, columns=features)
        log.info("  projected generated population onto %d conservation constraint(s)",
                 len(constraints))

    return result
