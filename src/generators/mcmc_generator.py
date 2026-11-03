"""
Correlation-preserving MCMC generator (prior-art baseline).

A faithful, dependency-free re-implementation of the virtual-population method
Onwude et al. (2022, Resources, Conservation and Recycling) used for their
orange-fruit digital twin: an MCMC sampler that draws a realistic population
whose joint correlation matches the sampled data, then feeds a physics-based
model. That paper used Gibbs sampling via the R package NMixMCMC (a normal
mixture); its own prior-art citation is Hertog, Scheerlinck & Nicolai (2009,
J. Comput. Appl. Math.), "Monte Carlo evaluation of biological variation: random
generation of correlated non-Gaussian model parameters."

This module follows that lineage with a Gaussian-copula Gibbs sampler:

  1. Map each real feature to standard-normal scores through its rank (the
     Gaussian-copula transform), and take the correlation R of those scores.
     Rank scores make R capture monotone dependence without assuming the
     marginals are Gaussian (the "non-Gaussian" in Hertog 2009).
  2. Gibbs-sample the latent MVN(0, R): each coordinate is redrawn from its
     analytic full conditional given the others. This is a genuine Markov chain
     (burn-in + thinning), not the ancestral forward sampling of `physics_mc`.
  3. Push each latent margin back through the real empirical quantile function,
     so every generated marginal matches the real one while the copula keeps the
     correlation. (NORTA / Gaussian-copula synthesis.)

Optional `constraints` enforce a physical feasibility region by rejection inside
the chain (e.g. mass balance for a proximate composition), making the sampler a
constrained MCMC. With no correlation structure it reduces to independent
marginal resampling, the same floor as an empty-graph `physics_mc`.

Unlike `physics_mc` (which drops all correlation when no structural graph is given),
this baseline preserves whatever joint correlation the data actually has.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

log = logging.getLogger("vp-lab")


def _normal_scores(x):
    """Gaussian-copula transform: map a real column to standard-normal scores
    through its rank, so corr of the scores is the copula (rank) correlation."""
    u = (rankdata(x, method="average") - 0.5) / len(x)
    return norm.ppf(u)


def _gibbs_conditionals(R):
    """Precompute, for each coordinate j, the regression weights b_j and the
    conditional std for x_j | x_{-j} under MVN(0, R): x_j = b_j . x_{-j} + eps,
    eps ~ N(0, cond_std_j^2)."""
    d = R.shape[0]
    weights, cond_std = [], []
    for j in range(d):
        others = [k for k in range(d) if k != j]
        R_oo = R[np.ix_(others, others)]
        R_jo = R[j, others]
        R_oo_inv = np.linalg.pinv(R_oo)
        b = R_jo @ R_oo_inv
        var = R[j, j] - b @ R_jo
        weights.append(b)
        cond_std.append(np.sqrt(max(var, 1e-9)))
    return weights, cond_std


def generate(df, features, n_samples=1000, constraints=None,
             chain_length=None, burn_in=1000, thin=5, max_reject=50):
    """
    Generate `n_samples` synthetic rows by Gaussian-copula Gibbs MCMC.

    `df`      real data (in source units); `features` the columns to model.
    `constraints`  optional list of callables row_dict -> bool (True = feasible).
                   Infeasible Gibbs draws are rejected and redrawn (constrained
                   MCMC); pass None for the unconstrained sampler.
    `chain_length` total Gibbs sweeps; defaults to burn_in + thin*n_samples.
    `burn_in`, `thin`  standard MCMC burn-in and thinning.
    `max_reject`   per-sample redraw cap before the draw is accepted anyway
                   (keeps a hard constraint from stalling the chain).

    Returns a DataFrame in real units with the real marginals reproduced and the
    data's joint correlation preserved through the copula.
    """
    X = np.asarray(df[features].values, dtype=float)
    n, d = X.shape

    # One feature: no correlation to preserve, just resample the marginal.
    if d == 1:
        vals = np.random.choice(X[:, 0], size=n_samples, replace=True)
        return pd.DataFrame({features[0]: vals})

    Z = np.column_stack([_normal_scores(X[:, j]) for j in range(d)])
    R = np.corrcoef(Z, rowvar=False)
    R = R + 1e-6 * np.eye(d)  # ridge for a safely invertible correlation
    weights, cond_std = _gibbs_conditionals(R)

    if chain_length is None:
        chain_length = burn_in + thin * n_samples

    sorted_cols = [np.sort(X[:, j]) for j in range(d)]

    def _to_real(z_row):
        """Latent normal row -> real units via each feature's empirical quantile."""
        u = norm.cdf(z_row)
        return np.array([np.quantile(sorted_cols[j], u[j]) for j in range(d)])

    def _feasible(real_row):
        if not constraints:
            return True
        row = dict(zip(features, real_row))
        return all(c(row) for c in constraints)

    z = np.zeros(d)
    samples, sweep, since_keep = [], 0, 0
    rejects = 0
    while len(samples) < n_samples:
        sweep += 1
        for j in range(d):
            others = [k for k in range(d) if k != j]
            z[j] = weights[j] @ z[others] + cond_std[j] * np.random.randn()
        since_keep += 1
        if sweep <= burn_in or since_keep < thin:
            continue
        since_keep = 0
        real_row = _to_real(z)
        if not _feasible(real_row):
            tries = 0
            while tries < max_reject and not _feasible(real_row):
                for j in range(d):
                    others = [k for k in range(d) if k != j]
                    z[j] = weights[j] @ z[others] + cond_std[j] * np.random.randn()
                real_row = _to_real(z)
                tries += 1
            rejects += 1
        samples.append(real_row)

    if constraints:
        log.info("  MCMC: %d/%d kept samples needed a constraint redraw", rejects, n_samples)

    return pd.DataFrame(np.array(samples), columns=features)
