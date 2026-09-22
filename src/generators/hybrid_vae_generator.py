import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata

log = logging.getLogger("vp-lab")


def _calibrate_marginals(generated, real):
    """
    Empirical-copula marginal calibration. Replaces each generated feature's
    values with the real values at matching quantiles: rank the generated
    column, read those quantiles off the real column's empirical distribution,
    and substitute. The substitution is monotonic in the generated ranks, so the
    learned rank (Spearman) dependence is preserved while every marginal is
    forced to match the real empirical marginal almost exactly — the
    copula-synthesis pattern of learned dependence with real margins.

    `real` is the training reference the VAE was fit on, so no evaluation data
    is used.
    """
    calibrated = np.empty_like(generated)
    n = generated.shape[0]
    for j in range(generated.shape[1]):
        quantiles = (rankdata(generated[:, j], method="average") - 0.5) / n
        # Map to the feature's EMPIRICAL support (actual observed values) via the
        # empirical inverse-CDF, not an interpolated quantile. This preserves
        # discreteness and point masses — e.g. a zero-inflated or binned score
        # whose real support is a handful of distinct values — instead of
        # smearing them across a continuum, matching how the copula inverse-CDF
        # resamples. For continuous features (many distinct values) it is
        # indistinguishable from interpolation. Monotonic in the generated ranks,
        # so the learned rank (Spearman) dependence is preserved.
        real_sorted = np.sort(real[:, j])
        idx = np.clip((quantiles * len(real_sorted)).astype(int), 0, len(real_sorted) - 1)
        calibrated[:, j] = real_sorted[idx]
    return calibrated


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim=128, dropout=0.0):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):

        z = self.encoder(x)

        mu = z[:, :self.latent_dim]
        # Clamp logvar before exponentiating so std = exp(0.5*logvar) cannot run
        # away on very small datasets. The bounds are wide enough never to bind
        # on a well-behaved fit.
        logvar = torch.clamp(z[:, self.latent_dim:], -8.0, 8.0)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        latent = mu + eps * std

        return self.decoder(latent), mu, logvar


def _covariance(batch):
    centered = batch - batch.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / (batch.shape[0] - 1)


def _fit_edges(x, features, causal_graph):
    """
    For each (parent, child) edge, fit child ~ parent on the real matrix `x`
    (the same space the VAE trains in) and return the mechanistic constants
    (slope, intercept, residual variance) plus the column indices. These are the
    same linear-Gaussian conditionals the physics-informed Monte Carlo generator
    samples from; here they become a differentiable constraint on the VAE
    instead of a sampler.
    """
    index = {name: i for i, name in enumerate(features)}
    edges = []

    for parent, child in causal_graph:
        if parent not in index or child not in index:
            raise ValueError(
                f"Causal edge ({parent!r} -> {child!r}) references a feature not in "
                f"FEATURES {features} — check the config's CAUSAL_GRAPH."
            )

        p = x[:, index[parent]]
        c = x[:, index[child]]

        slope = np.cov(p, c, ddof=0)[0, 1] / np.var(p)
        intercept = c.mean() - slope * p.mean()
        resid_var = np.var(c - (slope * p + intercept))

        edges.append({
            "parent_idx": index[parent],
            "child_idx": index[child],
            "slope": float(slope),
            "intercept": float(intercept),
            "resid_var": float(resid_var),
        })

    return edges


def build_edge_constants(x, features, causal_graph, fixed_slopes, feature_scale=None):
    """
    Like `_fit_edges`, but any edge whose (parent, child) key appears in
    `fixed_slopes` uses a slope imposed a priori (from a mechanistic/literature
    constant, given in SOURCE units) instead of the data-fitted one; all other
    edges are still fit from data. The intercept and residual variance are always
    taken from the data so the imposed constant only fixes the *slope* (the
    mechanism), leaving the developmental offset and the physical noise level
    identified from the sample.

    `x` is the scaled matrix the VAE trains in. A source-unit slope m maps to the
    scaled space as m * std_parent / std_child; `feature_scale` supplies those
    per-feature standard deviations (a StandardScaler's `scale_`), or None when
    `x` is already in source units (then the slope is used as-is).

    Example: grape Glucose->Fructose is imposed at m=1.0 (invertase produces
    glucose and fructose in 1:1 stoichiometry, equal molar mass), which the data
    independently confirms (fitted slope ~0.99), so imposing it does not change
    fidelity but makes the edge mechanism-derived rather than self-fitted.
    """
    index = {name: i for i, name in enumerate(features)}
    edges = []

    for parent, child in causal_graph:
        if parent not in index or child not in index:
            raise ValueError(
                f"Edge ({parent!r} -> {child!r}) references a feature not in "
                f"FEATURES {features} — check the config's CAUSAL_GRAPH."
            )
        pi, ci = index[parent], index[child]
        p, c = x[:, pi], x[:, ci]

        if (parent, child) in fixed_slopes:
            m = float(fixed_slopes[(parent, child)])
            if feature_scale is not None:
                slope = m * float(feature_scale[pi]) / float(feature_scale[ci])
            else:
                slope = m
        else:
            slope = np.cov(p, c, ddof=0)[0, 1] / np.var(p)

        intercept = c.mean() - slope * p.mean()
        resid_var = np.var(c - (slope * p + intercept))

        edges.append({
            "parent_idx": pi,
            "child_idx": ci,
            "slope": float(slope),
            "intercept": float(intercept),
            "resid_var": float(resid_var),
        })

    return edges


def _marginal_loss(generated, real):
    """
    Per-feature squared 1D Wasserstein distance: sort each column of the
    generated and real batches and take the mean squared difference between
    the two sorted sequences. This is the differentiable form of what the KS
    test measures: it pulls each generated feature's whole empirical
    distribution (shape, spread, tails) onto real, not just its first two
    moments.

    The physics term constrains only parent->child edges, leaving the causal
    graph's root variables free to be compressed by the VAE. Applying this term
    to samples drawn from the prior shapes the marginals of the generated
    population, root variables included.
    """
    gen_sorted, _ = torch.sort(generated, dim=0)
    real_sorted, _ = torch.sort(real, dim=0)
    return ((gen_sorted - real_sorted) ** 2).mean()


def _decode_samples(vae, reference_batch, n_samples, latent_dim, prior_type):
    """
    Decode `n_samples` synthetic rows. With prior_type="standard" the latents
    are drawn from the N(0, I) prior (the textbook VAE generator). With
    prior_type="aggregate" they are drawn from the aggregate posterior — the
    mixture (1/N) sum_i N(mu_i, sigma_i^2) over the encoded reference rows,
    sampled by picking a random reference row and reparameterizing from its
    posterior. That keeps the latents in the region the decoder learned to map to
    data, avoiding the prior-hole mismatch a standard-normal draw can fall into
    when the aggregate posterior does not fill the prior. Gradients flow through
    the encoder (aggregate case) and decoder, so this is usable inside the
    training loss as well as for final generation.
    """
    if prior_type == "aggregate":
        enc = vae.encoder(reference_batch)
        mu = enc[:, :latent_dim]
        logvar = torch.clamp(enc[:, latent_dim:], -8.0, 8.0)
        idx = torch.randint(0, reference_batch.shape[0], (n_samples,))
        std = torch.exp(0.5 * logvar[idx])
        z = mu[idx] + torch.randn(n_samples, latent_dim) * std
    else:
        z = torch.randn(n_samples, latent_dim)
    return vae.decoder(z)


def _build_constraints(constraints, features):
    """Turn a list of (weights_dict, bound) inequalities  sum_i w_i x_i <= bound
    (in real units) into coefficient matrix A (C x D) and bound vector b (C).
    These are genuine physical laws (mass balance / non-negativity), not fitted
    from data — what makes the constrained VAE physics-informed rather than only
    structure-informed."""
    idx = {f: i for i, f in enumerate(features)}
    A = np.zeros((len(constraints), len(features)), np.float32)
    b = np.zeros(len(constraints), np.float32)
    for r, (weights, bound) in enumerate(constraints):
        for f, w in weights.items():
            A[r, idx[f]] = w
        b[r] = bound
    return A, b


def _constraint_penalty(raw_batch, A_t, b_t):
    """Mean squared violation of A x <= b: relu(A x - b)^2. Zero when every row is
    physically feasible (conservation / non-negativity respected)."""
    viol = torch.relu(raw_batch @ A_t.T - b_t)
    return (viol ** 2).mean()


def _project_constraints(X, A, b, nonneg=True, iters=25):
    """
    Hard feasibility projection at generation: for each generated row, project it
    onto the region A x <= b by cyclic Euclidean projection onto each violated
    half-space (subtract (a.x - b) a / ||a||^2), a few passes to handle coupled
    constraints. Optionally clip to non-negative afterwards. Unlike the soft
    training penalty, this GUARANTEES the conservation / compositional laws hold
    in the returned population (0% violations) rather than merely discouraging
    violations. A and b are in source units, matching the generated matrix.
    """
    X = np.array(X, dtype=float, copy=True)
    for _ in range(iters):
        moved = False
        for r in range(A.shape[0]):
            w = A[r]
            wn = float((w ** 2).sum())
            if wn <= 0:
                continue
            viol = X @ w - b[r]
            m = viol > 1e-9
            if m.any():
                X[m] -= np.outer(viol[m], w) / wn
                moved = True
        if nonneg:
            X = np.clip(X, 0.0, None)
        if not moved:
            break
    return X


def _physics_loss(batch, edges):
    """
    Penalizes how far a batch departs from the fitted causal relationships,
    matching all three moments of each edge's real linear-Gaussian conditional
    so the constraint anchors the mechanism *without* collapsing spread:

    - `mean(resid)^2`         — intercept/bias: residuals centered on the line.
    - `cov(resid, parent)^2`  — slope: residuals uncorrelated with the parent
                                (a wrong slope leaves parent-correlated residual).
    - `(var(resid) - resid_var)^2` — the physical noise level: penalizes a
                                conditional spread that is too tight (the classic
                                VAE variance-collapse failure) as much as one too
                                loose, so the physics term defends spread rather
                                than suppressing it.

    Returns 0 when there are no edges (an empty causal graph, i.e. a dataset with
    no reliable mechanistic structure), in which case the physics-informed VAE
    reduces to a calibrated VAE with the covariance term.
    """
    total = batch.new_zeros(())
    if not edges:
        return total

    for edge in edges:
        parent = batch[:, edge["parent_idx"]]
        child = batch[:, edge["child_idx"]]

        resid = child - (edge["slope"] * parent + edge["intercept"])

        resid_centered = resid - resid.mean()
        parent_centered = parent - parent.mean()
        cov_resid_parent = (resid_centered * parent_centered).mean()

        total = total + (
            resid.mean() ** 2
            + cov_resid_parent ** 2
            + (resid.var(unbiased=False) - edge["resid_var"]) ** 2
        )

    return total / len(edges)


def generate(
    x,
    features,
    causal_graph,
    latent_dim,
    epochs,
    beta,
    n_samples=1000,
    scaler=None,
    kl_warmup_frac=0.3,
    use_minibatch=True,
    batch_size=128,
    cov_weight=1.0,
    physics_weight=1.0,
    marginal_weight=0.0,
    prior_type="standard",
    constrain_generated=False,
    calibrate_marginals=False,
    patience=200,
    hidden_dim=128,
    dropout=0.0,
    free_bits=0.0,
    constraints=None,
    constraint_weight=0.0,
    edge_constants=None,
    calib_reference=None,
):
    """
    The physics-informed VAE (PI-VAE): the same generative model as the plain
    VAE, plus a physics-consistency loss that injects the caller-supplied causal
    graph into training. It keeps the VAE's data-driven strengths (a learned
    latent joint, novel-individual sampling, no handcrafted marginals) while
    borrowing the physics-informed Monte Carlo generator's domain knowledge: the
    fitted linear-Gaussian relationship on each causal edge.

    Each (parent, child) edge in `causal_graph` is fit once on the real data
    (slope, intercept, residual variance — the same conditionals the physics-MC
    generator samples). Those constants become a differentiable penalty (see
    `_physics_loss`), weighted by `physics_weight`, that pushes each batch onto
    the mechanistic relationships while matching their real residual spread, so
    the constraint guides the generative manifold toward physically consistent
    samples without re-introducing variance shrinkage.

    Trains on `x` (expected already reasonably scaled) and samples `n_samples`
    rows from the prior; pass a fitted `scaler` only if `x` needs
    inverse-transforming back to source units. Loss is the standard VAE ELBO
    (reconstruction MSE + analytic KL to the standard-normal prior, `beta`
    linearly annealed from 0 over the first `kl_warmup_frac` of training to avoid
    posterior collapse), plus a covariance-matching term (`cov_weight`) that
    rewards preserving the full correlation matrix, plus the physics term
    (`physics_weight`). `cov_weight` shapes the whole covariance structure from
    data; `physics_weight` anchors the specific mechanistic edges an expert
    asserts. Set `physics_weight=0.0` to recover the plain VAE.

    A fourth term, weighted by `marginal_weight`, is the per-feature 1D
    Wasserstein distance (see `_marginal_loss`) between samples drawn from the
    prior and the real batch. It shapes every generated feature's whole
    distribution onto real, including the causal-graph root variables that the
    physics term leaves unconstrained; set `marginal_weight=0.0` to disable it.

    `prior_type` chooses how latents are drawn at generation and for the
    generated-sample loss terms: "standard" samples the N(0, I) prior,
    "aggregate" samples the aggregate posterior over the training rows (see
    `_decode_samples`).

    `constrain_generated` routes the covariance and physics terms onto a batch of
    freshly generated rows instead of the reconstructions, so those constraints
    shape the population actually sampled at generation. The default (False)
    keeps them on reconstructions, matching the plain VAE's convention.

    `calibrate_marginals` applies the empirical-copula post-step (see
    `_calibrate_marginals`): each feature is mapped onto the real training
    marginal at matching quantiles, forcing every marginal to match real almost
    exactly while preserving the learned rank dependence. A run using it is
    marginal-calibrated rather than purely learned. Default False.

    `use_minibatch`, `batch_size`, `patience`, `hidden_dim`, `dropout`, and
    `free_bits` behave as in the plain VAE generator — see its docstring for the
    capacity, collapse, and early-stopping trade-offs.
    """
    x = np.asarray(x, dtype=np.float32)
    if physics_weight <= 0:
        # Edge-consistency term disabled (default): skip fitting the structural
        # edges entirely, since they would only be multiplied by a zero weight.
        # The PI-VAE's imposed physics is the conservation constraint below, not
        # these data-fitted edges. _physics_loss returns 0 for an empty edge set.
        edges = []
    elif edge_constants is not None:
        # Use edge parameters specified a priori (e.g. from literature/mechanism)
        # instead of fitting slope/intercept/residual from the observed data. This
        # makes the structural term genuinely mechanism-informed, not self-fitted.
        edges = edge_constants
        log.info("  using %d PRE-SPECIFIED (literature/mechanistic) edge(s)", len(edges))
    else:
        edges = _fit_edges(x, features, causal_graph)
        log.info("  fit %d causal edge(s) as physics constraints: %s",
                 len(edges), ", ".join(f"{p}->{c}" for p, c in causal_graph))

    # Conservation / mass-balance constraints (real physical laws, in source units).
    # Evaluated on the generated batch after inverse-scaling, so they need the
    # scaler's mean/scale as tensors when the VAE trains in standardized space.
    use_constraints = bool(constraints) and constraint_weight > 0
    if use_constraints:
        A_np, b_np = _build_constraints(constraints, features)
        if scaler is not None:
            mean_t = torch.FloatTensor(scaler.mean_.astype(np.float32))
            scale_t = torch.FloatTensor(scaler.scale_.astype(np.float32))
            row_scale = np.maximum(np.abs(A_np) @ scaler.scale_.astype(np.float32), 1e-6)
        else:
            mean_t, scale_t = None, None
            row_scale = np.maximum(np.abs(A_np).sum(1), 1e-6)
        # normalize each inequality by the typical magnitude of its linear combo,
        # so the penalty is in ~std units and comparable to the other loss terms
        A_np = A_np / row_scale[:, None]
        b_np = b_np / row_scale
        A_t, b_t = torch.FloatTensor(A_np), torch.FloatTensor(b_np)
        log.info("  enforcing %d conservation constraint(s), weight %.2f",
                 len(constraints), constraint_weight)

    x_tensor = torch.FloatTensor(x)
    n = x_tensor.shape[0]
    step_size = batch_size if use_minibatch else n

    vae = VAE(input_dim=x_tensor.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=0.001
    )

    log_every = max(epochs // 4, 1)
    warmup_epochs = max(1, int(epochs * kl_warmup_frac))
    epoch_width = len(str(epochs))

    best_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        current_beta = beta * min(1.0, (epoch + 1) / warmup_epochs)

        perm = torch.randperm(n)
        total_loss = total_recon = total_kl = total_cov = total_phys = total_marg = 0.0
        n_batches = 0

        for start in range(0, n, step_size):
            batch = x_tensor[perm[start:start + step_size]]

            recon, mu, logvar = vae(batch)

            recon_loss = ((recon - batch) ** 2).mean()

            kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # always >= 0
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

            # A batch of freshly generated rows, drawn the same way final
            # generation is: always needed by the marginal term, and by the
            # covariance/physics terms when constrain_generated is set.
            if marginal_weight > 0 or constrain_generated or use_constraints:
                gen = _decode_samples(vae, batch, batch.shape[0], latent_dim, prior_type)

            cov_target = gen if constrain_generated else recon
            cov_loss = ((_covariance(cov_target) - _covariance(batch)) ** 2).mean()
            physics_loss = _physics_loss(cov_target, edges)

            if marginal_weight > 0:
                marginal_loss = _marginal_loss(gen, batch)
            else:
                marginal_loss = torch.zeros((), device=recon.device)

            if use_constraints:
                raw_gen = gen * scale_t + mean_t if scale_t is not None else gen
                constraint_loss = _constraint_penalty(raw_gen, A_t, b_t)
            else:
                constraint_loss = torch.zeros((), device=recon.device)

            loss = (
                recon_loss
                + current_beta * kl_loss
                + cov_weight * cov_loss
                + physics_weight * physics_loss
                + marginal_weight * marginal_loss
                + constraint_weight * constraint_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_cov += cov_loss.item()
            total_phys += physics_loss.item()
            total_marg += marginal_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if epoch % log_every == 0 or epoch == epochs - 1:
            log.info(
                "  epoch %s/%d — loss %7.4f (recon %7.4f, kl %7.4f, cov %7.4f, phys %7.4f, marg %7.4f, beta %5.3f)",
                str(epoch + 1).rjust(epoch_width), epochs,
                avg_loss, total_recon / n_batches,
                total_kl / n_batches, total_cov / n_batches,
                total_phys / n_batches, total_marg / n_batches, current_beta,
            )

        if epoch >= warmup_epochs:
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                log.info(
                    "  early stopping at epoch %d/%d — no improvement for %d epochs",
                    epoch + 1, epochs, patience,
                )
                break

    vae.eval()  # disable dropout for generation — it's a training-only regularizer

    with torch.no_grad():
        generated = _decode_samples(vae, x_tensor, n_samples, latent_dim, prior_type).numpy()

    if scaler is not None:
        generated = scaler.inverse_transform(generated)

    if calibrate_marginals:
        # In SOURCE units, against the training data. Calibrating here (rather than
        # in standardized space) preserves each feature's exact empirical support:
        # standardizing and inverse-transforming perturbs an atom such as an exact
        # zero into a tiny non-zero value, which destroys the point mass of a
        # zero-inflated / binned feature. In source units the copula inverse-CDF
        # maps generated ranks straight onto the real observed values.
        if calib_reference is not None:
            x_source = np.asarray(calib_reference)
        elif scaler is not None:
            x_source = scaler.inverse_transform(x)
        else:
            x_source = np.asarray(x)
        generated = _calibrate_marginals(generated, x_source)

    if use_constraints:
        # Hard feasibility projection in source units: guarantees the conservation
        # / compositional laws hold in the returned population (0% violations),
        # turning the soft training penalty into an enforced physical constraint.
        A_src, b_src = _build_constraints(constraints, features)
        generated = _project_constraints(generated, A_src, b_src)
        log.info("  projected generated population onto %d conservation constraint(s)",
                 len(constraints))

    return pd.DataFrame(
        generated,
        columns=features
    )
