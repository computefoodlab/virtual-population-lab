"""
Mechanistically-constrained conditional VAE (PI-VFP for the citrus digital twin).

This generator's constraint term is built from explicit mechanistic equations
applied during training (rather than relationships fitted from the same data):

  1. Conservation / mass balance (hard a-priori equations):
       - inequality pairs  x[heavy] >= x[light]   (e.g. rind fresh >= rind dry,
         so moisture content = fresh - dry >= 0),
       - non-negativity of physical quantities (e.g. %CI >= 0, moisture loss >= 0).
     Penalised on the decoder output in RAW units (differentiable inverse-transform
     of the scaler), so violations are pushed out during training.

  2. Kinetic-mean anchor (mechanistic form, fitted rate constants):
       the CONDITIONAL MEAN of an output at storage state (T, duration) is anchored
       to a kinetic law E[y | T, t] = y0(T) + k(T)*t (zero/first-order accumulation
       with a temperature-dependent rate). Each epoch, for a set of anchor
       conditions, samples are generated from the prior and their mean is pushed to
       the kinetic prediction -- so the mechanism sets the population MEAN while the
       VAE is free to learn the fruit-to-fruit SPREAD around it.

The generative distribution is otherwise a standard conditional VAE (ELBO +
covariance matching), so the model still learns the joint spread from data; only
the mean-trajectory and conservation are imposed mechanistically.
"""
import numpy as np
import torch
import torch.nn as nn

from src.generators.conditional_vae_generator import CVAE, _covariance


def train(x, cond, scaler, latent_dim, epochs, beta=1.0, hidden_dim=128,
          free_bits=1.0, kl_warmup_frac=0.3, patience=150, cov_weight=1.0, seed=0,
          cons_pairs=(), nonneg_idx=(), anchor_conds=None, anchor_targets=None,
          n_anchor=64, lambda_cons=5.0, lambda_kin=5.0):
    """
    x: standardized features (n, d); cond: (n, k) condition matrix; scaler: the
    fitted StandardScaler (for the differentiable raw-unit inverse-transform).
    cons_pairs: list of (heavy_idx, light_idx) enforcing x[heavy] >= x[light].
    nonneg_idx: feature indices constrained >= 0 in raw units.
    anchor_conds: (m, k) condition vectors; anchor_targets: list of dicts
        {feature_idx: target_raw_mean} giving the kinetic-mean target per condition.
    """
    torch.manual_seed(seed)
    x_t, c_t = torch.FloatTensor(x), torch.FloatTensor(cond)
    mean_t = torch.FloatTensor(scaler.mean_); scale_t = torch.FloatTensor(scaler.scale_)
    model = CVAE(x.shape[1], cond.shape[1], latent_dim, hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup = max(1, int(epochs * kl_warmup_frac))
    best, since = float("inf"), 0
    ac = torch.FloatTensor(anchor_conds) if anchor_conds is not None else None

    for epoch in range(epochs):
        cur_beta = beta * min(1.0, (epoch + 1) / warmup)
        recon, mu, logvar = model(x_t, c_t)
        recon_loss = ((recon - x_t) ** 2).mean()
        kl = torch.clamp(0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar), min=free_bits).sum(1).mean()
        cov = ((_covariance(recon) - _covariance(x_t)) ** 2).mean()

        recon_raw = recon * scale_t + mean_t
        cons = recon.new_zeros(())
        for h, l in cons_pairs:
            cons = cons + torch.relu(recon_raw[:, l] - recon_raw[:, h]).pow(2).mean()
        for j in nonneg_idx:
            cons = cons + torch.relu(-recon_raw[:, j]).pow(2).mean()

        kin = recon.new_zeros(())
        if ac is not None:
            for i in range(ac.shape[0]):
                z = torch.randn(n_anchor, latent_dim)
                cc = ac[i].unsqueeze(0).repeat(n_anchor, 1)
                gen_raw = model.decoder(torch.cat([z, cc], dim=1)) * scale_t + mean_t
                for oi, tgt in anchor_targets[i].items():
                    kin = kin + (gen_raw[:, oi].mean() - float(tgt)) ** 2
            kin = kin / ac.shape[0]

        loss = (recon_loss + cur_beta * kl + cov_weight * cov
                + lambda_cons * cons + lambda_kin * kin)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch >= warmup:
            if loss.item() < best - 1e-4:
                best, since = loss.item(), 0
            else:
                since += 1
                if since >= patience:
                    break
    return model


def generate(model, cond_vector, n_samples, latent_dim, scaler,
             cons_pairs=(), nonneg_idx=()):
    """Sample a population for one condition; project generated samples onto the
    hard conservation constraints (fresh >= dry; non-negativity) as a final repair."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        c = torch.FloatTensor(np.tile(cond_vector, (n_samples, 1)))
        out = model.decoder(torch.cat([z, c], dim=1)).numpy()
    out = scaler.inverse_transform(out)
    for h, l in cons_pairs:            # enforce x[heavy] >= x[light]
        viol = out[:, l] > out[:, h]
        mid = (out[viol, l] + out[viol, h]) / 2
        out[viol, l], out[viol, h] = mid, mid
    for j in nonneg_idx:
        out[:, j] = np.clip(out[:, j], 0, None)
    return out
