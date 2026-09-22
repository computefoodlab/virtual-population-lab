"""
Conditional VAE (CVAE) — a generative engine that conditions on a covariate
(e.g. growing region) so it can generate a population *for a specified
condition value*, including a value it never saw in training (extrapolation).
The unconditional engines can't do this; it's the capability the
leave-one-group-out transfer experiment (src.evaluation.loro) is built on.

Both encoder and decoder take the one-hot condition alongside their input, so
the decoder can be driven to a chosen condition at generation time.
"""
import numpy as np
import torch
import torch.nn as nn

# Reuse the physics-informed VAE's machinery so the conditional VAE can carry
# the same structural-graph physics loss.
from src.generators.hybrid_vae_generator import _fit_edges, _physics_loss


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        mu, logvar = h[:, :self.latent_dim], torch.clamp(h[:, self.latent_dim:], -8.0, 8.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(torch.cat([z, c], dim=1)), mu, logvar


def _covariance(batch):
    centered = batch - batch.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / (batch.shape[0] - 1)


def train(x, cond, latent_dim, epochs, beta=1.0, hidden_dim=128, free_bits=0.0,
          kl_warmup_frac=0.3, patience=300, cov_weight=1.0, seed=0,
          features=None, causal_graph=None, physics_weight=0.0, target_cond=None):
    """
    Fit a CVAE on standardized features `x` (n, d) with one-hot conditions
    `cond` (n, k). Loss is the VAE ELBO (reconstruction + beta-annealed KL with
    a free-bits floor) plus a covariance-matching term, matching the
    conventions of the other VAE engines. Returns the trained model.

    If `physics_weight > 0`, a structural-graph physics-consistency term is added
    (identical to the physics-informed VAE's): each edge's linear-Gaussian conditional
    is fit once on the training rows `x` (the seen conditions), then penalized on
    the reconstructions. The assumption — and what the transfer experiment tests
    — is that the fitted law is universal across conditions, so anchoring the
    decoder to it should carry the correct structure to an unseen condition.
    `features` and `causal_graph` must be given when physics_weight > 0.
    """
    torch.manual_seed(seed)
    x_t, c_t = torch.FloatTensor(x), torch.FloatTensor(cond)
    model = CVAE(x.shape[1], cond.shape[1], latent_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    edges = []
    tgt_c = None
    if physics_weight > 0 and causal_graph:
        # Fit the structural-edge conditionals on the seen (training) rows.
        edges = _fit_edges(np.asarray(x, dtype=np.float32), features, causal_graph)
        # If a target (unseen) condition is given, also constrain the physics of
        # samples generated for that condition each epoch: constraining
        # reconstructions of the seen condition alone barely moves the unseen
        # condition's decoder path.
        if target_cond is not None:
            tgt_c = torch.FloatTensor(np.tile(target_cond, (x.shape[0], 1)))

    warmup = max(1, int(epochs * kl_warmup_frac))
    best_loss, since_improved = float("inf"), 0

    for epoch in range(epochs):
        current_beta = beta * min(1.0, (epoch + 1) / warmup)
        recon, mu, logvar = model(x_t, c_t)

        recon_loss = ((recon - x_t) ** 2).mean()
        kl_per_dim = torch.clamp(0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar), min=free_bits)
        kl_loss = kl_per_dim.sum(dim=1).mean()
        cov_loss = ((_covariance(recon) - _covariance(x_t)) ** 2).mean()
        physics_loss = 0.0
        if edges:
            # Physics on the reconstructions of the seen rows...
            phys = _physics_loss(recon, edges)
            if tgt_c is not None:
                # ...and on freshly generated samples for the unseen target
                # condition (prior draw + target one-hot), so the law is imposed
                # where the model extrapolates.
                z = torch.randn(x_t.shape[0], latent_dim)
                gen_tgt = model.decoder(torch.cat([z, tgt_c], dim=1))
                phys = phys + _physics_loss(gen_tgt, edges)
            physics_loss = physics_weight * phys
        loss = recon_loss + current_beta * kl_loss + cov_weight * cov_loss + physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= warmup:
            if loss.item() < best_loss - 1e-4:
                best_loss, since_improved = loss.item(), 0
            else:
                since_improved += 1
                if since_improved >= patience:
                    break
    return model


def generate(model, cond_vector, n_samples, latent_dim, scaler):
    """
    Sample `n_samples` rows for a single condition. `cond_vector` is the (k,)
    one-hot for the target condition; latents are drawn from the N(0, I) prior
    (the right choice for extrapolating to an unseen condition — there's no
    aggregate posterior for a value with no training rows). `scaler` inverse-
    transforms outputs back to source units.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        c = torch.FloatTensor(np.tile(cond_vector, (n_samples, 1)))
        generated = model.decoder(torch.cat([z, c], dim=1)).numpy()
    return scaler.inverse_transform(generated)
