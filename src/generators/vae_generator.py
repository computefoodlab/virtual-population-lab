import logging

import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger("vp-lab")


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
        # Clamp logvar to a safe range before exponentiating: on very small
        # datasets an unconstrained logvar can run away, making
        # std = exp(0.5*logvar) explode and the whole population diverge. The
        # bounds are wide enough never to bind on a well-behaved fit.
        logvar = torch.clamp(z[:, self.latent_dim:], -8.0, 8.0)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        latent = mu + eps * std

        return self.decoder(latent), mu, logvar


def _covariance(batch):
    centered = batch - batch.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / (batch.shape[0] - 1)


def generate(
    x,
    features,
    latent_dim,
    epochs,
    beta,
    n_samples=1000,
    scaler=None,
    kl_warmup_frac=0.3,
    use_minibatch=True,
    batch_size=128,
    cov_weight=1.0,
    patience=200,
    hidden_dim=128,
    dropout=0.0,
    free_bits=0.0,
):
    """
    Trains on `x` (expected to already be reasonably scaled) and samples
    `n_samples` new rows from the prior. Pass a fitted `scaler` only if `x`
    needs to be inverse-transformed back to source units afterwards —
    leave it None when the source data is already standardized, since
    there's no original unscaled space to return to.

    Loss is the standard VAE ELBO (reconstruction MSE plus the analytic KL
    divergence between the approximate posterior N(mu, sigma^2) and the
    standard normal prior, weighted by `beta`), plus a covariance-matching
    term weighted by `cov_weight`: the squared difference between the
    reconstructed batch's covariance matrix and the real batch's. The ELBO
    alone has no direct incentive to get cross-feature correlations right — it
    only rewards accurate per-point reconstruction — so this term applies
    direct pressure toward preserving joint structure. `beta` is linearly
    annealed from 0 up to its target value over the first `kl_warmup_frac` of
    training — without this, the KL term can dominate before reconstruction has
    learned anything, collapsing the model onto near-zero variance (i.e.
    generated samples clustering tightly around the mean).

    With `use_minibatch=True`, each epoch shuffles the data and takes one
    gradient step per `batch_size` chunk instead of one step over the whole
    dataset — the added stochastic noise can help escape a plateaued loss
    that full-batch descent gets stuck in. Set `use_minibatch=False` for
    very small datasets, where a mini-batch may be too small to estimate a
    stable gradient at all.

    Training stops early once the loss hasn't improved for `patience`
    epochs, checked only after the KL warmup finishes — beta is still
    rising during warmup, so the loss naturally shifts then regardless of
    real convergence, and checking during that window would trigger a
    spurious early stop.

    `hidden_dim` and `dropout` control model capacity — the defaults (128,
    0.0) fit a few-thousand-row dataset comfortably, but on a small dataset
    (dozens to low hundreds of rows) that many parameters can memorize
    training noise instead of generalizing. Shrink `hidden_dim` and/or add
    dropout (e.g. 0.1-0.3) if the generalization check shows a large
    train/test gap.

    `free_bits` floors the KL cost per latent dimension: once a dimension's
    KL drops below this many nats, it stops contributing gradient to the KL
    term (though reconstruction can still make it informative). Without
    this (free_bits=0.0), KL annealing alone can still let every dimension
    collapse fully to the prior on some datasets — recon_loss plateauing
    at ~1.0 on standardized data (i.e. no better than always predicting the
    mean) with KL near zero is the signature to watch for. Try 0.5-2.0 if
    that happens.
    """
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
        total_loss = total_recon = total_kl = total_cov = 0.0
        n_batches = 0

        for start in range(0, n, step_size):
            batch = x_tensor[perm[start:start + step_size]]

            recon, mu, logvar = vae(batch)

            recon_loss = ((recon - batch) ** 2).mean()

            kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # always >= 0
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

            cov_loss = ((_covariance(recon) - _covariance(batch)) ** 2).mean()

            loss = recon_loss + current_beta * kl_loss + cov_weight * cov_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_cov += cov_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if epoch % log_every == 0 or epoch == epochs - 1:
            log.info(
                "  epoch %s/%d — loss %7.4f (recon %7.4f, kl %7.4f, cov %7.4f, beta %5.3f)",
                str(epoch + 1).rjust(epoch_width), epochs,
                avg_loss, total_recon / n_batches,
                total_kl / n_batches, total_cov / n_batches, current_beta,
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
        z = torch.randn(n_samples, latent_dim)

        generated = vae.decoder(z).numpy()

    if scaler is not None:
        generated = scaler.inverse_transform(generated)

    return pd.DataFrame(
        generated,
        columns=features
    )
