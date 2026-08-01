"""
Same-model near/far transfer evaluation (named "leave-one-group-out" for its
grouping, but with a same-model design).

Tests whether the model can generate a realistic population for a condition it
saw little or none of, and whether its uncertainty widens when it does.

For each group big enough to hold part of it out (a "train region"):

  1. Train ONE conditional-VAE ensemble on a train split of that region only.
  2. NEAR (interpolation): generate that region's condition, score against the
     region's *held-out same-region* rows. Unseen individuals, seen condition.
  3. FAR  (extrapolation): with the SAME ensemble, generate each *other* region's
     condition and score against that region's real rows. Unseen condition.

Because NEAR and FAR come from the same trained models, with only the queried
condition changing, the disagreement difference isolates the unseen-condition
effect. Reported per train region plus a MEAN row; the headline is FAR-vs-NEAR
ensemble disagreement.

Metrics vs the scored real rows: correlation distance + mean KS (fidelity),
coverage@90 (do the intervals still cover real?), and ensemble disagreement
(spread across members' per-feature means, per-global-std = epistemic
uncertainty; expect FAR > NEAR).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation.coverage import _central_coverage
from src.evaluation.evaluate import correlation_euclidean_dist, marginal_ks_table
from src.generators import conditional_vae_generator as cvae

FIGURE_DPI = 400

# Same-model near/far hyperparameters, shared across every dataset.
K_ENSEMBLE = 5      # ensemble members — their disagreement is the uncertainty signal
N_GEN = 400         # samples per member per queried condition
EPOCHS = 1200
PATIENCE = 250
LATENT_DIM = 6
HIDDEN_DIM = 128
FREE_BITS = 2.0
BETA = 1.0
NEAR_TEST_FRAC = 0.35   # fraction of a train region held out for the NEAR test
# A region must have at least this many rows to be a "train region" (needs a
# train split plus a held-out NEAR test). Smaller groups still serve as FAR
# targets for the larger train regions.
MIN_TRAIN_GROUP = 16
SEED = 42


def _onehot(index, k):
    v = np.zeros(k, dtype=np.float32)
    v[index] = 1.0
    return v


def _train_ensemble(train_df, features, group_col, groups):
    """Train a K-member CVAE ensemble on train_df (conditioned on its group).
    Returns (models, fitted scaler) so the SAME models can then be queried for
    any condition (near = own group, far = others)."""
    scaler = StandardScaler().fit(train_df[features])
    x = scaler.transform(train_df[features])
    cond = np.stack([_onehot(groups.index(v), len(groups)) for v in train_df[group_col]])
    models = [
        cvae.train(x, cond, latent_dim=LATENT_DIM, epochs=EPOCHS, beta=BETA,
                   hidden_dim=HIDDEN_DIM, free_bits=FREE_BITS, patience=PATIENCE, seed=SEED + 100 + k)
        for k in range(K_ENSEMBLE)
    ]
    return models, scaler


def _generate(models, scaler, target_idx, groups, features):
    """Generate the target condition from every ensemble member. Returns
    (pooled_df, per_member_means[K, n_features])."""
    target_vec = _onehot(target_idx, len(groups))
    pooled, member_means = [], []
    for m in models:
        gen = cvae.generate(m, target_vec, N_GEN, LATENT_DIM, scaler)
        pooled.append(gen)
        member_means.append(gen.mean(axis=0))
    return pd.DataFrame(np.vstack(pooled), columns=features), np.stack(member_means)


def _ecdf(gen, y):
    """P(gen <= y) for each y, from the generated population `gen`."""
    gs = np.sort(gen)
    return np.searchsorted(gs, y, side="right") / len(gs)


def _conformal_shat(gen, y_cal, level=0.90):
    """Split-conformal half-width (in probability units): the finite-sample
    quantile of the calibration reals' distance from the generated median, so
    that [F^-1(0.5-shat), F^-1(0.5+shat)] contains ~`level` of them. Capped at
    0.5 (the full generated range). This recalibrates an interval that is too
    narrow (small-n VAE shrinkage) or too wide (ensemble pooling)."""
    s = np.abs(_ecdf(gen, y_cal) - 0.5)
    m = len(s)
    if m == 0:
        return (1 - level) / 2 + level / 2  # degenerate; full-ish interval
    k = min(max(int(np.ceil((m + 1) * level)) - 1, 0), m - 1)
    return min(float(np.sort(s)[k]), 0.5)


def _conf_cov(gen, y_eval, shat):
    """Coverage of the conformal interval [F^-1(0.5-shat), F^-1(0.5+shat)] on y_eval."""
    lo, hi = np.quantile(gen, 0.5 - shat), np.quantile(gen, 0.5 + shat)
    return float(np.mean((y_eval >= lo) & (y_eval <= hi)))


# coverage_*_conf = conformal-recalibrated coverage: the per-feature interval
# width is calibrated on the NEAR held-out reals (target 0.90) and the same width
# is transferred off-support (FAR). NEAR conf should reach ~0.90 in-region; the
# remaining FAR conf gap is genuine distribution shift.
_SCHEMA = ["group", "n", "corr_far", "corr_near", "ks_far", "ks_near",
           "coverage_far", "coverage_near", "coverage_far_conf", "coverage_near_conf",
           "disagreement_far", "disagreement_near"]


def leave_one_group_out(df, all_features, group_col):
    """
    Same-model near/far over every train region (a group big enough to split).
    Returns a DataFrame with one row per train region plus a final 'MEAN' row,
    with columns corr/ks/coverage/disagreement suffixed _far/_near: near =
    held-out same region, far = other regions from the same ensemble.
    """
    features = [f for f in all_features if f != group_col]
    all_groups = sorted(df[group_col].unique())
    sizes = df[group_col].value_counts()

    global_std = df[features].std().values
    global_std = np.where(global_std > 1e-9, global_std, 1.0)

    trainable = [g for g in all_groups if sizes[g] >= MIN_TRAIN_GROUP]
    if not trainable or len(all_groups) < 2:
        import logging
        logging.getLogger("vp-lab").warning(
            "  near/far: need >=2 groups and at least one with >=%d rows to split "
            "(largest = %d). Skipping transfer for this dataset.",
            MIN_TRAIN_GROUP, int(sizes.max()))
        return pd.DataFrame(columns=_SCHEMA)

    skipped = len(all_groups) - len(trainable)
    if skipped:
        import logging
        logging.getLogger("vp-lab").info(
            "  near/far: %d train-region(s) with >=%d rows; %d smaller group(s) used "
            "only as FAR targets.", len(trainable), MIN_TRAIN_GROUP, skipped)

    rows = []
    for tg in trainable:
        reg = df[df[group_col] == tg]
        tr, te = train_test_split(reg, test_size=NEAR_TEST_FRAC, random_state=SEED)
        models, scaler = _train_ensemble(tr, features, group_col, all_groups)

        def _metrics(target_idx, real_df):
            gen, means = _generate(models, scaler, target_idx, all_groups, features)
            corr = correlation_euclidean_dist(real_df, gen, features)
            ks = marginal_ks_table(real_df, {"g": gen}, features)["ks_stat"].mean()
            cov = np.mean([_central_coverage(gen[f].values, real_df[f].values, 0.90) for f in features])
            dis = float(np.mean(means.std(axis=0) / global_std))
            return gen, corr, ks, cov, dis

        te_df = te[features].reset_index(drop=True)
        gen_near, cn, kn, vn, dn = _metrics(all_groups.index(tg), te_df)
        # Conformal width calibrated per feature on the NEAR held-out reals, then
        # reused off-support. NEAR conf is ~self-calibrated to 0.90; FAR conf
        # applies the SAME width to the shifted region.
        shat = {f: _conformal_shat(gen_near[f].values, te_df[f].values) for f in features}
        vconf_n = np.mean([_conf_cov(gen_near[f].values, te_df[f].values, shat[f]) for f in features])

        far = []
        for og in all_groups:
            if og == tg:
                continue
            real_far = df[df[group_col] == og][features].reset_index(drop=True)
            gen_far, cf, kf, vf, dff = _metrics(all_groups.index(og), real_far)
            vconf_f = np.mean([_conf_cov(gen_far[f].values, real_far[f].values, shat[f]) for f in features])
            far.append((cf, kf, vf, vconf_f, dff))
        cf, kf, vf, vconf_f, dff = np.mean(far, axis=0)

        rows.append({
            "group": tg, "n": len(te),
            "corr_far": cf, "corr_near": cn, "ks_far": kf, "ks_near": kn,
            "coverage_far": vf, "coverage_near": vn,
            "coverage_far_conf": vconf_f, "coverage_near_conf": vconf_n,
            "disagreement_far": dff, "disagreement_near": dn,
        })

    result = pd.DataFrame(rows)
    mean_row = {"group": "MEAN", "n": np.nan}
    for col in result.columns:
        if col not in ("group", "n"):
            mean_row[col] = result[col].mean()
    return pd.concat([result, pd.DataFrame([mean_row])], ignore_index=True)


def save_loro_plot(loro_df, group_col, path):
    """Two panels: ensemble disagreement (epistemic uncertainty) and correlation
    distance (fidelity), FAR vs NEAR per train region. The disagreement panel is
    the headline — FAR bars taller than NEAR = uncertainty widens off-support."""
    groups = loro_df[loro_df["group"] != "MEAN"]
    labels = groups["group"].astype(str).tolist()
    xs = np.arange(len(labels))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.bar(xs - width / 2, groups["disagreement_far"], width, label="FAR (different region)", color="#D55E00")
    ax1.bar(xs + width / 2, groups["disagreement_near"], width, label="NEAR (held-out same region)", color="#0072B2")
    ax1.set_title("Epistemic uncertainty (ensemble disagreement)\nhigher for a different region = knows what it doesn't know")
    ax1.set_ylabel("Ensemble disagreement (per-feature-std units)")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_xlabel(f"train region ({group_col})"); ax1.legend()

    ax2.bar(xs - width / 2, groups["corr_far"], width, label="FAR (different region)", color="#D55E00")
    ax2.bar(xs + width / 2, groups["corr_near"], width, label="NEAR (held-out same region)", color="#0072B2")
    ax2.set_title("Fidelity (correlation distance, lower = better)\nextrapolation costs fidelity")
    ax2.set_ylabel("Correlation distance to real group")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels)
    ax2.set_xlabel(f"train region ({group_col})"); ax2.legend()

    fig.suptitle(f"Same-model near/far transfer by {group_col} — conditional-VAE ensemble", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
