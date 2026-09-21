"""
Citrus digital-twin decision: chilling-injury at-risk fraction for a storage
condition (leave-one-condition-out).

The decision an operator faces is "what fraction of this batch will exceed a
chilling-injury tolerance after storage at temperature T for t days?" - a
*distribution* question, not a single number. This tests whether the generated
POPULATION answers it better than a single representative value or a bootstrap of
the batches already seen.

Setup: the citrus flagship spans 12 storage conditions (3 temperatures x 4
durations, ~20 fruit each). Each condition is held out in turn; the model is fit
on the other 11 and asked to produce the fruit for the held-out (T, t). The
at-risk fraction (CI above a tolerance) is predicted and scored by absolute error
against the true held-out fraction, and the tolerance is swept and reported at
each threshold.

  average          the pooled mean CI as a point -> a degenerate 0/1 answer that
                   cannot express a fraction
  pooled bootstrap resample CI from the seen conditions
  VFP (PI-VFP)     conditional VAE conditioned on temperature, duration and the
                   published chilling-injury damage integral Omega(T, t), with the
                   rind non-negativity / mass-balance conservation constraints;
                   generates the outcome distribution for the held-out (T, t)

The VFP conditions on the storage state and the published chilling-injury
equation (as a conditioning input, so the mechanism informs generation while the
fruit-to-fruit spread is learned from data) and imposes the rind conservation
constraints.

Run:  python -m src.experiments.citrus_digital_twin
(expects the citrus raw file at data/cleaned-citrus-exp6.csv - authors'
unpublished data, available on request.)
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from src.generators import mechanistic_cvae as mech

RAW = "data/cleaned-citrus-exp6.csv"
FEATURES = ["RindFresh", "RindDry", "MoistureLoss", "ChillingInjury", "Colour"]
COND = ["StorageTemp", "Duration"]
CI_IDX = FEATURES.index("ChillingInjury")
NONNEG = list(range(len(FEATURES)))          # all five quantities are non-negative
THRESHOLDS = [10, 15, 20, 25, 30]            # %CI reject tolerances; reported per level
SEEDS = [41, 42, 43]
N_GEN = 1000
LATENT = 4
EPOCHS = 500
OUT = "results/_summary"

R_GAS = 8.314  # J/mol/K

# Published chilling-injury kinetics (Onwude et al. 2024): the damage integral at a
# constant storage temperature reduces to  Omega(T, t) = kref * exp(-Ea/R (1/Tk -
# 1/Tref)) * t. Omega is supplied to the generator as an extra CONDITIONING input
# (physics-informed conditioning): generation is informed by the published
# equation while the fruit-to-fruit spread is still learned from data. By default
# the two rate parameters are fitted to the published form on each training fold;
# set USE_PUBLISHED_PARAMS = True and fill PUBLISHED to impose the paper's values.
USE_PUBLISHED_PARAMS = False
PUBLISHED = {"kref": None, "Ea": None, "Tref_C": None}


def omega(T, t, kref, Ea, Tref_K):
    """Published chilling-injury damage integral at constant temperature."""
    Tk = np.asarray(T) + 273.15
    return kref * np.exp(-Ea / R_GAS * (1.0 / Tk - 1.0 / Tref_K)) * np.asarray(t)


def fit_kinetics(df):
    """Estimate (kref, Ea) of the published damage-integral form on a training fold,
    or use the paper's values if USE_PUBLISHED_PARAMS. Returns (kref, Ea, Tref_K)."""
    T, t, y = df[COND[0]].values, df[COND[1]].values, df["ChillingInjury"].values
    Tref_K = float((T + 273.15).mean())
    if USE_PUBLISHED_PARAMS and all(PUBLISHED[k] is not None for k in ("kref", "Ea", "Tref_C")):
        return PUBLISHED["kref"], PUBLISHED["Ea"], PUBLISHED["Tref_C"] + 273.15
    p0 = [max(y.mean() / max(t.mean(), 1.0), 1e-3), -2.0e4]
    try:
        popt, _ = curve_fit(lambda X, kref, Ea: omega(X[0], X[1], kref, Ea, Tref_K),
                            (T, t), y, p0=p0, maxfev=20000)
        return float(popt[0]), float(popt[1]), Tref_K
    except Exception:
        return p0[0], p0[1], Tref_K


def snap_to_support(vals, support):
    """Map generated values onto the real observed support (nearest observed value).
    Preserves discreteness and the zero atom of a zero-inflated / binned feature
    (e.g. chilling injury) while leaving the conditional model's per-condition
    frequencies intact - the condition-preserving analog of the PI-VAE's
    empirical-support marginal calibration."""
    support = np.sort(np.unique(np.asarray(support)))
    idx = np.clip(np.searchsorted(support, vals), 0, len(support) - 1)
    left = np.clip(idx - 1, 0, len(support) - 1)
    take_left = np.abs(vals - support[left]) <= np.abs(vals - support[idx])
    return np.where(take_left, support[left], support[idx])


def at_risk(ci_values, thr):
    return float(np.mean(np.asarray(ci_values) > thr))


def run():
    d = pd.read_csv(RAW)
    conds = sorted(d.groupby(COND).size().index.tolist())
    tm, ts = d[COND[0]].mean(), d[COND[0]].std()
    dm, ds = d[COND[1]].mean(), d[COND[1]].std()
    xm, xs = (d[COND[0]] * d[COND[1]]).mean(), (d[COND[0]] * d[COND[1]]).std()

    def run_seed(seed):
        rows = []
        for T, Dur in conds:
            te = d[(d[COND[0]] == T) & (d[COND[1]] == Dur)]
            tr = d[~((d[COND[0]] == T) & (d[COND[1]] == Dur))]
            # Fit the published damage-integral form on this fold; the standardized
            # Omega(T, t) becomes the fourth conditioning input.
            kref, Ea, Tref_K = fit_kinetics(tr)
            om_tr = omega(tr[COND[0]].values, tr[COND[1]].values, kref, Ea, Tref_K)
            om_mu, om_sd = float(om_tr.mean()), float(om_tr.std()) + 1e-9

            def cvec(t_, dd_):
                return np.array([(t_ - tm) / ts, (dd_ - dm) / ds, (t_ * dd_ - xm) / xs,
                                 (omega(t_, dd_, kref, Ea, Tref_K) - om_mu) / om_sd])

            sc = StandardScaler().fit(tr[FEATURES])
            cond = np.array([cvec(t, dd) for t, dd in zip(tr[COND[0]], tr[COND[1]])])
            np.random.seed(seed); torch.manual_seed(seed)
            m = mech.train(sc.transform(tr[FEATURES]), cond, sc, latent_dim=LATENT,
                           epochs=EPOCHS, beta=1.0, hidden_dim=128, free_bits=1.0,
                           patience=120, cov_weight=1.0, seed=seed, nonneg_idx=NONNEG,
                           lambda_cons=5.0, lambda_kin=0.0)
            g = mech.generate(m, cvec(T, Dur), N_GEN, LATENT, sc, nonneg_idx=NONNEG)[:, CI_IDX]
            g = snap_to_support(g, tr["ChillingInjury"].values)  # keep the zero atom / discrete CI levels
            rng = np.random.RandomState(seed)
            boot = rng.choice(tr["ChillingInjury"].values, size=N_GEN, replace=True)
            avg_ci = tr["ChillingInjury"].mean()
            for thr in THRESHOLDS:
                true = at_risk(te["ChillingInjury"].values, thr)
                rows.append((thr, abs((1.0 if avg_ci > thr else 0.0) - true),
                             abs(at_risk(boot, thr) - true), abs(at_risk(g, thr) - true)))
        return pd.DataFrame(rows, columns=["thr", "avg", "boot", "vfp"])

    per_seed = [run_seed(s) for s in SEEDS]
    allr = pd.concat(per_seed)
    print(f"\n=== Citrus digital-twin decision (leave-one-condition-out, "
          f"{len(SEEDS)} seeds, {len(conds)} conditions) ===")
    print(f"model: conditional VAE conditioned on (storage temperature, duration, "
          f"published chilling-injury damage integral) with rind conservation constraints")
    print(f"at-risk-fraction MAE per CI tolerance (lower is better):\n")
    print(f"{'CI thr %':>9}{'avg':>10}{'bootstrap':>12}{'VFP':>10}")
    records = []
    for thr in THRESHOLDS:
        sub = allr[allr.thr == thr]
        mae = {c: sub[c].mean() for c in ("avg", "boot", "vfp")}
        print(f"{thr:>9}{mae['avg']:>10.3f}{mae['boot']:>12.3f}{mae['vfp']:>10.3f}")
        records.append({"ci_threshold": thr, "avg_mae": round(mae["avg"], 4),
                        "bootstrap_mae": round(mae["boot"], 4), "vfp_mae": round(mae["vfp"], 4),
                        "n": len(sub)})
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(records).to_csv(f"{OUT}/citrus_digital_twin.csv", index=False)
    print(f"\nwrote: {OUT}/citrus_digital_twin.csv")


if __name__ == "__main__":
    run()
