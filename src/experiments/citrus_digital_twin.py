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
                   citrus flagship's published mechanistic equations (chilling-injury
                   damage integral, Arrhenius colour kinetics, transpiration mass
                   loss; Onwude et al. 2022, 2024), with the rind non-negativity /
                   mass-balance conservation constraints; generates the outcome
                   distribution for the held-out (T, t)

The VFP conditions on the storage state and the commodity's published kinetic
equations (as conditioning inputs, so the mechanisms inform generation while the
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

# The digital twin conditions on the citrus flagship's published mechanistic
# equations (Onwude et al. 2022, 2024), each an Arrhenius rate integrated over
# storage time -> a rate x time prediction at constant temperature:
#   - ChillingInjury: damage integral  Omega(T,t) = kref*exp(-Ea/R (1/Tk-1/Tref))*t
#   - Colour:         Arrhenius quality kinetics  -dA/dt = k(T)*A^n
#   - MoistureLoss:   transpiration-driven mass loss (T/time form)
# Each equation's prediction is supplied to the generator as a physics-informed
# CONDITIONING input (standardized): generation is informed by the published
# equations while the fruit-to-fruit spread is learned from data. The two rate
# parameters of each equation are fitted to its published form on each training
# fold (the papers calibrate rather than tabulate them); see the Onwude papers.
KINETIC_OUTCOMES = ["ChillingInjury", "Colour", "MoistureLoss"]


def omega(T, t, kref, Ea, Tref_K):
    """Arrhenius rate x time at constant temperature (the published equation form)."""
    Tk = np.asarray(T) + 273.15
    return kref * np.exp(-Ea / R_GAS * (1.0 / Tk - 1.0 / Tref_K)) * np.asarray(t)


def fit_kinetics(df, outcome):
    """Fit (kref, Ea) of the published Arrhenius-rate x time form to `outcome` on a
    training fold. Returns (kref, Ea, Tref_K)."""
    T, t, y = df[COND[0]].values, df[COND[1]].values, df[outcome].values
    Tref_K = float((T + 273.15).mean())
    p0 = [max(abs(y).mean() / max(t.mean(), 1.0), 1e-3), -2.0e4]
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
            # Fit each published equation on this fold; each standardized prediction
            # becomes a physics-informed conditioning input.
            eqs = {}
            for c in KINETIC_OUTCOMES:
                kref, Ea, Tref_K = fit_kinetics(tr, c)
                pred = omega(tr[COND[0]].values, tr[COND[1]].values, kref, Ea, Tref_K)
                eqs[c] = (kref, Ea, Tref_K, float(pred.mean()), float(pred.std()) + 1e-9)

            def cvec(t_, dd_):
                base = [(t_ - tm) / ts, (dd_ - dm) / ds, (t_ * dd_ - xm) / xs]
                for c in KINETIC_OUTCOMES:
                    kref, Ea, Tref_K, mu, sd = eqs[c]
                    base.append((omega(t_, dd_, kref, Ea, Tref_K) - mu) / sd)
                return np.array(base)

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
    print(f"model: conditional VAE conditioned on (storage temperature, duration, and the "
          f"published chilling-injury, colour, and moisture kinetic equations) with rind "
          f"conservation constraints")
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
