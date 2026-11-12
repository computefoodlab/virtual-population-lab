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
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import betaln

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
# equations, each an Arrhenius rate integrated over storage time -> a rate x time
# prediction at constant temperature:
#   - ChillingInjury: damage integral  Omega(T,t) = k0*exp(-Ea/R (1/Tk-1/Tref))*t
#   - Colour:         Arrhenius quality kinetics  -dA/dt = k(T)*A^n
#   - MoistureLoss:   transpiration-driven; T-dependence via Clausius-Clapeyron
# Each prediction is supplied to the generator as a CONDITIONING input
# (standardized): generation is informed by the published equations while the
# fruit-to-fruit spread is learned from data. The activation energies are IMPOSED
# from published sources and physical constants, NOT fitted to our data; k0 = 1
# since it cancels under standardization, so only the published Ea enters.
PUB_EA = {
    "ChillingInjury": -147754.0,  # Shrivastava et al. 2022 (Nature Food)
    "Colour":           44585.0,  # Onwude et al. 2022 (Resour. Conserv. Recycl., Table 2)
    "MoistureLoss":     44000.0,  # latent heat of vaporisation of water (Clausius-Clapeyron)
}
KINETIC_OUTCOMES = ["ChillingInjury", "Colour", "MoistureLoss"]


def omega(T, t, kref, Ea, Tref_K):
    """Arrhenius rate x time at constant temperature (the published equation form)."""
    Tk = np.asarray(T) + 273.15
    return kref * np.exp(-Ea / R_GAS * (1.0 / Tk - 1.0 / Tref_K)) * np.asarray(t)


def imposed_kinetics(df, outcome):
    """Return (k0, Ea, Tref_K) with Ea IMPOSED from the published value for `outcome`
    (not fitted to our data). k0 = 1 because it cancels when the conditioning input is
    standardized, so only the published activation energy shapes the signal. Tref is the
    training-fold mean temperature and only sets the reference point of the rate."""
    Tref_K = float((df[COND[0]].values + 273.15).mean())
    return 1.0, PUB_EA[outcome], Tref_K


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


def glm_at_risk(tr, T, Dur, thr, stats):
    """Condition-aware non-generative baseline: a binomial GLM for P(CI > thr) in
    standardized (temperature, duration, temperature x duration). Unlike pooling
    baselines it extrapolates a held-out condition from its covariates."""
    tm, ts, dm, ds, xm, xs = stats
    y = (tr["ChillingInjury"].values > thr).astype(float)
    if y.sum() == 0:
        return 0.0
    if y.sum() == len(y):
        return 1.0
    X = sm.add_constant(np.column_stack([
        (tr[COND[0]] - tm) / ts, (tr[COND[1]] - dm) / ds,
        (tr[COND[0]] * tr[COND[1]] - xm) / xs]), has_constant="add")
    xq = np.array([[1.0, (T - tm) / ts, (Dur - dm) / ds, (T * Dur - xm) / xs]])
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200)
        return float(m.predict(xq)[0])
    except Exception:
        return float(y.mean())


def eb_at_risk(tr, conds_tr, thr):
    """Empirical-Bayes Beta-Binomial prior mean, pooling the seen conditions. For a
    held-out condition with no local data this is the partial-pooling (mixed-effects)
    fallback, which cannot use the held condition's covariates."""
    counts = []
    for a, b in conds_tr:
        s = tr[(tr[COND[0]] == a) & (tr[COND[1]] == b)]["ChillingInjury"].values
        counts.append(((s > thr).sum(), len(s)))
    k = np.array([c[0] for c in counts], float)
    n = np.array([c[1] for c in counts], float)
    if k.sum() == 0:
        return 0.0
    if (k == n).all():
        return 1.0
    try:
        r = minimize(lambda p: -np.sum(betaln(k + np.exp(p[0]), n - k + np.exp(p[1]))
                                       - betaln(np.exp(p[0]), np.exp(p[1]))),
                     [0.0, 0.0], method="Nelder-Mead")
        a, b = np.exp(r.x)
        return float(a / (a + b))
    except Exception:
        return float(k.sum() / n.sum())


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
            # Impose each published activation energy; each standardized prediction
            # becomes a conditioning input.
            eqs = {}
            for c in KINETIC_OUTCOMES:
                kref, Ea, Tref_K = imposed_kinetics(tr, c)
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
            conds_tr = [c for c in conds if c != (T, Dur)]
            stats = (tm, ts, dm, ds, xm, xs)
            for thr in THRESHOLDS:
                true = at_risk(te["ChillingInjury"].values, thr)
                single_frac = 1.0 if avg_ci > thr else 0.0
                boot_frac = at_risk(boot, thr)
                vfp_frac = at_risk(g, thr)
                glm_frac = glm_at_risk(tr, T, Dur, thr, stats)          # condition-aware baseline
                eb_frac = eb_at_risk(tr, conds_tr, thr)                 # empirical-Bayes pooling
                shrink_frac = at_risk(tr["ChillingInjury"].values, thr)  # pooled-fraction shrinkage
                rows.append((thr, float(T), float(Dur), true,
                             single_frac, boot_frac, glm_frac, eb_frac, shrink_frac, vfp_frac,
                             abs(single_frac - true), abs(boot_frac - true),
                             abs(glm_frac - true), abs(eb_frac - true),
                             abs(shrink_frac - true), abs(vfp_frac - true)))
        return pd.DataFrame(rows, columns=["thr", "T", "Dur", "true",
                                           "single_frac", "boot_frac", "glm_frac", "eb_frac",
                                           "shrink_frac", "vfp_frac",
                                           "avg", "boot", "glm", "eb", "shrink", "vfp"])

    per_seed = [run_seed(s) for s in SEEDS]
    allr = pd.concat(per_seed)
    print(f"\n=== Citrus digital-twin decision (leave-one-condition-out, "
          f"{len(SEEDS)} seeds, {len(conds)} conditions) ===")
    print(f"model: conditional VAE conditioned on (storage temperature, duration, and the "
          f"published chilling-injury, colour, and moisture kinetic equations) with rind "
          f"conservation constraints")

    # (1) at-risk-fraction absolute error per tolerance, reported as mean +/- s.d.
    # over the seed x condition folds (n = seeds x conditions), so the table shows
    # variability, not just a point error.
    print(f"\nat-risk-fraction absolute error per CI tolerance (mean +/- s.d., lower is better):\n")
    cols = [("single", "avg"), ("bootstrap", "boot"), ("GLM", "glm"),
            ("EmpBayes", "eb"), ("Shrink", "shrink"), ("VFP", "vfp")]
    print("CI thr % | " + " | ".join(f"{name:>12}" for name, _ in cols))
    records = []
    for thr in THRESHOLDS:
        sub = allr[allr.thr == thr]
        rec = {"ci_threshold": thr}
        cells = []
        for name, c in cols:
            m, s = sub[c].mean(), sub[c].std()
            cells.append(f"{m:.3f}+/-{s:.3f}")
            rec[f"{c}_mae"] = round(m, 4)
            rec[f"{c}_sd"] = round(s, 4)
        rec["n"] = len(sub)
        print(f"{thr:>8} | " + " | ".join(f"{x:>12}" for x in cells))
        records.append(rec)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(records).to_csv(f"{OUT}/citrus_digital_twin.csv", index=False)
    print(f"\nwrote: {OUT}/citrus_digital_twin.csv")

    # (1b) Table S9b: worst-case (max over the twelve conditions) at-risk-fraction
    # error, seed-averaged per condition. For a reject decision the worst case over
    # conditions matters more than the average.
    print("\nworst-case at-risk-fraction error over the 12 conditions "
          "(seed-averaged per condition; lower is better):")
    print("CI thr % | " + " | ".join(f"{name:>12}" for name, _ in cols))
    worst = []
    for thr in THRESHOLDS:
        sub = allr[allr.thr == thr]
        per_cond = sub.groupby(["T", "Dur"]).agg(**{c: (c, "mean") for _, c in cols}).reset_index()
        rec = {"ci_threshold": thr}
        cells = []
        for name, c in cols:
            mx = float(per_cond[c].max())
            cells.append(f"{mx:.3f}")
            rec[f"{c}_worst"] = round(mx, 4)
        print(f"{thr:>8} | " + " | ".join(f"{x:>12}" for x in cells))
        worst.append(rec)
    pd.DataFrame(worst).to_csv(f"{OUT}/citrus_digital_twin_worstcase.csv", index=False)
    print(f"wrote: {OUT}/citrus_digital_twin_worstcase.csv")

    # (2) operational consequence: the ACTUAL predicted at-risk fraction of each
    # method against the true fraction, per storage condition, at a representative
    # tolerance -- so the table reads "true 31%, single 0%, bootstrap 24%, VFP 29%",
    # rather than merely "VFP has the lowest error".
    pc = (allr
          .groupby(["thr", "T", "Dur"])
          .agg(true_pct=("true", "mean"), single_pct=("single_frac", "mean"),
               bootstrap_pct=("boot_frac", "mean"), glm_pct=("glm_frac", "mean"),
               vfp_pct=("vfp_frac", "mean"))
          .reset_index())
    for c in ("true_pct", "single_pct", "bootstrap_pct", "glm_pct", "vfp_pct"):
        pc[c] = (pc[c] * 100).round(1)
    pc["single_abs_err_pct"] = (pc["single_pct"] - pc["true_pct"]).abs().round(1)
    pc["bootstrap_abs_err_pct"] = (pc["bootstrap_pct"] - pc["true_pct"]).abs().round(1)
    pc["glm_abs_err_pct"] = (pc["glm_pct"] - pc["true_pct"]).abs().round(1)
    pc["vfp_abs_err_pct"] = (pc["vfp_pct"] - pc["true_pct"]).abs().round(1)
    # per-condition winner (smallest absolute error) at each threshold
    err = pc[["single_abs_err_pct", "bootstrap_abs_err_pct", "glm_abs_err_pct", "vfp_abs_err_pct"]].values
    names = np.array(["single", "bootstrap", "glm", "vfp"])
    pc["closest"] = names[err.argmin(axis=1)]
    for thr in THRESHOLDS:
        sub = pc[pc.thr == thr]
        wins = sub["closest"].value_counts().to_dict()
        risky = sub[sub.true_pct >= 20]
        wins_risky = risky["closest"].value_counts().to_dict()
        print(f"\n[{thr}% tolerance] per-condition winner counts (of {len(sub)}): {wins}")
        print(f"   on the {len(risky)} conditions with true risk >= 20%: {wins_risky}")
    pc.to_csv(f"{OUT}/citrus_digital_twin_percondition.csv", index=False)
    print(f"\nwrote: {OUT}/citrus_digital_twin_percondition.csv (all thresholds)")

    # (3) paired significance of the VFP improvement. The 12 storage conditions are
    # the experimental unit, so we seed-average each condition's absolute error and
    # test the VFP against the bootstrap and the single value with a paired Wilcoxon
    # signed-rank test, reporting the mean error reduction with a 95% CI.
    from scipy.stats import wilcoxon, t as tdist
    def ci95(x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return float(x.mean()), float(x.mean())
        se = x.std(ddof=1) / np.sqrt(len(x))
        h = tdist.ppf(0.975, len(x) - 1) * se
        return float(x.mean() - h), float(x.mean() + h)
    print("\npaired significance on the 12 conditions (seed-averaged; VFP vs each baseline):")
    sig = []
    for thr in THRESHOLDS:
        sub = allr[allr.thr == thr]
        pc_err = (sub.groupby(["T", "Dur"])
                  .agg(avg=("avg", "mean"), boot=("boot", "mean"), glm=("glm", "mean"),
                       vfp=("vfp", "mean"))
                  .reset_index())
        v = pc_err["vfp"].values
        for base_name, base_col in [("GLM", "glm"), ("bootstrap", "boot"), ("single", "avg")]:
            b = pc_err[base_col].values
            diff = b - v  # positive = VFP better (lower error)
            lo, hi = ci95(diff)
            try:
                _, p = wilcoxon(v, b)
            except ValueError:
                p = float("nan")
            print(f"  thr {thr:>3}% vs {base_name:>9}: mean error reduction "
                  f"{diff.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), "
                  f"Wilcoxon p={p:.4f}, n={len(v)}")
            sig.append({"ci_threshold": thr, "comparison": f"VFP vs {base_name}",
                        "mean_error_reduction": round(float(diff.mean()), 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                        "wilcoxon_p": (round(float(p), 4) if p == p else None),
                        "n": int(len(v))})
    pd.DataFrame(sig).to_csv(f"{OUT}/citrus_digital_twin_significance.csv", index=False)
    print(f"\nwrote: {OUT}/citrus_digital_twin_significance.csv")


def run_extrapolation():
    """Table S9d: leave-one-temperature-out extrapolation. The model is trained on two
    storage temperatures and asked to predict the held-out temperature's conditions,
    which the condition-aware GLM cannot fit locally. Only three temperature folds, so
    this is a directional check, not a powered significance test."""
    d = pd.read_csv(RAW)
    conds = sorted(d.groupby(COND).size().index.tolist())
    temps = sorted(d[COND[0]].unique())
    tm, ts = d[COND[0]].mean(), d[COND[0]].std()
    dm, ds = d[COND[1]].mean(), d[COND[1]].std()
    xm, xs = (d[COND[0]] * d[COND[1]]).mean(), (d[COND[0]] * d[COND[1]]).std()
    stats = (tm, ts, dm, ds, xm, xs)
    rows = []
    for tp in temps:
        tr = d[d[COND[0]] != tp]
        held = [c for c in conds if c[0] == tp]
        eqs = {}
        for c in KINETIC_OUTCOMES:
            kref, Ea, Tref_K = imposed_kinetics(tr, c)
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
        gen = {(T, Dur): {thr: [] for thr in THRESHOLDS} for (T, Dur) in held}
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            m = mech.train(sc.transform(tr[FEATURES]), cond, sc, latent_dim=LATENT,
                           epochs=EPOCHS, beta=1.0, hidden_dim=128, free_bits=1.0,
                           patience=120, cov_weight=1.0, seed=seed, nonneg_idx=NONNEG,
                           lambda_cons=5.0, lambda_kin=0.0)
            for (T, Dur) in held:
                g = mech.generate(m, cvec(T, Dur), N_GEN, LATENT, sc, nonneg_idx=NONNEG)[:, CI_IDX]
                g = snap_to_support(g, tr["ChillingInjury"].values)
                for thr in THRESHOLDS:
                    gen[(T, Dur)][thr].append(at_risk(g, thr))
        for (T, Dur) in held:
            te = d[(d[COND[0]] == T) & (d[COND[1]] == Dur)]
            for thr in THRESHOLDS:
                true = at_risk(te["ChillingInjury"].values, thr)
                vfp = float(np.mean(gen[(T, Dur)][thr]))
                glm = glm_at_risk(tr, T, Dur, thr, stats)
                rows.append({"thr": thr, "T": float(T), "Dur": float(Dur), "true": true,
                             "glm_err": abs(glm - true), "vfp_err": abs(vfp - true)})
    df = pd.DataFrame(rows)
    print("\n=== Table S9d: leave-one-temperature-out extrapolation (VFP vs condition-aware GLM) ===")
    print(f"{'CI thr %':>9}{'GLM':>12}{'VFP':>12}")
    for thr in THRESHOLDS:
        sub = df[df.thr == thr]
        print(f"{thr:>9}{sub.glm_err.mean():>12.3f}{sub.vfp_err.mean():>12.3f}")
    print(f"overall    GLM {df.glm_err.mean():.3f}   VFP {df.vfp_err.mean():.3f}  "
          f"({len(temps)} temperature folds; directional)")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/citrus_digital_twin_extrapolation.csv", index=False)
    print(f"wrote: {OUT}/citrus_digital_twin_extrapolation.csv")


if __name__ == "__main__":
    run()
    run_extrapolation()
