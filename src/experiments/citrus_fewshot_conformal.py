"""
Few-shot conformal transfer on citrus (leave-one-condition-out).

Zero-shot off-support coverage is below nominal (the honest concession): a model
cannot calibrate to a storage condition it has never seen. This asks how quickly
conformal recalibration closes that gap given a few fruit from the unseen
condition.

Setup: leave-one-storage-condition-out on the citrus flagship (the held-out
condition is the "unseen group"). The VFP (conditional VAE conditioned on
temperature and duration, with rind non-negativity) is trained on the other 11
conditions and generates a population for the held-out condition. Of the held-out
fruit, k are used as a split-conformal calibration set and coverage@90 is measured
on the REMAINING held-out fruit; k is swept. Per-feature central intervals,
averaged across features and conditions.

Run:  python -m src.experiments.citrus_fewshot_conformal
(expects data/cleaned-citrus-exp6.csv - authors' unpublished data, on request.)
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.generators import mechanistic_cvae as mech

RAW = "data/cleaned-citrus-exp6.csv"
FEATURES = ["RindFresh", "RindDry", "MoistureLoss", "ChillingInjury", "Colour"]
NONNEG = list(range(len(FEATURES)))
COND = ["StorageTemp", "Duration"]
LEVEL = 0.90
KSHOTS = [0, 3, 5, 10]
SEEDS = [41, 42, 43]
N_GEN = 2000
LATENT = 4
EPOCHS = 500
OUT = "results/_summary"


def interval(gen, level):
    return np.quantile(gen, (1 - level) / 2), np.quantile(gen, (1 + level) / 2)


def conformal_q(gen, cal, level):
    """Split-conformal widening of the central interval from k calibration points."""
    lo, hi = interval(gen, level)
    s = np.maximum(lo - cal, cal - hi)
    n = len(cal)
    k = min(n, int(np.ceil((n + 1) * level)))
    return lo, hi, (np.sort(s)[k - 1] if n > 0 else 0.0)


def run():
    d = pd.read_csv(RAW)
    conds = sorted(d.groupby(COND).size().index.tolist())
    tm, ts = d[COND[0]].mean(), d[COND[0]].std()
    dm, ds = d[COND[1]].mean(), d[COND[1]].std()
    xm, xs = (d[COND[0]] * d[COND[1]]).mean(), (d[COND[0]] * d[COND[1]]).std()

    def cvec(T, Dur):
        return np.array([(T - tm) / ts, (Dur - dm) / ds, (T * Dur - xm) / xs])

    def run_seed(seed):
        rng = np.random.RandomState(seed)
        cov = {k: [] for k in KSHOTS}
        for T, Dur in conds:
            te = d[(d[COND[0]] == T) & (d[COND[1]] == Dur)].reset_index(drop=True)
            tr = d[~((d[COND[0]] == T) & (d[COND[1]] == Dur))]
            if len(te) < 12:
                continue
            sc = StandardScaler().fit(tr[FEATURES])
            cond = np.array([cvec(t, dd) for t, dd in zip(tr[COND[0]], tr[COND[1]])])
            np.random.seed(seed); torch.manual_seed(seed)
            m = mech.train(sc.transform(tr[FEATURES]), cond, sc, latent_dim=LATENT,
                           epochs=EPOCHS, beta=1.0, hidden_dim=128, free_bits=1.0,
                           patience=120, cov_weight=1.0, seed=seed, nonneg_idx=NONNEG,
                           lambda_cons=5.0, lambda_kin=0.0)
            g = np.clip(mech.generate(m, cvec(T, Dur), N_GEN, LATENT, sc, nonneg_idx=NONNEG), 0, None)
            idx = rng.permutation(len(te))
            for k in KSHOTS:
                cal_i, ev_i = idx[:k], idx[k:]
                if len(ev_i) < 3:
                    continue
                fcov = []
                for j, fname in enumerate(FEATURES):
                    gv, ev = g[:, j], te[fname].values[ev_i]
                    if k == 0:
                        lo, hi = interval(gv, LEVEL)
                    else:
                        lo, hi, q = conformal_q(gv, te[fname].values[cal_i], LEVEL)
                        lo, hi = lo - q, hi + q
                    fcov.append(np.mean((ev >= lo) & (ev <= hi)))
                cov[k].append(np.mean(fcov))
        return {k: np.mean(v) for k, v in cov.items() if v}

    agg = {k: [] for k in KSHOTS}
    for s in SEEDS:
        r = run_seed(s)
        for k in KSHOTS:
            if k in r:
                agg[k].append(r[k])

    print(f"\n=== Few-shot conformal transfer on citrus (leave-one-condition-out, "
          f"{len(SEEDS)} seeds) ===")
    print(f"off-support coverage@90 (target 0.90) vs target samples k:\n")
    records = []
    for k in KSHOTS:
        m = float(np.mean(agg[k]))
        print(f"  k={k:<3} coverage@90 = {m:.3f}")
        records.append({"target_samples_k": k, "coverage_at_90": round(m, 4), "n": len(agg[k])})
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(records).to_csv(f"{OUT}/citrus_fewshot_conformal.csv", index=False)
    print(f"\nwrote: {OUT}/citrus_fewshot_conformal.csv")


if __name__ == "__main__":
    run()
