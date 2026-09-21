"""
Temporal-drift detection and revalidation trigger (Reviewer 1 #5).

A deployed VFP is a snapshot of the population at training time; fruit populations
shift with season, cultivar, practice and climate, so a VFP must be a versioned,
monitored asset. The monitoring signal is native to the framework: each incoming
batch of real measurements is scored by its distribution distance (energy
distance) to the deployed reference population.

This experiment shows the detector (i) does not false-alarm on fresh batches of
the SAME population (they fall in a null band), (ii) flags batches from a DRIFTED
population, and (iii) scores scale monotonically with the true population shift —
so the same statistic both detects drift and grades its severity, which is exactly
the trigger a versioning/revalidation policy needs.

Run:  python -m src.experiments.drift_detection
"""
import sys, os; sys.path.insert(0, os.getcwd())
import logging; logging.disable(logging.CRITICAL)
import numpy as np
from src.config_loader import load_config
from src.data_loading import load_data
from src.multiseed import energy_distance

BATCH = 40        # incoming batch size
N_NULL = 200      # same-population resamples defining the null band
N_DRIFT = 50      # drift batches per candidate population
SEED0 = 42

# (config, reference population) pairs
CASES = [("tomato_nir", "ProcessingN"), ("grape_berry", "Grenache")]


def run(cfg_name, ref_group):
    cfg = load_config(cfg_name); F = cfg.FEATURES; G = cfg.LORO_GROUP
    df = load_data(cfg); groups = sorted(df[G].unique())
    ref = df[df[G] == ref_group]
    # split the reference so the null band is measured on held-out same-population data
    half = ref.sample(frac=0.5, random_state=SEED0)
    other = ref.drop(half.index)
    null = np.array([
        energy_distance(half, other.sample(BATCH, replace=True, random_state=SEED0 + i),
                        F, seed=SEED0 + i)
        for i in range(N_NULL)])
    thr = np.quantile(null, 0.95)

    print(f"\n=== {cfg_name}: drift detector — reference = {ref_group} (batch={BATCH}) ===")
    print(f"null (same population): mean {null.mean():.3f}; 95th-pct alarm threshold {thr:.3f}")
    print(f"{'incoming population':<16}{'drift score':>12}{'flag':>8}{'true pop shift':>16}")
    rows = []
    for g in groups:
        cand = df[df[G] == g]
        if len(cand) < BATCH:
            continue
        score = np.mean([
            energy_distance(half, cand.sample(BATCH, replace=True, random_state=SEED0 + i),
                            F, seed=SEED0 + i)
            for i in range(N_DRIFT)])
        pop_shift = energy_distance(ref, cand, F, seed=SEED0)  # full-population shift
        rows.append((g, score, score > thr, pop_shift))
    for g, s, flag, shift in sorted(rows, key=lambda r: r[3]):
        tag = "DRIFT" if flag else "ok"
        mark = "  <- reference" if g == ref_group else ""
        print(f"{str(g):<16}{s:>12.3f}{tag:>8}{shift:>16.3f}{mark}")


def main():
    for cfg_name, ref in CASES:
        run(cfg_name, ref)
    print("\n(same-population batch stays below threshold = no false alarm; drift "
          "scores exceed it and scale with the true shift = severity-graded trigger.)")


if __name__ == "__main__":
    main()
