"""
Consolidate every dataset's result CSVs into one master table.

Writes to results/_summary/:
  all_metrics.csv   — one row per (dataset, engine): corr, KS, gap, coverage,
                      calibration error, TSTR accuracy/AUC
  nearfar.csv       — one row per dataset: FAR/NEAR disagreement + widening %
  results.xlsx      — the same, as two sheets (if openpyxl is installed)

and prints both as Markdown tables.
"""
import os
import pandas as pd

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "apple_quality": "Apple (quality)",
    "banana_quality": "Banana (quality)",
    "biofood_date_region": "Date (nutrients)",
    "biofood_safou_region": "Safou (nutrients)",
    "mango_composition": "Mango (Vit C)",
}
ENGINES = ["physics_mc", "regression", "vae", "hybrid_vae"]
ENGINE_NAME = {"physics_mc": "Physics-MC", "regression": "Regression",
               "vae": "VAE", "hybrid_vae": "Physics-VAE"}
# summary_metrics.csv uses display names; map them back to engine keys. The
# parenthesised PI-VAE label is the one older CSVs were written with.
DISPLAY_TO_KEY = {
    "Physics-Informed Monte Carlo": "physics_mc", "Regression": "regression",
    "Variational Autoencoder": "vae", "Physics-Informed VAE": "hybrid_vae",
    "Physics-Informed VAE (Hybrid)": "hybrid_vae",
}


def _read(cfg, name):
    path = f"results/{cfg}/{name}"
    return pd.read_csv(path) if os.path.exists(path) else None


rows = []
for cfg, label in DATASETS.items():
    summ = _read(cfg, "summary_metrics.csv")
    cov = _read(cfg, "coverage.csv")
    gap = _read(cfg, "generalization_gap.csv")
    tstr = _read(cfg, "tstr.csv")
    if summ is None:
        continue
    summ = summ.assign(engine=summ["Engine"].map(DISPLAY_TO_KEY)).set_index("engine")
    cov = cov.set_index("method") if cov is not None else None
    gap = gap.set_index("method") if gap is not None else None
    tstr = tstr.set_index("method") if tstr is not None else None
    for e in ENGINES:
        r = {"dataset": label, "engine": ENGINE_NAME[e]}
        if e in summ.index:
            r["corr_dist"] = round(summ.loc[e, "Correlation Distance (Euclidean)"], 3)
            r["mean_ks"] = round(summ.loc[e, "Mean KS Statistic"], 3)
        if gap is not None and e in gap.index:
            r["gen_gap"] = round(gap.loc[e, "gap"], 3)
        if cov is not None and e in cov.index:
            r["coverage@90"] = round(cov.loc[e, "coverage_at_90"], 3)
            r["calib_err"] = round(cov.loc[e, "calibration_error"], 3)
        if tstr is not None and e in tstr.index:
            r["tstr_acc"] = round(tstr.loc[e, "accuracy"], 3)
            r["tstr_auc"] = ("" if pd.isna(tstr.loc[e, "roc_auc"]) else round(tstr.loc[e, "roc_auc"], 3))
        rows.append(r)
    # add the real-data TSTR ceiling as a reference row
    if tstr is not None and "real" in tstr.index:
        rows.append({"dataset": label, "engine": "Real (ceiling)",
                     "tstr_acc": round(tstr.loc["real", "accuracy"], 3),
                     "tstr_auc": ("" if pd.isna(tstr.loc["real", "roc_auc"]) else round(tstr.loc["real", "roc_auc"], 3))})

metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT}/all_metrics.csv", index=False)

# near/far summary (from each loro.csv MEAN row)
nf = []
for cfg, label in DATASETS.items():
    lo = _read(cfg, "loro.csv")
    if lo is None or lo.empty:
        continue
    m = lo[lo["group"] == "MEAN"]
    if m.empty:
        continue
    m = m.iloc[0]
    nf.append({
        "dataset": label,
        "disagree_far": round(m["disagreement_far"], 4),
        "disagree_near": round(m["disagreement_near"], 4),
        "widening_%": round(100 * (m["disagreement_far"] / m["disagreement_near"] - 1)),
        "coverage_far": round(m["coverage_far"], 3),
        "coverage_near": round(m["coverage_near"], 3),
    })
nearfar = pd.DataFrame(nf)
nearfar.to_csv(f"{OUT}/nearfar.csv", index=False)

# optional Excel with two sheets
try:
    with pd.ExcelWriter(f"{OUT}/results.xlsx") as xl:
        metrics.to_excel(xl, sheet_name="metrics", index=False)
        nearfar.to_excel(xl, sheet_name="near_far", index=False)
    xlsx = "results.xlsx"
except Exception as e:  # openpyxl missing, etc.
    xlsx = f"(Excel skipped: {e})"

def _md(df):
    cols = list(df.columns)
    widths = {c: max(len(str(c)), *(len(str(v)) for v in df[c])) for c in cols}
    head = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    body = ["| " + " | ".join(str(v).ljust(widths[c]) for c, v in zip(cols, row)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


print("### All metrics (per dataset x engine)\n")
print(_md(metrics.fillna("")))
print("\n### Near/far (uncertainty widens off-support)\n")
print(_md(nearfar))
print(f"\nwrote: {OUT}/all_metrics.csv, {OUT}/nearfar.csv, {xlsx}")
