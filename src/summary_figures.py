"""
Cross-dataset summary figures for the five finalized datasets. Reads the
aggregated multi-seed results (results/_summary/multiseed_metrics.csv and
multiseed_nearfar.csv) so the figures show mean +/- standard deviation with error
bars rather than a single seed. Regenerate those first with
`python -m src.multiseed --seeds 41 42 43 44 45 --loro`.

  fig1_leaderboard.png  — 4 engines x 5 datasets on correlation distance + KS (mean +/- s.d.)
  fig2_downstream.png   — TSTR accuracy per engine vs the real ceiling (apple, banana)
  fig3_nearfar.png      — near/far ensemble disagreement (uncertainty widens off-support)

Palette: Okabe-Ito (colourblind-safe); colour is bound to engine, never to rank.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

# Fixed engine order and categorical colours.
ENGINES = ["physics_mc", "regression", "vae", "hybrid_vae"]
SHORT = {"physics_mc": "Physics-MC", "regression": "Regression",
         "vae": "VAE", "hybrid_vae": "Physics-VAE"}
COLOR = {"physics_mc": "#0072B2", "regression": "#E69F00",
         "vae": "#009E73", "hybrid_vae": "#D55E00", "real": "#555555"}
DATASETS = {"apple_quality": "Apple\n(quality)", "banana_quality": "Banana\n(quality)",
            "biofood_date_region": "Date\n(nutrients)", "biofood_safou_region": "Safou\n(nutrients)",
            "mango_composition": "Mango\n(Vit C)"}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})

_M = pd.read_csv(f"{OUT}/multiseed_metrics.csv")
N_SEEDS = int(_M["n"].max())


def get(cfg, eng, metric):
    r = _M[(_M.config == cfg) & (_M.engine == eng) & (_M.metric == metric)]
    return (float(r["mean"].iloc[0]), float(r["std"].iloc[0])) if not r.empty else (np.nan, np.nan)


def _bars(ax, groups, means, errs, title, ylabel):
    """Grouped bars with error bars: groups on x, ENGINES within each group."""
    n = len(ENGINES); gw = 0.8; bw = gw / n
    x = np.arange(len(groups))
    for i, eng in enumerate(ENGINES):
        pos = x - gw / 2 + bw * (i + 0.5)
        m, e = np.array(means[eng]), np.array(errs[eng])
        ax.bar(pos, m, bw * 0.86, yerr=e, color=COLOR[eng], label=SHORT[eng], zorder=3,
               capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#555"})
        for p, v, err in zip(pos, m, e):
            if not np.isnan(v):
                ax.text(p, v + (err if not np.isnan(err) else 0) + ax.get_ylim()[1] * 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=8)
    ax.margins(y=0.20)


# ---------- Figure 1: leaderboard (mean +/- s.d.) ----------
corr_m = {e: [] for e in ENGINES}; corr_s = {e: [] for e in ENGINES}
ks_m = {e: [] for e in ENGINES}; ks_s = {e: [] for e in ENGINES}
for cfg in DATASETS:
    for e in ENGINES:
        m, s = get(cfg, e, "corr_dist"); corr_m[e].append(m); corr_s[e].append(s)
        m, s = get(cfg, e, "mean_ks"); ks_m[e].append(m); ks_s[e].append(s)
labels = list(DATASETS.values())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
_bars(a1, labels, corr_m, corr_s, "Joint structure  (lower is better)", "Correlation distance")
_bars(a2, labels, ks_m, ks_s, "Marginal fit  (lower is better)", "Mean KS statistic")
handles, leg = a1.get_legend_handles_labels()
fig.legend(handles, leg, frameon=False, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 0.925), fontsize=10)
fig.suptitle(f"Engine leaderboard across the five datasets  (mean ± s.d. over {N_SEEDS} seeds)",
             x=0.02, ha="left", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(f"{OUT}/fig1_leaderboard.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------- Figure 2: downstream TSTR accuracy (mean +/- s.d.) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, cfg in zip(axes, ["apple_quality", "banana_quality"]):
    order = ["real"] + ENGINES
    names = {"real": "Real (ceiling)", **SHORT}
    ms = [(o, *get(cfg, o, "tstr_acc")) for o in order]
    ms = [(o, m, s) for o, m, s in ms if not np.isnan(m)]
    xs = np.arange(len(ms))
    ax.bar(xs, [m for _, m, _ in ms], 0.62, yerr=[s for _, _, s in ms],
           color=[COLOR[o] for o, _, _ in ms], zorder=3, capsize=3,
           error_kw={"elinewidth": 1.0, "ecolor": "#555"})
    for xi, (_, m, s) in zip(xs, ms):
        ax.text(xi, m + s + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ceil = dict((o, m) for o, m, _ in ms)["real"]
    ax.axhline(ceil, color="#555555", lw=1.2, ls="--", zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([names[o] for o, _, _ in ms], rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title(DATASETS[cfg].replace("\n", " "), fontsize=12, weight="bold", loc="left")
    lo = min(m - s for _, m, s in ms)
    ax.set_ylim(lo * 0.95, (ceil + 0.05))
fig.suptitle(f"Downstream utility (train-on-synthetic, test-on-real) — Physics-VAE nearest the real ceiling "
             f"(mean ± s.d. over {N_SEEDS} seeds)",
             x=0.02, ha="left", fontsize=12.5, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig2_downstream.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------- Figure 3: near/far uncertainty (mean +/- s.d.) ----------
nf = pd.read_csv(f"{OUT}/multiseed_nearfar.csv")
nf_seeds = int(nf["n_seeds"].max())
groups, far, far_s, near, near_s, pct = [], [], [], [], [], []
for cfg in ["mango_composition", "banana_quality", "biofood_date_region", "biofood_safou_region"]:
    r = nf[nf.config == cfg]
    if r.empty:
        continue
    r = r.iloc[0]
    groups.append(DATASETS[cfg].replace("\n", " "))
    far.append(r["disagreement_far_mean"]); far_s.append(r["disagreement_far_std"])
    near.append(r["disagreement_near_mean"]); near_s.append(r["disagreement_near_std"])
    pct.append(r["widening_pct_mean"])
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(groups)); bw = 0.38
ax.bar(x - bw / 2, far, bw * 0.9, yerr=far_s, color="#D55E00", label="FAR (different group)",
       zorder=3, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#555"})
ax.bar(x + bw / 2, near, bw * 0.9, yerr=near_s, color="#0072B2", label="NEAR (held-out same group)",
       zorder=3, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#555"})
for xi, f, fs, n, ns, p in zip(x, far, far_s, near, near_s, pct):
    ax.text(xi, max(f + fs, n + ns) * 1.06, f"+{p:.0f}%", ha="center", va="bottom", fontsize=11,
            weight="bold", color="#D55E00")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Ensemble disagreement (per-feature-std)")
ax.margins(y=0.22); ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title(f"Support-aware uncertainty: disagreement widens for an unseen group (region/cultivar)\n"
             f"(same model, held-out same-group vs different-group; mean ± s.d. over {nf_seeds} seeds)",
             fontsize=12.5, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_nearfar.png", dpi=300, bbox_inches="tight"); plt.close()

print(f"wrote fig1_leaderboard.png, fig2_downstream.png, fig3_nearfar.png "
      f"(multi-seed, mean ± s.d.) to {OUT}/")
