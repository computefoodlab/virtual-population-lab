"""
Physics-VAE loss-term documentation: every term's scientific meaning,
mathematical definition, and weight. The term definitions mirror
src.generators.hybrid_vae_generator; the per-dataset weights are read from the
configs, so they cannot drift from what the experiments actually used.

Writes to results/_summary/:
  loss_terms.tex     — the loss-term table (meaning / math / weight)   [supplementary]
  loss_terms.md      — same table in Markdown (for the report / README)
  hyperparams.tex    — per-dataset weights + VAE hyperparameters, from configs
  hyperparams.md     — same in Markdown

Reproduce with `python -m src.loss_table`.
"""
import os

from src.config_loader import load_config

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "apple_quality": "Apple",
    "banana_quality": "Banana",
    "biofood_date_region": "Date",
    "biofood_safou_region": "Safou",
    "mango_composition": "Mango",
}

# Each term: name, weight symbol, scientific meaning, LaTeX math, Markdown math.
# L = L_rec + beta(t) L_KL + lambda_cov L_cov + lambda_phys L_phys + lambda_marg L_marg
TERMS = [
    {
        "name": "Reconstruction",
        "weight": r"$1$ (fixed)",
        "weight_md": "1 (fixed)",
        "meaning": "Data fidelity: the decoder must reproduce each real individual from its "
                   "latent code. The ELBO backbone.",
        "tex": r"$\mathcal{L}_{\mathrm{rec}}=\frac{1}{BD}\sum_{i}\lVert \hat{x}_i-x_i\rVert_2^2$",
        "md": "L_rec = mean_i ‖x̂_i − x_i‖² (MSE over B rows, D features)",
    },
    {
        "name": "KL divergence",
        "weight": r"$\beta(t)$, floor $c$",
        "weight_md": "β(t), free-bits floor c",
        "meaning": "Regularizes the approximate posterior toward the $\\mathcal{N}(0,I)$ prior so "
                   "the latent space is smooth and samplable. $\\beta$ is annealed $0\\!\\to\\!\\beta$ "
                   "over the first 30\\% of epochs; the per-dimension free-bits floor $c$ prevents "
                   "posterior collapse.",
        "meaning_md": "Regularizes the posterior toward the N(0,I) prior (smooth, samplable latent "
                      "space). β annealed 0→β over first 30% of epochs; per-dim free-bits floor c "
                      "prevents posterior collapse.",
        "tex": r"$\mathcal{L}_{\mathrm{KL}}=\frac{1}{B}\sum_i\sum_j \max\!\Big(\tfrac12\big(\mu_{ij}^2"
               r"+e^{s_{ij}}-1-s_{ij}\big),\,c\Big)$, $s=\log\sigma^2$",
        "md": "L_KL = mean_i Σ_j max( ½(μ_ij² + e^{s} − 1 − s), c ),  s = logσ²",
    },
    {
        "name": "Covariance matching",
        "weight": r"$\lambda_{\mathrm{cov}}$",
        "weight_md": "λ_cov",
        "meaning": "Preserves the full multivariate dependence: matches the generated feature "
                   "covariance matrix to the real one, which reconstruction + KL alone do not reward.",
        "tex": r"$\mathcal{L}_{\mathrm{cov}}=\frac{1}{D^2}\big\lVert \mathrm{Cov}(\tilde{X})"
               r"-\mathrm{Cov}(X)\big\rVert_F^2$",
        "md": "L_cov = (1/D²) ‖Cov(X̃) − Cov(X)‖²_F   (X̃ = generated batch)",
    },
    {
        "name": "Physics consistency",
        "weight": r"$\lambda_{\mathrm{phys}}$",
        "weight_md": "λ_phys",
        "meaning": "Enforces each expert-asserted causal edge as a fitted linear-Gaussian "
                   "conditional. The three sub-terms match the edge's intercept (residual mean $0$), "
                   "slope (residual uncorrelated with the parent), and physical noise level "
                   "(residual variance), so the mechanism is anchored without collapsing conditional "
                   "spread. Zero when the causal graph is empty.",
        "meaning_md": "Enforces each causal edge as a fitted linear-Gaussian conditional. The three "
                      "sub-terms match the edge's intercept (residual mean 0), slope (residual "
                      "uncorrelated with parent), and physical noise level (residual variance) — "
                      "anchoring the mechanism without collapsing spread. Zero if graph empty.",
        "tex": r"$\mathcal{L}_{\mathrm{phys}}=\frac{1}{|E|}\sum_{(p,c)\in E}\!\Big[\bar{r}^2"
               r"+\mathrm{Cov}(r,x_p)^2+\big(\mathrm{Var}(r)-\sigma^2_{pc}\big)^2\Big]$, "
               r"$r=x_c-(\beta_{pc}x_p+\alpha_{pc})$",
        "md": "L_phys = (1/|E|) Σ_(p→c) [ r̄² + Cov(r,x_p)² + (Var(r) − σ²_pc)² ],  "
              "r = x_c − (β_pc·x_p + α_pc)",
    },
    {
        "name": "Marginal matching",
        "weight": r"$\lambda_{\mathrm{marg}}$",
        "weight_md": "λ_marg",
        "meaning": "Differentiable analogue of the KS statistic (squared 1-D Wasserstein): pulls "
                   "each generated feature's whole empirical distribution onto real, including the "
                   "causal-graph root variables that the physics term leaves unconstrained.",
        "tex": r"$\mathcal{L}_{\mathrm{marg}}=\frac{1}{BD}\sum_j\sum_k\big(\tilde{x}_{(k)j}"
               r"-x_{(k)j}\big)^2$  (sorted order statistics)",
        "md": "L_marg = (1/BD) Σ_j Σ_k ( x̃_(k)j − x_(k)j )²   (per-feature sorted values)",
    },
]

# The fitted edge constants (from _fit_edges), documented once.
EDGE_TEX = (
    r"Edge constants are fit once on the training data by ordinary least squares: "
    r"slope $\beta_{pc}=\mathrm{Cov}(x_p,x_c)/\mathrm{Var}(x_p)$, intercept "
    r"$\alpha_{pc}=\bar{x}_c-\beta_{pc}\bar{x}_p$, residual variance "
    r"$\sigma^2_{pc}=\mathrm{Var}\!\big(x_c-(\beta_{pc}x_p+\alpha_{pc})\big)$ — the identical "
    r"linear-Gaussian conditionals the physics-informed Monte Carlo baseline samples from."
)
EDGE_MD = ("Edge constants fit once on training data by OLS: slope β_pc = Cov(x_p,x_c)/Var(x_p), "
           "intercept α_pc = mean(x_c) − β_pc·mean(x_p), residual variance "
           "σ²_pc = Var(x_c − (β_pc·x_p + α_pc)) — the same linear-Gaussian conditionals the "
           "physics-informed Monte Carlo baseline samples.")

CALIB_TEX = (
    r"After training, an optional empirical-copula post-step (not a loss term) maps each generated "
    r"feature onto the real training marginal at matching quantiles: it preserves the learned "
    r"rank (Spearman) dependence while forcing every marginal to match real almost exactly, and "
    r"guarantees physically valid values."
)
CALIB_MD = ("A post-training empirical-copula step (not a loss term) maps each generated feature onto "
            "the real training marginal at matching quantiles — preserves the learned rank "
            "dependence, forces marginals to match real, guarantees physically valid values.")


def write_loss_terms_tex():
    L = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Physics-VAE training objective, term by term. The total loss is "
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\beta(t)\,\mathcal{L}_{\mathrm{KL}}"
        r"+\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}+\lambda_{\mathrm{phys}}"
        r"\mathcal{L}_{\mathrm{phys}}+\lambda_{\mathrm{marg}}\mathcal{L}_{\mathrm{marg}}$, "
        r"minimized by Adam. $B$ = batch rows, $D$ = features, $E$ = causal edges; per-dataset "
        r"weights are in Table~\ref{tab:hyperparams}.}",
        r"\label{tab:loss}",
        r"\begin{tabular}{p{2.4cm} p{1.7cm} p{5.2cm} p{5.6cm}}", r"\toprule",
        r"Term & Weight & Scientific meaning & Mathematical definition \\", r"\midrule",
    ]
    for t in TERMS:
        meaning = t.get("meaning", "")
        L.append(f"{t['name']} & {t['weight']} & {meaning} & {t['tex']} \\\\[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\\[4pt]", r"\begin{minipage}{\textwidth}\footnotesize",
          EDGE_TEX, r"\\[2pt]", CALIB_TEX, r"\end{minipage}", r"\end{table*}"]
    open(f"{OUT}/loss_terms.tex", "w").write("\n".join(L) + "\n")


def write_loss_terms_md():
    L = ["## Physics-VAE loss terms", "",
         "Total objective (minimized by Adam):", "",
         "```",
         "L = L_rec + β(t)·L_KL + λ_cov·L_cov + λ_phys·L_phys + λ_marg·L_marg",
         "```",
         "B = batch rows, D = features, E = causal edges. Per-dataset weights: see the "
         "hyperparameter table.", "",
         "| Term | Weight | Scientific meaning | Mathematical definition |",
         "|---|---|---|---|"]
    for t in TERMS:
        meaning = t.get("meaning_md", t.get("meaning", "")).replace("\\%", "%").replace("$", "")
        meaning = meaning.replace("\\mathcal{N}(0,I)", "N(0,I)")
        L.append(f"| {t['name']} | {t['weight_md']} | {meaning} | {t['md']} |")
    L += ["", f"**Fitted edge constants.** {EDGE_MD}", "",
          f"**Post-hoc calibration.** {CALIB_MD}"]
    open(f"{OUT}/loss_terms.md", "w").write("\n".join(L) + "\n")


def _graph_desc(graph):
    if not graph:
        return "none (empty)"
    return ", ".join(f"{p}→{c}" for p, c in graph)


def collect_hyperparams():
    rows = []
    for cfg_name, label in DATASETS.items():
        c = load_config(cfg_name)
        rows.append({
            "dataset": label,
            "lambda_cov": c.VAE_COV_WEIGHT,
            "lambda_phys": c.VAE_PHYSICS_WEIGHT,
            "lambda_marg": c.VAE_MARGINAL_WEIGHT,
            "beta": c.VAE_BETA,
            "free_bits": c.VAE_FREE_BITS,
            "latent_dim": c.LATENT_DIM,
            "hidden_dim": c.VAE_HIDDEN_DIM,
            "dropout": c.VAE_DROPOUT,
            "epochs": c.VAE_EPOCHS,
            "n_edges": len(c.CAUSAL_GRAPH),
            "graph": _graph_desc(c.CAUSAL_GRAPH),
        })
    return rows


def write_hyperparams_tex(rows):
    L = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Per-dataset Physics-VAE loss weights and training hyperparameters, as used in "
        r"all reported runs. $\lambda_{\mathrm{phys}}$ has no effect on Banana or Mango, whose "
        r"causal graphs are empty. "
        r"Every value is a training-only choice; no evaluation data informs any prior, weight, or "
        r"preprocessing statistic.}",
        r"\label{tab:hyperparams}",
        r"\begin{tabular}{l ccc cc ccc c}", r"\toprule",
        r"Dataset & $\lambda_{\mathrm{cov}}$ & $\lambda_{\mathrm{phys}}$ & $\lambda_{\mathrm{marg}}$ "
        r"& $\beta$ & $c$ & $d_z$ & hidden & epochs & edges \\", r"\midrule",
    ]
    for r in rows:
        L.append(
            f"{r['dataset']} & {r['lambda_cov']:g} & {r['lambda_phys']:g} & {r['lambda_marg']:g} & "
            f"{r['beta']:g} & {r['free_bits']:g} & {r['latent_dim']} & {r['hidden_dim']} & "
            f"{r['epochs']} & {r['n_edges']} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/hyperparams.tex", "w").write("\n".join(L) + "\n")


def write_hyperparams_md(rows):
    L = ["## Per-dataset Physics-VAE weights & hyperparameters", "",
         "(Read directly from the configs; every value is a training-only choice — no evaluation "
         "data informs any prior, weight, or preprocessing statistic. λ_phys has no effect on "
         "Banana or Mango, whose causal graphs are empty.)", "",
         "| Dataset | λ_cov | λ_phys | λ_marg | β | free-bits c | latent d_z | hidden | epochs | causal edges |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['dataset']} | {r['lambda_cov']:g} | {r['lambda_phys']:g} | {r['lambda_marg']:g} | "
            f"{r['beta']:g} | {r['free_bits']:g} | {r['latent_dim']} | {r['hidden_dim']} | "
            f"{r['epochs']} | {r['graph']} |"
        )
    open(f"{OUT}/hyperparams.md", "w").write("\n".join(L) + "\n")


def main():
    write_loss_terms_tex()
    write_loss_terms_md()
    rows = collect_hyperparams()
    write_hyperparams_tex(rows)
    write_hyperparams_md(rows)
    print("wrote loss_terms.tex/.md and hyperparams.tex/.md to", OUT)
    print("\n" + open(f"{OUT}/loss_terms.md").read())
    print("\n" + open(f"{OUT}/hyperparams.md").read())


if __name__ == "__main__":
    main()
