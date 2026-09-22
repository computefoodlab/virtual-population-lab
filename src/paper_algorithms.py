"""
Model definitions for the paper: one equation block per baseline for the main
text, plus the detailed Physics-VAE algorithm and equations for the
supplementary. Every equation mirrors the actual generator code:
  physics_mc_generator, regression_generator, vae_generator, hybrid_vae_generator.

Writes to results/_summary/:
  baselines.tex / .md            — one equation block per engine (main text)
  physicsvae_algorithm.tex / .md — full Physics-VAE training+generation algorithm (supp.)
  algorithms_standalone.tex      — compilable wrapper (needs algorithm, algpseudocode, amsmath)

Reproduce with `python -m src.paper_algorithms`.
"""
import os

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ baselines (main text)
BASELINES_TEX = r"""% Per-baseline model equations (main manuscript). Notation: a real feature
% vector x in R^D; training matrix X (n x D); synthetic matrix \tilde{X} (N x D).
% Requires amsmath, amssymb.
\paragraph{Structural Monte Carlo.} Ancestral sampling over a caller-supplied
structural graph with edge set $E$ and roots $R$. The edges are estimated from
observational data by ordinary least squares, so they encode conditional
relationships rather than identified causal effects; ancestral sampling requires
only a valid factorization of the joint, which any topological ordering of the
graph supplies. Each root is drawn from its real
Gaussian marginal; each edge $(p\!\to\!c)\in E$ draws the child from an
ordinary-least-squares fit plus Gaussian residual noise, in topological order:
\begin{align}
\tilde{x}_r &\sim \mathcal{N}\!\big(\hat{\mu}_r,\hat{\sigma}_r^2\big), \qquad r\in R,\\
\tilde{x}_c &= \beta_{pc}\,\tilde{x}_p + \alpha_{pc} + \varepsilon_c,\quad
\varepsilon_c\sim\mathcal{N}\!\big(0,\sigma_{pc}^2\big),\quad (p\!\to\!c)\in E,
\end{align}
where $(\beta_{pc},\alpha_{pc})$ are the OLS coefficients of $x_c$ on $x_p$ and
$\sigma_{pc}^2$ the residual variance, all estimated on the training data. Every
conditional is Gaussian, so plain ancestral sampling is exact (no MCMC needed).

\paragraph{Regression (nonparametric).} For each feature $j$ a random forest $f_j$
predicts $x_j$ from the remaining features $x_{-j}$, fit on the training data.
Out-of-bag predictions $\hat{x}^{\mathrm{oob}}_j(i)$ give leakage-free fitted values
and residuals $r_j(i)=x_j(i)-\hat{x}^{\mathrm{oob}}_j(i)$. A synthetic row picks a
random training base index $b\sim\mathrm{Unif}\{1,\dots,n\}$ and adds a bootstrapped
residual per feature:
\begin{equation}
\tilde{x}_j = \hat{x}^{\mathrm{oob}}_j(b) + e_j,\qquad e_j\sim \widehat{F}_{r_j},
\end{equation}
with $\widehat{F}_{r_j}$ the empirical residual distribution. (This is the
nonparametric analogue of the linear model $y=B^{\top}x+\varepsilon$,
$\varepsilon\sim\mathcal{N}(0,\Sigma_\varepsilon)$; the forest replaces $B^\top x$
and the empirical residuals replace the Gaussian $\varepsilon$.)

\paragraph{Variational autoencoder.} A standard VAE with encoder $q_\phi$, decoder
$g_\theta$, and standard-normal prior:
\begin{align}
q_\phi(z\mid x) &= \mathcal{N}\!\big(\mu_\phi(x),\,\mathrm{diag}\,\sigma^2_\phi(x)\big),
\quad z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I),\\
\hat{x} &= g_\theta(z),\qquad p(z)=\mathcal{N}(0,I),\\
\mathcal{L}_{\mathrm{VAE}} &= \underbrace{\lVert x-\hat{x}\rVert_2^2}_{\text{recon.}}
+\ \beta\,D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\Vert\,\mathcal{N}(0,I)\big)
+\ \lambda_{\mathrm{cov}}\big\lVert \mathrm{Cov}(\tilde{X})-\mathrm{Cov}(X)\big\rVert_F^2.
\end{align}
Generation draws $z\sim\mathcal{N}(0,I)$ and returns $\tilde{x}=g_\theta(z)$.

\paragraph{Physics-VAE (proposed).} The VAE above augmented with a physics-consistency
term on the fitted structural edges and a marginal-matching term, followed by an
empirical-copula calibration step:
\begin{equation}
\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\beta(t)\,\mathcal{L}_{\mathrm{KL}}
+\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}
+\lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}
+\lambda_{\mathrm{marg}}\mathcal{L}_{\mathrm{marg}}.
\end{equation}
Each term's definition and weight are given in Table~\ref{tab:loss}; the full
training and generation procedure is in Algorithm~\ref{alg:physvae} (Supplementary).
"""

BASELINES_MD = r"""## Baseline model equations (main text)

Notation: real feature vector x ∈ R^D; training matrix X (n×D); synthetic X̃ (N×D).

**Structural Monte Carlo** — ancestral sampling over the structural graph (roots R, edges E):
```
roots:  x̃_r ~ N(μ_r, σ_r²)                                        r ∈ R
edges:  x̃_c = β_pc·x̃_p + α_pc + ε_c,  ε_c ~ N(0, σ²_pc)          (p→c) ∈ E
```
(β_pc, α_pc) = OLS of x_c on x_p; σ²_pc = residual variance; all fit on training data. Every conditional is Gaussian → exact ancestral sampling, no MCMC.

**Regression (nonparametric)** — per-feature random forest + empirical residual resampling:
```
for each feature j:  f_j predicts x_j from x_{-j};  OOB residuals r_j = x_j − x̂_j^oob
generate:  x̃_j = x̂_j^oob(b) + e_j,   b ~ Unif{1..n},   e_j ~ Ê[r_j]
```
Nonparametric analogue of the linear model y = Bᵀx + ε, ε ~ N(0, Σε): the forest replaces Bᵀx, empirical residuals replace ε.

**Variational autoencoder** — standard VAE:
```
q_φ(z|x) = N(μ_φ(x), diag σ²_φ(x));   z = μ_φ(x) + σ_φ(x)⊙ε, ε ~ N(0,I)
x̂ = g_θ(z);   p(z) = N(0,I)
L_VAE = ‖x − x̂‖² + β·KL(q_φ‖N(0,I)) + λ_cov·‖Cov(X̃) − Cov(X)‖²_F
generate:  z ~ N(0,I) → x̃ = g_θ(z)
```

**Physics-VAE (proposed)** — VAE + physics + marginal terms + copula calibration:
```
L = L_rec + β(t)·L_KL + λ_cov·L_cov + λ_phys·L_phys + λ_marg·L_marg
```
Term definitions/weights: loss_terms.md; full procedure: physicsvae_algorithm.md.
"""

# ------------------------------------------------------------ Physics-VAE algorithm (supp.)
ALG_TEX = r"""% Full Physics-VAE training + generation (Supplementary).
% Requires: algorithm, algpseudocode, amsmath, amssymb.
\begin{algorithm}[t]
\caption{Physics-VAE: training and virtual-population generation}
\label{alg:physvae}
\begin{algorithmic}[1]
\Require training matrix $X\in\mathbb{R}^{n\times D}$; structural edges $E$; weights
$\lambda_{\mathrm{cov}},\lambda_{\mathrm{phys}},\lambda_{\mathrm{marg}},\beta$;
free-bits $c$; epochs $T$; latent dim $d_z$; population size $N$
\State Fit scaler on $X$ (\textbf{training data only}); standardize $X$
\For{each edge $(p\!\to\!c)\in E$} \Comment{physics constants, fit once on $X$}
  \State $\beta_{pc}\gets \mathrm{Cov}(x_p,x_c)/\mathrm{Var}(x_p)$;\ \
         $\alpha_{pc}\gets \bar{x}_c-\beta_{pc}\bar{x}_p$;\ \
         $\sigma^2_{pc}\gets \mathrm{Var}\!\big(x_c-(\beta_{pc}x_p+\alpha_{pc})\big)$
\EndFor
\State initialize encoder $\phi$, decoder $\theta$
\For{$t=1$ to $T$}
  \State $\beta(t)\gets \beta\cdot\min\!\big(1,\ t/(0.3T)\big)$ \Comment{KL warm-up}
  \State $(\mu,s)\gets \mathrm{Enc}_\phi(X)$;\ \ $s\gets \mathrm{clamp}(s,-8,8)$ \Comment{$s=\log\sigma^2$}
  \State $Z\gets \mu + e^{s/2}\odot\epsilon,\ \ \epsilon\sim\mathcal{N}(0,I)$ \Comment{reparameterize}
  \State $\hat{X}\gets \mathrm{Dec}_\theta(Z)$ \Comment{reconstructions}
  \State $Z'\sim\mathcal{N}(0,I)^{n\times d_z}$;\ \ $\tilde{X}\gets \mathrm{Dec}_\theta(Z')$ \Comment{generated batch}
  \State $\mathcal{L}_{\mathrm{rec}}\gets \tfrac{1}{nD}\lVert \hat{X}-X\rVert^2$
  \State $\mathcal{L}_{\mathrm{KL}}\gets \tfrac{1}{n}\sum_i\sum_j \max\!\big(\tfrac12(\mu_{ij}^2+e^{s_{ij}}-1-s_{ij}),\,c\big)$
  \State $\mathcal{L}_{\mathrm{cov}}\gets \tfrac{1}{D^2}\lVert \mathrm{Cov}(\tilde{X})-\mathrm{Cov}(X)\rVert_F^2$
  \For{each edge $(p\!\to\!c)\in E$} \Comment{on generated $\tilde{X}$; $r=\tilde{x}_c-(\beta_{pc}\tilde{x}_p+\alpha_{pc})$}
    \State accumulate $\bar{r}^2+\mathrm{Cov}(r,\tilde{x}_p)^2+(\mathrm{Var}(r)-\sigma^2_{pc})^2$
  \EndFor
  \State $\mathcal{L}_{\mathrm{phys}}\gets \tfrac{1}{|E|}\sum_{\text{edges}}(\cdots)$
  \State $\mathcal{L}_{\mathrm{marg}}\gets \tfrac{1}{nD}\sum_j\sum_k\big(\tilde{x}_{(k)j}-x_{(k)j}\big)^2$ \Comment{sorted per feature}
  \State $\mathcal{L}\gets \mathcal{L}_{\mathrm{rec}}+\beta(t)\mathcal{L}_{\mathrm{KL}}
          +\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}
          +\lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}
          +\lambda_{\mathrm{marg}}\mathcal{L}_{\mathrm{marg}}$
  \State $(\phi,\theta)\gets \mathrm{Adam}\ \text{step on}\ \nabla_{\phi,\theta}\mathcal{L}$;\ \ early-stop on plateau
\EndFor
\State $Z'\sim\mathcal{N}(0,I)^{N\times d_z}$;\ \ $\tilde{X}\gets \mathrm{Dec}_\theta(Z')$ \Comment{generate population}
\For{each feature $j$} \Comment{empirical-copula calibration onto real training marginal}
  \State $\tilde{X}_{\cdot j}\gets \widehat{Q}^{\,\mathrm{real}}_j\!\big((\mathrm{rank}(\tilde{X}_{\cdot j})-\tfrac12)/N\big)$
\EndFor
\State inverse-transform $\tilde{X}$ with the training scaler
\State \Return $\tilde{X}$
\end{algorithmic}
\end{algorithm}
"""

ALG_MD = r"""## Physics-VAE algorithm — training + generation (Supplementary)

Inputs: training matrix X (n×D); structural edges E; weights λ_cov, λ_phys, λ_marg, β; free-bits c; epochs T; latent dim d_z; population size N.

```
1.  Fit scaler on X (TRAINING DATA ONLY); standardize X.
2.  For each edge (p→c) in E  — physics constants, fit once on X:
        β_pc = Cov(x_p,x_c)/Var(x_p);  α_pc = mean(x_c) − β_pc·mean(x_p)
        σ²_pc = Var( x_c − (β_pc·x_p + α_pc) )
3.  Initialize encoder φ, decoder θ.
4.  for t = 1..T:
      β(t) = β·min(1, t/(0.3T))                         # KL warm-up
      (μ, s) = Enc_φ(X);  s = clamp(s, −8, 8)           # s = logσ²
      Z  = μ + exp(s/2)⊙ε,  ε ~ N(0,I)                  # reparameterize (encode→sample)
      X̂  = Dec_θ(Z)                                     # decode reconstructions
      Z' ~ N(0,I)^{n×d_z};  X̃ = Dec_θ(Z')               # generated batch
      L_rec  = mean‖X̂ − X‖²
      L_KL   = mean_i Σ_j max( ½(μ² + e^s − 1 − s), c )
      L_cov  = (1/D²)‖Cov(X̃) − Cov(X)‖²_F               # on generated X̃
      L_phys = (1/|E|) Σ_(p→c) [ r̄² + Cov(r,x̃_p)² + (Var(r) − σ²_pc)² ],  r = x̃_c − (β_pc·x̃_p + α_pc)
      L_marg = (1/nD) Σ_j Σ_k ( x̃_(k)j − x_(k)j )²       # sorted order statistics per feature
      L = L_rec + β(t)·L_KL + λ_cov·L_cov + λ_phys·L_phys + λ_marg·L_marg
      Adam step on ∇_{φ,θ} L;  early-stop on plateau     # backprop
5.  Generate: Z' ~ N(0,I)^{N×d_z};  X̃ = Dec_θ(Z').
6.  Copula calibration — for each feature j:
        X̃_·j ← Q̂_j^real( (rank(X̃_·j) − ½) / N )         # map onto real training marginal
7.  Inverse-transform X̃ with the training scaler.  Return X̃.
```

Notes: the covariance and physics terms are evaluated on the freshly generated batch X̃
(`constrain_generated=True`), so they shape the population actually sampled at generation.
All priors, edge constants, scaler, and calibration reference come exclusively from X (the
training split). Set λ_phys = 0 to recover a calibrated VAE; empty E ⇒ L_phys = 0.
"""

STANDALONE = r"""% Compilable check for the paper algorithms/equations.
% (pdflatex algorithms_standalone.tex)
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[margin=1in]{geometry}
\begin{document}
\input{baselines.tex}
\input{physicsvae_algorithm.tex}
\end{document}
"""


def main():
    open(f"{OUT}/baselines.tex", "w").write(BASELINES_TEX)
    open(f"{OUT}/baselines.md", "w").write(BASELINES_MD)
    open(f"{OUT}/physicsvae_algorithm.tex", "w").write(ALG_TEX)
    open(f"{OUT}/physicsvae_algorithm.md", "w").write(ALG_MD)
    open(f"{OUT}/algorithms_standalone.tex", "w").write(STANDALONE)
    print("wrote baselines.tex/.md, physicsvae_algorithm.tex/.md, algorithms_standalone.tex to", OUT)
    print("\n" + BASELINES_MD + "\n" + ALG_MD)


if __name__ == "__main__":
    main()
