# MLS-SBO: Mathematical Formulation

This document states precisely the mathematics implemented in
`scp_uno/mls_sbo_core.py` (algorithm `MLS_SBO`). Code symbols are given in
parentheses.

## 1. Problem

$$\min_{x \in \Omega} \; f(x) \quad \text{s.t.} \quad c_j(x) \le 0,\; j=1..m,
\qquad \Omega = \{x : \ell \le x \le u\}$$

Internally all quantities live in the normalized box $\xi \in [0,1]^d$,
$\xi = (x-\ell)/(u-\ell)$; gradients are scaled accordingly
(`MLSSBOptimizer._eval`). Every true evaluation returns the tuple
$(f, \nabla f, c, \nabla c)$ — one FEM solve plus adjoints — and is stored in
the sample archive

$$\mathcal{S} = \{(\xi_i,\, f_i,\, g_i,\, c_i,\, J_i)\}_{i=1}^{N},
\qquad g_i = \nabla f(\xi_i),\; J_i = \nabla c(\xi_i).$$

## 2. Weights and length scale

All fits use Gaussian weights centred at an evaluation point $z$
(`MovingLeastSquares._weights`):

$$w_i(z) \;=\; \exp\!\Big(-\frac{\|z-\xi_i\|^2 - r_{\min}^2(z)}{2h^2}\Big),
\qquad r_{\min}^2(z) = \min_l \|z-\xi_l\|^2 .$$

The subtraction of $r_{\min}^2$ (largest weight $\equiv 1$) is exact — a
weighted least-squares solution is invariant to a common scaling of the
weights — and prevents underflow at small $h$, which is what makes the
interpolation limit $h \to 0$ and linear exactness hold at any length scale.

**Length-scale evolution** (`_min_dist_in_tr`): at iteration $k$ with
trust-region radius $\delta_k$,

$$h_k \;=\; \operatorname{clip}\big(\gamma_{ls}\, d_{\min},\; h_{\min},\, h_{\max}\big),
\qquad
d_{\min} = \min_{\substack{i \,:\, \|\xi_i - \xi_k\|_\infty \le \delta_k \\ \xi_i \ne \xi_k}} \|\xi_i - \xi_k\| ,$$

i.e. $h$ tracks the minimal distance from the trust-region centre to a
sampled point inside the trust region ($\gamma_{ls}$ = `ls_factor`, default
2). As the trust region shrinks and samples cluster, the fit localizes at the
pace of the sampling. Only the window of the `max_points` = 60 samples
nearest the centre enters any fit (`_fit_surrogate`), with the incumbent best
always retained.

## 3. Per-query gradient-enhanced linear MLS (the surrogate)

Used by the batch acquisition and as the planar model source. Outputs are
standardized per output $o$: $\tilde y = (y - \mu_o)/\sigma_o$. At a query
point $z$ the local linear model $q(s) = \alpha + \beta^\top s$, $s = \xi - z$,
is fit by weighted least squares to **both observation types** (Hermite MLS):

* value rows:&nbsp; $\alpha + \beta^\top s_i \approx \tilde f_i$, weight $w_i(z)$;
* gradient rows: $\beta_l \approx \tilde g_{i,l}$, $l = 1..d$, weight $w_i(z)$.

The stationarity conditions give one shared $(d{+}1)\times(d{+}1)$ normal
system for all $m{+}1$ outputs (`_solve_local`; $\varepsilon$ = Tikhonov
jitter):

$$
\underbrace{\begin{pmatrix}
\sum_i w_i & \sum_i w_i s_i^\top \\[2pt]
\sum_i w_i s_i & \sum_i w_i\,(s_i s_i^\top + I)
\end{pmatrix}}_{A(z)}
\begin{pmatrix}\alpha\\ \beta\end{pmatrix}
=
\begin{pmatrix}
\sum_i w_i \tilde f_i \\[2pt]
\sum_i w_i (s_i \tilde f_i + \tilde g_i)
\end{pmatrix}.
$$

The $+I$ block is the contribution of the gradient equations. Prediction mean
$= \alpha$; the **diffuse derivative** $= \beta$ (the true derivative would
add terms from $\partial w_i/\partial z$; see §5). Properties (test-enforced):
exact reproduction of affine functions at any $h$; interpolation of the
nearest sample's value and gradient as $h \to 0$.

**Uncertainty proxy** (no kriging variance exists): the sample-density
complement with *unshifted* weights $\bar w_i(z) = e^{-\|z-\xi_i\|^2/2h^2}$,

$$\sigma(z) = \sqrt{\max\Big(0,\; 1 - \sum_i \bar w_i(z)\Big)} \in [0,1],$$

zero where sampling is dense, $\to 1$ far from all data, with the exact
gradient $\nabla \sigma = -\nabla\big(\sum_i \bar w_i\big)/(2\sigma)$.

## 4. The exploitation model: centre-frozen anchored separable quadratic

The step model freezes the weights at the trust-region centre $\xi_k$ — the
"moving" of MLS happens **across iterations**, as the centre moves — so the
model is an exact polynomial whose values and gradients are mutually
consistent for the SQP subproblem solver (`AnchoredSeparableQuadratic`).

**Intermediate variables** (per output $o$ and coordinate $j$;
`intermediate`): with `"linear"`, $y_j = \xi_j$. With `"mma"`, a reciprocal
transform is placed on the descent side of the anchor gradient
$g_{k,j}^{(o)}$ (asymptote distance $a = \max(a_0,\, 1.3\,\delta_k)$):

$$
y_j = \begin{cases}
1/(\xi_j - L_j), & g^{(o)}_{k,j} < 0,\; L_j = \xi_{k,j} - a
& \text{(lower asymptote)}\\[2pt]
1/(U_j - \xi_j), & g^{(o)}_{k,j} > 0,\; U_j = \xi_{k,j} + a
& \text{(upper asymptote)}\\[2pt]
\xi_j, & g^{(o)}_{k,j} = 0 .
\end{cases}
$$

Let $t_j(\xi) = y_j(\xi) - y_j(\xi_k)$ and
$c_j = g_{k,j} \big/ \tfrac{dy_j}{d\xi_j}(\xi_k)$. The model of output $o$ is

$$\boxed{\;
m_o(\xi) \;=\; f_{k,o} \;+\; \sum_{j=1}^{d} c_{j,o}\, t_j(\xi)
\;+\; \tfrac12 \sum_{j=1}^{d} q_{j,o}\, t_j(\xi)^2 \;}
$$

with gradient
$\partial m_o/\partial \xi_j = \big(c_{j,o} + q_{j,o}\,t_j\big)\,\tfrac{dy_j}{d\xi_j}(\xi)$.
Since $t(\xi_k) = 0$, the anchor is exact by construction:
$m_o(\xi_k) = f_{k,o}$ and $\nabla m_o(\xi_k) = g_k^{(o)}$ — the
fully-linear-model requirement of trust-region theory, free because adjoint
gradients are available.

**Diagonal curvature fit.** With $w_i = w_i(\xi_k)$ frozen, and the gradient
residuals in $y$-space
$h_{i,j} = \big(\partial f/\partial y_j\big)\big|_{\xi_i} - c_j
= g_{i,j}\big/\tfrac{dy_j}{d\xi_j}(\xi_i) - c_j$:

* **gradient rows** ($d$ per sample, decoupled per coordinate):
  $t_{ij}\, q_j \approx h_{i,j}$, weight $w_i$;
* **value rows** (1 per sample, coupling all $q_j$; enabled per output by
  `fit_values` ∈ {`none`, `constraints` *(default)*, `all`}):
  $\tfrac12 \sum_j q_j t_{ij}^2 \approx r_i
  = y^{obs}_i - f_k - \textstyle\sum_j c_j t_{ij}$, weight
  $w_i^{v} = 4\,d\, w_i / \|t_i\|^2$.

The value-row weight removes the units imbalance (gradient equations are
$O(t)$, value equations $O(t^2)$ — raw least squares lets the gradients
drown the values at local distances) and gives one value equation the mass of
the sample's $d$ gradient equations. Gradient-only outputs solve the
decoupled quotient

$$q_j = \frac{\sum_i w_i\, t_{ij}\, h_{i,j}}{\sum_i w_i\, t_{ij}^2 + \varepsilon};$$

value-carrying outputs solve the coupled $d \times d$ normal system
(once per iteration, per output)

$$\Big[\operatorname{diag}\big(\textstyle\sum_i w_i t_{ij}^2\big)
+ \tfrac14 \textstyle\sum_i w_i^{v}\, v_i v_i^\top + \varepsilon I\Big]\, q
= \Big[\textstyle\sum_i w_i\, t_i \odot h_i
+ \tfrac12 \textstyle\sum_i w_i^{v}\, r_i\, v_i\Big],
\qquad v_i = t_i \odot t_i .$$

On a separable quadratic the fit is exact (test-enforced). The default
`min_fit_neighbors` = 1 keeps the tightest (most local) bandwidth; larger
values floor $h$ at the $k$-th-nearest-sample distance.

**Planar variant** (`model="planar"`, `PlanarMLSModel`): the per-query MLS of
§3 is solved **once at $\xi_k$ with frozen weights**, giving the affine model
$m(\xi) = \hat\alpha + \hat\beta^\top(\xi - \xi_k)$ (raw units). Its value
and slope blend neighbour values *and* gradients; it does not
hard-interpolate the incumbent (fully-linear accuracy, which trust-region
theory accepts) and has no curvature term.

## 5. Why the model is centre-frozen (consistency)

For a *moving* fit, the true derivative of the prediction is

$$\frac{d\,\alpha(z)}{dz} \;=\; \underbrace{\beta(z)}_{\text{diffuse}}
\;+\; \underbrace{\frac{\partial \alpha}{\partial w}\frac{\partial w}{\partial z}}_{\text{weight motion}},$$

and the diffuse part alone is **not** the derivative of the value actually
returned. Feeding an SQP solver the pair (moving value, diffuse gradient)
therefore hands it an inconsistent problem and corrupts its line searches —
a defect that was measured on the GGP benchmark. Freezing $w$ at $\xi_k$ per
iteration makes the model a polynomial, for which value and gradient are
exactly consistent; the diffuse derivative remains in use only inside the
heuristic batch acquisition (§7).

## 6. Subproblem, trust region, restarts

**Step subproblem** (`_solve_subproblem`), the whole step in the sequential
regime `batch_size=1`:

$$\xi^+ \;=\; \arg\min_{\xi}\; m_0(\xi)
\quad \text{s.t.} \quad m_j(\xi) \le 0,\;\;
\xi \in \big[\max(0, \xi_k - \delta),\, \min(1, \xi_k + \delta)\big],$$

solved by SLSQP with the models' analytic gradients, multistarted from
$\{\xi_k,\ \xi_{best},\ 2\ \text{random TR points}\}$; fallbacks in order:
phase-1 feasibility restoration
$\min \sum_j \max(0, m_j)^2$, then L-BFGS-B on the penalized model merit.
If $\xi^+ = \xi_k$ (model KKT point), a **null step** shrinks $\delta$
without spending a true evaluation.

**Acceptance and radius** — $\ell_1$ merit
$\varphi(\xi) = f(\xi) + \mu \sum_j \max(0, c_j(\xi))$, ratio

$$\rho_k = \frac{\varphi(\xi_k) - \varphi(\xi^+)}
{m^\varphi(\xi_k) - m^\varphi(\xi^+)},$$

with $m^\varphi$ the same merit on the model. Accept the centre move when
the actual reduction is positive and $\rho_k \ge \eta_a$ ($10^{-4}$);
three-zone radius update: shrink $\times$ 0.5 if the step failed or
$\rho_k < 0.25$, hold in the middle band, expand $\times$ 2 (capped) if
$\rho_k \ge 0.5$ and the step hit $\ge 0.8\,\delta$.

**Restarts** — on trust-region collapse ($\delta < \delta_{\min}$) or stall,
restart from the incumbent best with radius cycling
$\delta \leftarrow \delta_0 / 2^{(r-1) \bmod 3}$ and one randomized
evaluation inside the new region (so the refit differs from the run that
stalled), up to `n_resets` times — the full evaluation budget is always
spent, as a sequential optimizer such as MMA would.

## 7. Batch mode (optional, `batch_size` = q > 1)

Point 1 is the subproblem solution of §6; points $2..q$ minimize the
penalized lower-confidence-bound acquisition shared with GE-SBO
(`gesbo_core.propose_batch`, duck-typed on §3's surrogate):

$$A_j(\xi) = \hat m_0(\xi) - \kappa_j\,\sigma(\xi)
+ \varrho \sum_c \max\big(0, \hat m_c(\xi) + \text{shift}_c\big)^2
+ w_r \sum_{b < j} e^{-\|\xi - \xi_b\|^2 / 2h_r^2},
\qquad \kappa_j = \kappa_0\, \gamma_\kappa^{\,j-1},$$

with $\hat m$ the standardized per-query MLS means, $\sigma$ the density
proxy, and the last term a diversity repulsion from already-selected batch
members.

## 8. Reproducibility

The optimizer itself is deterministic (seeded RNG; deterministic linear
algebra). Bit-reproducibility of a full run additionally requires a
deterministic FEM backend: the `amjax` (JAX/XLA) solver carries a
per-process $O(10^{-13})$ perturbation from thread-pool reduction ordering,
which the chaotic trust-region trajectory amplifies into different local
minima; `fem_solver: direct` (scipy SuperLU) is verified bit-deterministic
end-to-end.

## 9. Defaults (config ↔ symbols)

| Symbol | Config field | Default |
|---|---|---|
| $\gamma_{ls}$ | `ls_factor` | 2.0 |
| $h_{\min}, h_{\max}$ | `ls_min`, `ls_max` | $10^{-3}$, 2.0 |
| window $N$ | `max_points` | 60 |
| $\varepsilon$ | `regularization` | $10^{-8}$ |
| intermediate variables | `intermediate` | `"linear"` |
| $a_0$ | `asy_init` | 0.5 |
| value rows | `fit_values` | `"constraints"` |
| bandwidth floor | `min_fit_neighbors` | 1 |
| model | `model` | `"quadratic"` |
| $\delta_0, \delta_{\min}, \delta_{\max}$ | `tr_init`, `tr_min`, `tr_max` | 0.25, $10^{-5}$, 0.75 |
| shrink / expand | `tr_shrink`, `tr_expand` | 0.5, 2.0 |
| $\eta_a$, shrink zone, expand zone | `eta_accept`, `eta_shrink`, `eta_expand` | $10^{-4}$, 0.25, 0.5 |
| $\mu$ | `penalty` | 100 |
| restarts | `n_resets` | 8 |
| $q$ | `batch_size` | 4 (`ggp` CLI); 1 = sequential regime |
| $\kappa_0, \gamma_\kappa, w_r$ | `kappa_base`, `kappa_growth`, `repulsion_weight` | 1, 2, 1 |
