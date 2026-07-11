# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Transformer-based learned optimizer ("learning to optimize") — core engine.

This module implements approach "B": instead of a hand-designed acquisition
(GE-SBO) or convex approximation (MMA), a **small set-transformer policy**
reads the recent optimization history and directly proposes the next **batch**
of query points. The policy is trained offline by imitating a *privileged
teacher* on synthetic differentiable tasks whose optimum is known, then reused
as-is on real problems (the GGP FEM pipeline) through the ``TRANSFORMER_OPT``
GEMSEO algorithm in :mod:`scp_uno.transformer_opt`.

Design choices that make one trained model transfer across problems:

* **Dimension-agnostic latent**: all features live in the gradient
  *active subspace* ``z = W^T x`` of fixed size ``r`` (reusing
  :func:`scp_uno.gesbo_core.active_subspace`), padded when ``d < r``.
  The design-space dimension never appears in the network.
* **Scale invariance**: token features are trust-region-relative
  (positions divided by the radius) and merit-scale-normalized (values and
  gradients divided by the window's merit spread), and actions are steps in
  ``[-1, 1]^r`` *relative to the trust region*. A policy trained on toy
  functions therefore sees exactly the same feature distribution on a FEM
  compliance landscape.
* **Set structure**: history samples are tokens with no positional encoding
  (recency enters as an explicit ``age`` feature), so the policy is
  permutation-invariant over the history, and a learned query token reads out
  ``q`` proposal heads — the multi-point batch.
* **Trust-region safeguard**: the classical accept / shrink / expand loop
  wraps the learned proposer, so a bad proposal costs one batch, not the run.

Training is behaviour cloning with a winner-takes-all loss: the teacher's
target is the (trust-region-clipped) step towards the true optimum, the loss
is ``min_j ||s_j - t||^2`` over the ``q`` heads (plus a small mean term), so
heads specialize on the distinct modes seen across multimodal tasks.

Everything here needs only NumPy + JAX (no GEMSEO, no FEniCS); JAX is used
for the network and its training. Inference reuses the same JAX forward.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from scp_uno.gesbo_core import EvaluateFn, active_subspace

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PolicyConfig:
    """Architecture of the proposer network (fixed at training time)."""

    latent_dim: int = 12       # r: active-subspace size seen by the policy
    context: int = 16          # K: history tokens fed to the policy
    n_heads_out: int = 4       # q: proposal heads = batch size per iteration
    d_model: int = 64
    n_layers: int = 2
    n_attn_heads: int = 4
    d_mlp: int = 256

    @property
    def n_features(self) -> int:
        # [z_off (r), f_rel (1), grad (r), viol (1), age (1), is_center (1)]
        return 2 * self.latent_dim + 4


@dataclasses.dataclass
class TransformerOptConfig:
    """Rollout / trust-region configuration of the learned optimizer."""

    max_evals: int = 200
    max_outer_iter: int = 200
    tr_init: float = 0.25
    tr_min: float = 1e-5
    tr_max: float = 0.75
    tr_shrink: float = 0.5
    tr_expand: float = 2.0
    penalty: float = 100.0          # l1 merit weight for constraints
    constraint_tol: float = 1e-6
    seed: int = 0


@dataclasses.dataclass
class TransformerOptResult:
    x_opt: np.ndarray
    f_opt: float
    constraints: np.ndarray
    is_feasible: bool
    n_evals: int
    n_iter: int
    status: str
    history: List[dict]


# --------------------------------------------------------------------------- #
# Tokenization (shared verbatim between training and inference)
# --------------------------------------------------------------------------- #
def latent_basis(gradients: np.ndarray, d: int, r: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Basis ``W (d, r)`` of the policy latent: active subspace, zero-padded
    when the design space is smaller than the latent."""
    if d <= r:
        W = np.zeros((d, r))
        W[:, :d] = np.eye(d)
        return W
    return active_subspace(np.atleast_2d(gradients), r, rng)


def build_tokens(
    X: np.ndarray,          # (n, d) normalized samples, most recent last
    merit: np.ndarray,      # (n,) merit values (objective + penalty)
    G: np.ndarray,          # (n, d) merit gradients
    viol: np.ndarray,       # (n,) max constraint violation (<=0 if feasible)
    center_idx: int,
    W: np.ndarray,          # (d, r) latent basis
    delta: float,
    cfg: PolicyConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the most recent ``K`` samples as policy tokens.

    Returns ``(tokens (K, F), mask (K,))`` with padding rows masked out. All
    features are trust-region-relative and merit-scale-normalized, so the
    encoding is invariant to affine rescaling of the objective and to the
    design-space dimension.
    """
    K, r = cfg.context, cfg.latent_dim
    n = len(X)
    take = np.arange(max(0, n - K), n)
    # always keep the center in context even if it scrolled out of the window
    if center_idx not in take:
        take = np.concatenate([[center_idx], take[1:]])

    fc = merit[center_idx]
    scale = float(np.std(merit[take]))
    scale = scale if scale > 1e-12 else max(abs(fc), 1.0)

    tokens = np.zeros((K, cfg.n_features))
    mask = np.zeros(K, dtype=bool)
    for slot, i in enumerate(take):
        z_off = np.clip(W.T @ (X[i] - X[center_idx]) / delta, -3.0, 3.0)
        f_rel = np.clip((merit[i] - fc) / scale, -3.0, 3.0)
        g_tok = np.clip((W.T @ G[i]) * delta / scale, -3.0, 3.0)
        age = (n - 1 - i) / max(K, 1)
        tokens[slot, :r] = z_off
        tokens[slot, r] = f_rel
        tokens[slot, r + 1 : 2 * r + 1] = g_tok
        tokens[slot, 2 * r + 1] = 1.0 if viol[i] > 0 else 0.0
        tokens[slot, 2 * r + 2] = age
        tokens[slot, 2 * r + 3] = 1.0 if i == center_idx else 0.0
        mask[slot] = True
    return tokens, mask


# --------------------------------------------------------------------------- #
# The network (pure JAX, parameters as a flat dict of arrays)
# --------------------------------------------------------------------------- #
def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def init_params(cfg: PolicyConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    """Initialize transformer parameters (returned as NumPy arrays)."""
    rng = np.random.default_rng(seed)

    def dense(shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    p: Dict[str, np.ndarray] = {
        "embed_w": dense((cfg.n_features, cfg.d_model), cfg.n_features),
        "embed_b": np.zeros(cfg.d_model, np.float32),
        "query": dense((1, cfg.d_model), cfg.d_model),
        "out_w": dense((cfg.d_model, cfg.n_heads_out * cfg.latent_dim), cfg.d_model),
        "out_b": np.zeros(cfg.n_heads_out * cfg.latent_dim, np.float32),
    }
    for layer in range(cfg.n_layers):
        p[f"l{layer}_qkv_w"] = dense((cfg.d_model, 3 * cfg.d_model), cfg.d_model)
        p[f"l{layer}_qkv_b"] = np.zeros(3 * cfg.d_model, np.float32)
        p[f"l{layer}_proj_w"] = dense((cfg.d_model, cfg.d_model), cfg.d_model)
        p[f"l{layer}_proj_b"] = np.zeros(cfg.d_model, np.float32)
        p[f"l{layer}_mlp1_w"] = dense((cfg.d_model, cfg.d_mlp), cfg.d_model)
        p[f"l{layer}_mlp1_b"] = np.zeros(cfg.d_mlp, np.float32)
        p[f"l{layer}_mlp2_w"] = dense((cfg.d_mlp, cfg.d_model), cfg.d_mlp)
        p[f"l{layer}_mlp2_b"] = np.zeros(cfg.d_model, np.float32)
        for ln in ("ln1", "ln2"):
            p[f"l{layer}_{ln}_g"] = np.ones(cfg.d_model, np.float32)
            p[f"l{layer}_{ln}_b"] = np.zeros(cfg.d_model, np.float32)
    p["ln_f_g"] = np.ones(cfg.d_model, np.float32)
    p["ln_f_b"] = np.zeros(cfg.d_model, np.float32)
    return p


def forward(params, tokens, mask, cfg: PolicyConfig):
    """Policy forward pass: ``(K, F), (K,) -> (q, r)`` steps in ``[-1, 1]``.

    JAX-traceable; batch with ``jax.vmap``. The learned query token attends to
    the (masked) history tokens through ``n_layers`` pre-norm blocks, and a
    linear head reads out the ``q`` proposals.
    """
    jax, jnp = _jax()

    def layer_norm(x, g, b):
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - mu) / jnp.sqrt(var + 1e-6) + b

    h = tokens @ params["embed_w"] + params["embed_b"]          # (K, D)
    h = jnp.concatenate([params["query"], h], axis=0)            # (K+1, D)
    full_mask = jnp.concatenate([jnp.ones(1, bool), mask])       # query visible

    D = cfg.d_model
    nh = cfg.n_attn_heads
    dh = D // nh
    neg = jnp.asarray(-1e9, h.dtype)

    for layer in range(cfg.n_layers):
        x = layer_norm(h, params[f"l{layer}_ln1_g"], params[f"l{layer}_ln1_b"])
        qkv = x @ params[f"l{layer}_qkv_w"] + params[f"l{layer}_qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(-1, nh, dh).transpose(1, 0, 2)             # (nh, K+1, dh)
        k = k.reshape(-1, nh, dh).transpose(1, 0, 2)
        v = v.reshape(-1, nh, dh).transpose(1, 0, 2)
        att = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)           # (nh, K+1, K+1)
        att = jnp.where(full_mask[None, None, :], att, neg)
        att = jax.nn.softmax(att, axis=-1)
        y = (att @ v).transpose(1, 0, 2).reshape(-1, D)
        h = h + y @ params[f"l{layer}_proj_w"] + params[f"l{layer}_proj_b"]
        x = layer_norm(h, params[f"l{layer}_ln2_g"], params[f"l{layer}_ln2_b"])
        x = jax.nn.gelu(x @ params[f"l{layer}_mlp1_w"] + params[f"l{layer}_mlp1_b"])
        h = h + x @ params[f"l{layer}_mlp2_w"] + params[f"l{layer}_mlp2_b"]

    out = layer_norm(h[0], params["ln_f_g"], params["ln_f_b"])
    steps = out @ params["out_w"] + params["out_b"]
    return jnp.tanh(steps).reshape(cfg.n_heads_out, cfg.latent_dim)


# --------------------------------------------------------------------------- #
# Synthetic task families (privileged teacher: the optimum is known)
# --------------------------------------------------------------------------- #
class _Task:
    """Differentiable synthetic problem with known optimum in [0, 1]^d."""

    def __init__(self, kind: str, d: int, rng: np.random.Generator):
        self.d = d
        self.kind = kind
        self.x_star = rng.uniform(0.15, 0.85, d)
        if kind == "quadratic":
            k = rng.integers(2, min(d, 10) + 1)
            B = rng.standard_normal((d, k)) / np.sqrt(d)
            self.A = B @ B.T + 10 ** rng.uniform(-3, -1) * np.eye(d)
            self.scale = 10 ** rng.uniform(-1, 2)
        elif kind == "two_well":
            self.x_star2 = np.clip(
                self.x_star + rng.uniform(-0.6, 0.6, d), 0.05, 0.95
            )
            self.depth2 = rng.uniform(0.2, 0.9)     # second well is shallower
            self.scale = 10 ** rng.uniform(-1, 2)
        elif kind == "valley":                       # Rosenbrock-like curved valley
            self.R, _ = np.linalg.qr(rng.standard_normal((d, d)))
            self.scale = 10 ** rng.uniform(-2, 0)
        else:
            raise ValueError(kind)

    def evaluate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.kind == "quadratic":
            e = x - self.x_star
            return self.scale * float(e @ self.A @ e), self.scale * 2 * self.A @ e
        if self.kind == "two_well":
            e1, e2 = x - self.x_star, x - self.x_star2
            f1, f2 = float(e1 @ e1), self.depth2 + float(e2 @ e2)
            if f1 <= f2:
                return self.scale * f1, self.scale * 2 * e1
            return self.scale * f2, self.scale * 2 * e2
        # curved valley: f = sum 100 (y[i+1] - y[i]^2)^2 + (y[i])^2, y = R(x-x*)
        y = self.R @ (x - self.x_star)
        a, b = y[:-1], y[1:]
        t = b - a * a
        f = float(100 * t @ t + a @ a)
        gy = np.zeros(self.d)
        gy[:-1] = -400 * t * a + 2 * a
        gy[1:] += 200 * t
        return self.scale * f, self.scale * self.R.T @ gy


def _sample_task(rng: np.random.Generator, r: int) -> _Task:
    kind = rng.choice(["quadratic", "two_well", "valley"], p=[0.5, 0.3, 0.2])
    # cover d < r, d ~ r and d >> r so both padding and reduction are learned
    d = int(rng.choice([max(2, r // 2), r, 2 * r, 4 * r, 8 * r]))
    return _Task(kind, d, rng)


def generate_training_batch(
    rng: np.random.Generator,
    cfg: PolicyConfig,
    n_tasks: int = 8,
    rollout_len: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a noisy teacher on fresh synthetic tasks and record states.

    Returns ``(tokens (S, K, F), masks (S, K), targets (S, r))`` where each
    state's target is the trust-region-clipped latent step towards the true
    optimum — exactly what the winner-takes-all loss trains heads to hit.
    The rollout follows the teacher with noise (plus trust-region updates),
    so recorded states match the distribution the policy visits at test time.
    """
    K, r = cfg.context, cfg.latent_dim
    S: List[np.ndarray] = []
    M: List[np.ndarray] = []
    T: List[np.ndarray] = []

    for _ in range(n_tasks):
        task = _sample_task(rng, r)
        d = task.d
        delta = float(rng.uniform(0.05, 0.4))
        X: List[np.ndarray] = []
        F: List[float] = []
        G: List[np.ndarray] = []
        center = rng.uniform(0.0, 1.0, d)

        def record(x):
            f, g = task.evaluate(x)
            X.append(x.copy())
            F.append(f)
            G.append(g)
            return len(X) - 1

        ci = record(center)
        for _ in range(rollout_len):
            Xa, Fa, Ga = np.asarray(X), np.asarray(F), np.asarray(G)
            W = latent_basis(Ga[-K:], d, r, rng)
            tokens, mask = build_tokens(
                Xa, Fa, Ga, np.full(len(Xa), -1.0), ci, W, delta, cfg
            )
            target = np.clip(W.T @ (task.x_star - Xa[ci]) / delta, -1.0, 1.0)
            S.append(tokens)
            M.append(mask)
            T.append(target)

            # noisy-teacher transition (records realistic accept/shrink states)
            step = np.clip(target + 0.3 * rng.standard_normal(r), -1.0, 1.0)
            x_new = np.clip(Xa[ci] + delta * (W @ step), 0.0, 1.0)
            j = record(x_new)
            if F[j] < F[ci]:
                ci = j
                if np.max(np.abs(step)) > 0.9:
                    delta = min(delta * 2.0, 0.5)
            else:
                delta = max(delta * 0.5, 1e-3)

    return (
        np.asarray(S, np.float32),
        np.asarray(M, bool),
        np.asarray(T, np.float32),
    )


# --------------------------------------------------------------------------- #
# Training (behaviour cloning, winner-takes-all over the q heads)
# --------------------------------------------------------------------------- #
def train_policy(
    cfg: PolicyConfig,
    steps: int = 3000,
    lr: float = 3e-4,
    n_tasks: int = 8,
    rollout_len: int = 10,
    seed: int = 0,
    log_every: int = 200,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, np.ndarray]:
    """Train the proposer by imitation of the privileged teacher.

    Loss per state: ``min_j ||s_j - t||^2 + 0.05 * mean_j ||s_j - t||^2``.
    The winner-takes-all term lets heads specialize (multimodal tasks pull
    different heads towards different basins); the small mean term keeps
    losing heads from drifting. Returns the trained parameters as NumPy.
    """
    jax, jnp = _jax()

    params = {k: jnp.asarray(v) for k, v in init_params(cfg, seed).items()}
    rng = np.random.default_rng(seed)

    def loss_fn(p, tokens, masks, targets):
        pred = jax.vmap(lambda t, m: forward(p, t, m, cfg))(tokens, masks)
        err = jnp.sum((pred - targets[:, None, :]) ** 2, axis=-1)   # (S, q)
        return jnp.mean(jnp.min(err, axis=1)) + 0.05 * jnp.mean(err)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # hand-rolled Adam (keeps dependencies to jax alone)
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v_) for k, v_ in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    last = float("nan")
    for it in range(1, steps + 1):
        tokens, masks, targets = generate_training_batch(
            rng, cfg, n_tasks=n_tasks, rollout_len=rollout_len
        )
        val, g = grad_fn(params, jnp.asarray(tokens), jnp.asarray(masks),
                         jnp.asarray(targets))
        last = float(val)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            params[k] = params[k] - lr * mh / (jnp.sqrt(vh) + eps)
        if on_step is not None:
            on_step(it, last)
        if log_every and it % log_every == 0:
            LOGGER.info("train step %d: loss=%.4f", it, last)

    return {k: np.asarray(v_) for k, v_ in params.items()}


def save_params(path, params: Dict[str, np.ndarray], cfg: PolicyConfig) -> None:
    meta = {f"cfg_{f.name}": np.asarray(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)}
    np.savez_compressed(path, **params, **meta)


def load_params(path) -> Tuple[Dict[str, np.ndarray], PolicyConfig]:
    data = np.load(path)
    cfg = PolicyConfig(**{
        f.name: int(data[f"cfg_{f.name}"])
        for f in dataclasses.fields(PolicyConfig)
        if f"cfg_{f.name}" in data
    })
    params = {k: data[k] for k in data.files if not k.startswith("cfg_")}
    return params, cfg


# --------------------------------------------------------------------------- #
# Rollout driver (trust-region-safeguarded learned proposer)
# --------------------------------------------------------------------------- #
class TransformerOptimizer:
    """Run the trained proposer inside the classical trust-region loop.

    Same evaluation contract as :class:`scp_uno.gesbo_core.GESBOptimizer`
    (``evaluate: x -> (f, grad_f, cons, jac_cons)``, ``c(x) <= 0``); the
    policy proposes ``q`` points per iteration (the batch), the best batch
    point by l1 merit moves the center, otherwise the radius shrinks.
    """

    def __init__(
        self,
        evaluate: EvaluateFn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        params: Dict[str, np.ndarray],
        policy_cfg: PolicyConfig,
        config: Optional[TransformerOptConfig] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.evaluate = evaluate
        self.lb = np.asarray(lower_bounds, float)
        self.ub = np.asarray(upper_bounds, float)
        self.scale = np.maximum(self.ub - self.lb, 1e-30)
        self.pcfg = policy_cfg
        self.cfg = config or TransformerOptConfig()
        self.on_iteration = on_iteration
        self._rng = np.random.default_rng(self.cfg.seed)

        jax, jnp = _jax()
        p = {k: jnp.asarray(v) for k, v in params.items()}
        self._policy = jax.jit(
            lambda tokens, mask: forward(p, tokens, mask, self.pcfg)
        )

        self.X: List[np.ndarray] = []
        self.F: List[float] = []        # raw objective
        self.C: List[np.ndarray] = []
        self.Gm: List[np.ndarray] = []  # merit gradient
        self.n_evals = 0

    # ------------------------------------------------------------- sampling
    def _eval(self, xi: np.ndarray) -> int:
        x = self.lb + self.scale * xi
        f, g, c, J = self.evaluate(x)
        g = np.asarray(g, float).flatten() * self.scale
        c = np.asarray(c, float).flatten()
        J = (np.asarray(J, float).reshape(len(c), len(self.lb))
             * self.scale[None, :])
        gm = g.copy()
        for i in range(len(c)):
            if c[i] > 0:
                gm += self.cfg.penalty * J[i]
        self.X.append(np.asarray(xi, float).copy())
        self.F.append(float(f))
        self.C.append(c)
        self.Gm.append(gm)
        self.n_evals += 1
        return len(self.X) - 1

    def _merit(self, i: int) -> float:
        viol = float(np.sum(np.maximum(0.0, self.C[i]))) if len(self.C[i]) else 0.0
        return self.F[i] + self.cfg.penalty * viol

    def _violation(self, i: int) -> float:
        return float(np.max(self.C[i])) if len(self.C[i]) else -np.inf

    def _best_index(self) -> int:
        feas = [i for i in range(len(self.X))
                if self._violation(i) <= self.cfg.constraint_tol]
        if feas:
            return min(feas, key=lambda i: self.F[i])
        return min(range(len(self.X)), key=self._violation)

    # ----------------------------------------------------------------- run
    def run(self, x0: np.ndarray) -> TransformerOptResult:
        cfg, pcfg = self.cfg, self.pcfg
        d = len(self.lb)
        K, r = pcfg.context, pcfg.latent_dim
        delta = cfg.tr_init
        history: List[dict] = []
        status = "max_outer_iter reached"

        xi0 = np.clip((np.asarray(x0, float) - self.lb) / self.scale, 0.0, 1.0)
        ci = self._eval(xi0)

        for it in range(cfg.max_outer_iter):
            if self.n_evals >= cfg.max_evals:
                status = "evaluation budget exhausted"
                break
            if delta < cfg.tr_min:
                status = "trust region collapsed"
                break

            Xa = np.asarray(self.X)
            Fm = np.asarray([self._merit(i) for i in range(len(self.X))])
            Ga = np.asarray(self.Gm)
            Va = np.asarray([self._violation(i) for i in range(len(self.X))])
            W = latent_basis(Ga[-K:], d, r, self._rng)
            tokens, mask = build_tokens(Xa, Fm, Ga, Va, ci, W, delta, pcfg)
            steps = np.asarray(self._policy(tokens, mask))       # (q, r)

            # ---- evaluate the proposed batch (independent: parallelizable) ----
            new_idx: List[int] = []
            for s in steps:
                if self.n_evals >= cfg.max_evals:
                    break
                xi = np.clip(Xa[ci] + delta * (W @ s), 0.0, 1.0)
                if any(np.linalg.norm(xi - self.X[j]) < 1e-12 for j in new_idx):
                    xi = np.clip(
                        xi + 0.1 * delta * self._rng.standard_normal(d) / np.sqrt(d),
                        0.0, 1.0,
                    )
                new_idx.append(self._eval(xi))
            if not new_idx:
                status = "evaluation budget exhausted"
                break

            # ---- trust-region acceptance on the merit ----
            cand = min(new_idx, key=self._merit)
            step_inf = float(np.max(np.abs(self.X[cand] - Xa[ci])))
            if self._merit(cand) < self._merit(ci):
                ci = cand
                if step_inf >= 0.8 * delta:
                    delta = min(delta * cfg.tr_expand, cfg.tr_max)
            else:
                delta *= cfg.tr_shrink

            rec = {
                "iter": it, "n_evals": self.n_evals, "delta": delta,
                "batch_indices": new_idx,
                "merit_center": self._merit(ci),
                "best_f": self.F[self._best_index()],
            }
            history.append(rec)
            if self.on_iteration is not None:
                self.on_iteration(rec)
            LOGGER.info(
                "TransformerOpt iter %d: evals=%d f_best=%.6e merit=%.6e delta=%.3e",
                it, self.n_evals, rec["best_f"], rec["merit_center"], delta,
            )

        ib = self._best_index()
        return TransformerOptResult(
            x_opt=self.lb + self.scale * self.X[ib],
            f_opt=self.F[ib],
            constraints=self.C[ib].copy(),
            is_feasible=self._violation(ib) <= cfg.constraint_tol,
            n_evals=self.n_evals,
            n_iter=len(history),
            status=status,
            history=history,
        )


def transformer_opt_minimize(
    evaluate: EvaluateFn,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    params: Dict[str, np.ndarray],
    policy_cfg: PolicyConfig,
    config: Optional[TransformerOptConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
) -> TransformerOptResult:
    """Functional entry point mirroring :func:`scp_uno.gesbo_core.gesbo_minimize`."""
    return TransformerOptimizer(
        evaluate, lower_bounds, upper_bounds, params, policy_cfg, config,
        on_iteration,
    ).run(x0)
