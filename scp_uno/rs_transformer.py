# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Transformer proposer for the reduced-space optimizer — fully generic.

Dimension-agnostic learned optimizer built on the subspace formulation of
:mod:`scp_uno.reduced_space`: at each iteration the state is expressed in the
``r``-dimensional frame (steepest descent, orthogonalized constraint
gradient, momentum, secant — default ``r = 4``) and the policy predicts the
trust-region step ``alpha in [-1, 1]^r`` **plus a trust-radius multiplier**
(learned step-size control). The number of design variables never appears —
every feature is a projection onto the frame normalized by the trust radius
and the local gradient scales — so one trained model applies unchanged to
problems of any dimension.

**Teacher & training.** The privileged teacher solves the *true* subspace
subproblem by vectorized Sobol sampling + shrinking Gaussian refinement
(feasibility-first), and supplies a radius target (grow when the optimum sits
on the trust boundary, soft-shrink when it is small and interior). Training
rolls the *actual* driver with a recording noisy teacher on synthetic
families unrelated to topology optimization (free / linearly-constrained /
ball-constrained quadratics and curved valleys, dimensions 4..256; toy-SIMP
and all GGP data are **excluded** and kept as held-out tests). At run time
the transformer replaces the GEK sub-optimization at **zero inner-evaluation
cost**: one true evaluation per iteration.

Requires NumPy + JAX.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from scp_uno.mma_teacher import TeacherTask
from scp_uno.reduced_space import (
    IterationContext,
    ReducedSpaceConfig,
    ReducedSpaceDriver,
    ReducedSpaceResult,
    _feasibility_first_pick,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RSPolicyConfig:
    """Architecture of the reduced-space proposer network."""

    subspace_dim: int = 4      # must match ReducedSpaceConfig.subspace_dim
    context: int = 12          # history samples fed as tokens
    d_model: int = 64
    n_layers: int = 2
    n_attn_heads: int = 4
    d_mlp: int = 256

    @property
    def n_features(self) -> int:
        # [a (r), oop (1), f (1), grad_f (r), c (1), grad_c (r), flag, age, center]
        return 3 * self.subspace_dim + 6

    @property
    def n_outputs(self) -> int:
        return self.subspace_dim + 1        # step + log-radius multiplier


# --------------------------------------------------------------------------- #
# Tokenization (shared verbatim between training rollouts and inference)
# --------------------------------------------------------------------------- #
def build_rs_tokens(ctx: IterationContext, K: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the last ``K`` evaluated samples in the *current* frame.

    Per token: in-subspace coordinates and out-of-subspace distance
    (trust-radius units), objective value and projected gradient (scaled by
    ``||grad f|| * delta``), aggregated constraint value and projected
    gradient (scaled likewise), a constraint flag, age and a center flag.
    Dimensionless and independent of the design-space dimension.
    """
    drv = ctx.driver
    center = ctx.center
    r = ctx.dim
    F = 3 * r + 6
    s_f = float(np.linalg.norm(center.g)) * ctx.delta + 1e-300
    s_c = ((float(np.linalg.norm(center.g_agg)) * ctx.delta + 1e-300)
           if ctx.constrained else 1.0)

    samples = drv.samples[-K:]
    if center not in samples:
        samples = [center] + samples[1:]
    tokens = np.zeros((K, F))
    mask = np.zeros(K, dtype=bool)
    n_tot = len(drv.samples)
    for slot, s in enumerate(samples):
        a, oop = ctx.project(s)
        tokens[slot, 0:r] = np.clip(a, -3.0, 3.0)
        tokens[slot, r] = min(oop, 3.0)
        tokens[slot, r + 1] = np.clip((s.f - center.f) / s_f, -3.0, 3.0)
        tokens[slot, r + 2: 2 * r + 2] = np.clip(
            ctx.gradk(s, "f") * ctx.delta / s_f, -3.0, 3.0)
        if ctx.constrained and s.c_agg is not None:
            tokens[slot, 2 * r + 2] = np.clip(s.c_agg / s_c, -3.0, 3.0)
            tokens[slot, 2 * r + 3: 3 * r + 3] = np.clip(
                ctx.gradk(s, "c") * ctx.delta / s_c, -3.0, 3.0)
            tokens[slot, 3 * r + 3] = 1.0
        age = n_tot - 1 - drv.samples.index(s) if s in drv.samples else 0
        tokens[slot, 3 * r + 4] = min(age / max(K, 1), 2.0)
        tokens[slot, 3 * r + 5] = 1.0 if s is center else 0.0
        mask[slot] = True
    return tokens, mask


# --------------------------------------------------------------------------- #
# The network (pure JAX; query token reads out the step and radius signal)
# --------------------------------------------------------------------------- #
def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def init_params(cfg: RSPolicyConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def dense(shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    p: Dict[str, np.ndarray] = {
        "embed_w": dense((cfg.n_features, cfg.d_model), cfg.n_features),
        "embed_b": np.zeros(cfg.d_model, np.float32),
        "query": dense((1, cfg.d_model), cfg.d_model),
        "out_w": dense((cfg.d_model, cfg.n_outputs), cfg.d_model),
        "out_b": np.zeros(cfg.n_outputs, np.float32),
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


def forward(params, tokens, mask, cfg: RSPolicyConfig):
    """``(K, F), (K,) -> (r + 1,)`` in ``[-1, 1]`` (step coords + radius signal)."""
    jax, jnp = _jax()

    def layer_norm(h, g, b):
        mu = jnp.mean(h, axis=-1, keepdims=True)
        var = jnp.var(h, axis=-1, keepdims=True)
        return g * (h - mu) / jnp.sqrt(var + 1e-6) + b

    h = tokens @ params["embed_w"] + params["embed_b"]
    h = jnp.concatenate([params["query"], h], axis=0)
    full_mask = jnp.concatenate([jnp.ones(1, bool), mask])

    D, nh = cfg.d_model, cfg.n_attn_heads
    dh = D // nh
    neg = jnp.asarray(-1e9, h.dtype)
    for layer in range(cfg.n_layers):
        z = layer_norm(h, params[f"l{layer}_ln1_g"], params[f"l{layer}_ln1_b"])
        qkv = z @ params[f"l{layer}_qkv_w"] + params[f"l{layer}_qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(-1, nh, dh).transpose(1, 0, 2)
        k = k.reshape(-1, nh, dh).transpose(1, 0, 2)
        v = v.reshape(-1, nh, dh).transpose(1, 0, 2)
        att = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)
        att = jnp.where(full_mask[None, None, :], att, neg)
        att = jax.nn.softmax(att, axis=-1)
        y = (att @ v).transpose(1, 0, 2).reshape(-1, D)
        h = h + y @ params[f"l{layer}_proj_w"] + params[f"l{layer}_proj_b"]
        z = layer_norm(h, params[f"l{layer}_ln2_g"], params[f"l{layer}_ln2_b"])
        z = jax.nn.gelu(z @ params[f"l{layer}_mlp1_w"] + params[f"l{layer}_mlp1_b"])
        h = h + z @ params[f"l{layer}_mlp2_w"] + params[f"l{layer}_mlp2_b"]

    out = layer_norm(h[0], params["ln_f_g"], params["ln_f_b"])
    return jnp.tanh(out @ params["out_w"] + params["out_b"])


# --------------------------------------------------------------------------- #
# Vectorized teacher (true subspace subproblem) and training-data proposer
# --------------------------------------------------------------------------- #
def subspace_teacher_target(
    task: TeacherTask,
    ctx: IterationContext,
    rng: np.random.Generator,
    n_sobol: int = 2048,
    n_refine_rounds: int = 6,
    n_refine: int = 128,
) -> np.ndarray:
    """Feasibility-first solution of the true trust-region subproblem.

    Sobol exploration of the cube followed by shrinking Gaussian refinement
    around the incumbent; all through the task's vectorized values-only
    evaluation, so it is cheap on the synthetic training families.
    """
    r = ctx.dim

    def score(P: np.ndarray) -> np.ndarray:
        X = np.clip(
            ctx.center.x[None, :] + ctx.delta * (P @ ctx.E.T),
            ctx.driver.lb, ctx.driver.ub,
        )
        f, c = task.values_batch(X)
        return f, c

    sob = qmc.Sobol(d=r, scramble=True, seed=int(rng.integers(1 << 31)))
    P = np.concatenate([2.0 * sob.random(n_sobol) - 1.0,
                        np.eye(r), -np.eye(r), np.zeros((1, r))])
    f, c = score(P)
    best = _feasibility_first_pick(P, f, c)
    sigma = 0.3
    for _ in range(n_refine_rounds):
        Q = np.clip(best[None, :] + sigma * rng.standard_normal((n_refine, r)),
                    -1.0, 1.0)
        Q = np.concatenate([Q, best[None, :]])
        fq, cq = score(Q)
        best = _feasibility_first_pick(Q, fq, cq)
        sigma *= 0.5
    return best


def radius_signal(alpha_star: np.ndarray, boundary_frac: float = 0.9,
                  interior_frac: float = 0.3) -> float:
    """Teacher radius signal in [-1, 1]: grow on boundary optima, soft-shrink
    on small interior optima (mapped through ``gamma = gamma_max ** signal``)."""
    a_inf = float(np.max(np.abs(alpha_star)))
    if a_inf >= boundary_frac:
        return 0.5
    if a_inf <= interior_frac:
        return -0.4
    return 0.0


class RecordingTeacherProposer:
    """Training-rollout proposer: records (tokens, target) built with the
    inference tokenizer, then executes the (noisy) teacher step and its
    radius multiplier so the rollout dynamics match inference."""

    def __init__(self, task: TeacherTask, cfg: RSPolicyConfig,
                 rng: np.random.Generator, gamma_max: float = 2.5,
                 noise: float = 0.15):
        self.task = task
        self.cfg = cfg
        self.rng = rng
        self.gamma_max = gamma_max
        self.noise = noise
        self.records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def propose(self, ctx: IterationContext
                ) -> Tuple[np.ndarray, Optional[float]]:
        tokens, mask = build_rs_tokens(ctx, self.cfg.context)
        alpha = subspace_teacher_target(self.task, ctx, self.rng)
        t_gamma = radius_signal(alpha)
        target = np.concatenate([alpha, [t_gamma]]).astype(np.float32)
        self.records.append((tokens, mask, target))
        if self.rng.random() < 0.3:
            alpha = np.clip(
                alpha + self.noise * self.rng.standard_normal(ctx.dim), -1, 1)
        return alpha, float(self.gamma_max ** t_gamma)


def sample_rs_task(rng: np.random.Generator,
                   dims: Sequence[int] = (4, 8, 16, 32, 64, 128, 256)
                   ) -> TeacherTask:
    """Generic training families — deliberately excludes toy-SIMP (held out)
    and anything GGP-related."""
    kind = rng.choice(["quad_free", "quad_lin", "quad_ball", "valley"],
                      p=[0.25, 0.3, 0.25, 0.2])
    return TeacherTask(kind, int(rng.choice(dims)), rng)


def generate_training_batch(
    rng: np.random.Generator,
    cfg: RSPolicyConfig,
    n_states: int = 64,
    rollout_evals: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the recording noisy teacher through the *actual* driver on fresh
    synthetic tasks; returns ``(tokens, masks, targets)`` arrays."""
    toks: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    tgts: List[np.ndarray] = []
    while len(toks) < n_states:
        task = sample_rs_task(rng)
        d = task.d

        def ev(x, _t=task):
            f, g, c, gc = _t.evaluate(x)
            if c is None:
                return f, g, np.zeros(0), np.zeros((0, len(x)))
            return f, g, np.array([c]), gc[None, :]

        proposer = RecordingTeacherProposer(task, cfg, rng)
        driver = ReducedSpaceDriver(
            ev, np.zeros(d), np.ones(d), proposer,
            ReducedSpaceConfig(
                max_evals=rollout_evals,
                subspace_dim=cfg.subspace_dim,
                delta_init=float(10 ** rng.uniform(-2, -0.5)),
                seed=int(rng.integers(1 << 31)),
            ),
        )
        driver.run(rng.uniform(0.05, 0.95, d))
        for t, m, y in proposer.records:
            toks.append(t)
            masks.append(m)
            tgts.append(y)
            if len(toks) >= n_states:
                break
    return (np.asarray(toks, np.float32), np.asarray(masks, bool),
            np.asarray(tgts, np.float32))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_policy(
    cfg: RSPolicyConfig,
    steps: int = 3000,
    lr: float = 3e-4,
    batch_states: int = 64,
    gamma_weight: float = 0.3,
    seed: int = 0,
    log_every: int = 200,
    on_step: Optional[Callable[[int, float], None]] = None,
    checkpoint_path=None,
    checkpoint_every: int = 0,
) -> Dict[str, np.ndarray]:
    """MSE behaviour cloning of the subspace teacher (step + radius signal)."""
    jax, jnp = _jax()

    params = {k: jnp.asarray(v) for k, v in init_params(cfg, seed).items()}
    rng = np.random.default_rng(seed)
    r = cfg.subspace_dim
    w = jnp.concatenate([jnp.ones(r), jnp.asarray([gamma_weight])])

    def loss_fn(p, tokens, masks, targets):
        pred = jax.vmap(lambda t, m: forward(p, t, m, cfg))(tokens, masks)
        return jnp.mean(((pred - targets) ** 2) @ w) / (r + gamma_weight)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v_) for k, v_ in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for it in range(1, steps + 1):
        tokens, masks, targets = generate_training_batch(
            rng, cfg, n_states=batch_states
        )
        val, g = grad_fn(params, jnp.asarray(tokens), jnp.asarray(masks),
                         jnp.asarray(targets))
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - b1 ** it)) / (
                jnp.sqrt(v[k] / (1 - b2 ** it)) + eps)
        if on_step is not None:
            on_step(it, float(val))
        if log_every and it % log_every == 0:
            LOGGER.info("train step %d: loss=%.5f", it, float(val))
        if (checkpoint_path is not None and checkpoint_every
                and it % checkpoint_every == 0):
            save_params(checkpoint_path,
                        {k: np.asarray(v_) for k, v_ in params.items()}, cfg)

    return {k: np.asarray(v_) for k, v_ in params.items()}


def save_params(path, params: Dict[str, np.ndarray], cfg: RSPolicyConfig) -> None:
    meta = {f"cfg_{f.name}": np.asarray(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)}
    np.savez_compressed(path, **params, **meta, rs_version=np.asarray(2))


def load_params(path) -> Tuple[Dict[str, np.ndarray], RSPolicyConfig]:
    data = np.load(path)
    version = int(data["rs_version"]) if "rs_version" in data else 0
    if version != 2:
        raise ValueError(
            f"reduced-space policy weights at {path} are version {version}; "
            "retrain with scripts/train_rs_transformer.py (current: 2)."
        )
    cfg = RSPolicyConfig(**{
        f.name: int(data[f"cfg_{f.name}"])
        for f in dataclasses.fields(RSPolicyConfig)
        if f"cfg_{f.name}" in data
    })
    params = {k: data[k] for k in data.files
              if not (k.startswith("cfg_") or k == "rs_version")}
    return params, cfg


# --------------------------------------------------------------------------- #
# Inference proposer + functional entry point
# --------------------------------------------------------------------------- #
class TransformerRSProposer:
    """Predict the step and radius multiplier — zero inner evaluations."""

    def __init__(self, params: Dict[str, np.ndarray], cfg: RSPolicyConfig,
                 gamma_max: float = 2.5, use_policy_radius: bool = True):
        jax, jnp = _jax()
        p = {k: jnp.asarray(v) for k, v in params.items()}
        self.cfg = cfg
        self.gamma_max = gamma_max
        self.use_policy_radius = use_policy_radius
        self._fwd = jax.jit(lambda t, m: forward(p, t, m, cfg))

    def propose(self, ctx: IterationContext
                ) -> Tuple[np.ndarray, Optional[float]]:
        if ctx.dim != self.cfg.subspace_dim:
            raise ValueError(
                f"driver subspace_dim={ctx.dim} does not match the policy "
                f"({self.cfg.subspace_dim}); align ReducedSpaceConfig."
            )
        tokens, mask = build_rs_tokens(ctx, self.cfg.context)
        out = np.asarray(self._fwd(tokens, mask))
        if not self.use_policy_radius:
            return out[: ctx.dim], None
        return out[: ctx.dim], float(self.gamma_max ** out[ctx.dim])


def rs_transformer_minimize(
    evaluate,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    params: Dict[str, np.ndarray],
    policy_cfg: RSPolicyConfig,
    config: Optional[ReducedSpaceConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
    use_policy_radius: bool = True,
) -> ReducedSpaceResult:
    """Reduced-space optimization with the transformer proposer."""
    config = config or ReducedSpaceConfig()
    config.subspace_dim = policy_cfg.subspace_dim
    return ReducedSpaceDriver(
        evaluate, lower_bounds, upper_bounds,
        TransformerRSProposer(params, policy_cfg, config.gamma_max,
                              use_policy_radius),
        config, on_iteration,
    ).run(x0)
