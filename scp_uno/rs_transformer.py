# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Transformer proposer for the reduced-space (2D) optimizer — fully generic.

This is the dimension-agnostic learned optimizer built on the reduced-space
formulation of :mod:`scp_uno.reduced_space`: at each iteration the state is
expressed in the 2D frame ``(e1, e2)`` spanned by the objective gradient and
the orthogonalized aggregated-constraint gradient, and the policy predicts
the next step ``(alpha, beta)`` inside the trust region. The number of design
variables never appears anywhere — every feature is a projection onto the
frame, normalized by the trust radius and the local gradient scales — so one
trained model applies unchanged to problems of any dimension.

**Teacher & training.** The privileged teacher solves the *true* 2D
subproblem by a dense grid on the plane (feasibility-first: best feasible
objective, else least violation) — exactly what the GEK proposer approximates
with inner evaluations. Training rolls the *actual* driver with a recording
noisy-teacher proposer on synthetic task families that are unrelated to
topology optimization (free / linearly-constrained / ball-constrained
quadratics and curved valleys, dimensions 4..256; the toy-SIMP family and any
GGP data are deliberately **excluded** and kept as held-out tests), and the
policy learns ``tokens -> (alpha, beta)`` by MSE. At run time the transformer
replaces the GEK sub-optimization at **zero inner-evaluation cost**: one true
evaluation per iteration.

Requires NumPy + JAX.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scp_uno.mma_teacher import TeacherTask
from scp_uno.reduced_space import (
    IterationContext,
    ReducedSpaceConfig,
    ReducedSpaceDriver,
    ReducedSpaceResult,
    _feasibility_first_pick,
)

LOGGER = logging.getLogger(__name__)

N_FEATURES = 12


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RSPolicyConfig:
    """Architecture of the reduced-space proposer network."""

    context: int = 12          # history samples fed as tokens
    d_model: int = 64
    n_layers: int = 2
    n_attn_heads: int = 4
    d_mlp: int = 256

    @property
    def n_features(self) -> int:
        return N_FEATURES


# --------------------------------------------------------------------------- #
# Tokenization (shared verbatim between training rollouts and inference)
# --------------------------------------------------------------------------- #
def build_rs_tokens(ctx: IterationContext, K: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the last ``K`` evaluated samples in the *current* 2D frame.

    Per token: in-plane coordinates and out-of-plane distance (trust-radius
    units), objective value and its 2D gradient (scaled by ``||grad f|| *
    delta`` — the linear objective-change scale over the trust region),
    aggregated constraint value and 2D gradient (scaled likewise), a
    constraint flag, age and a center flag. Dimensionless and independent of
    the design-space dimension by construction.
    """
    drv = ctx.driver
    center = ctx.center
    s_f = float(np.linalg.norm(center.g)) * ctx.delta + 1e-300
    s_c = ((float(np.linalg.norm(center.g_agg)) * ctx.delta + 1e-300)
           if ctx.constrained else 1.0)

    samples = drv.samples[-K:]
    if center not in samples:
        samples = [center] + samples[1:]
    tokens = np.zeros((K, N_FEATURES))
    mask = np.zeros(K, dtype=bool)
    n_tot = len(drv.samples)
    for slot, s in enumerate(samples):
        a, b, r = ctx.project(s)
        ga, gb = ctx.grad2d(s, "f")
        tokens[slot, 0] = np.clip(a, -3.0, 3.0)
        tokens[slot, 1] = np.clip(b, -3.0, 3.0)
        tokens[slot, 2] = np.clip(r, 0.0, 3.0)
        tokens[slot, 3] = np.clip((s.f - center.f) / s_f, -3.0, 3.0)
        tokens[slot, 4] = np.clip(ga * ctx.delta / s_f, -3.0, 3.0)
        tokens[slot, 5] = np.clip(gb * ctx.delta / s_f, -3.0, 3.0)
        if ctx.constrained and s.c_agg is not None:
            ca, cb = ctx.grad2d(s, "c")
            tokens[slot, 6] = np.clip(s.c_agg / s_c, -3.0, 3.0)
            tokens[slot, 7] = np.clip(ca * ctx.delta / s_c, -3.0, 3.0)
            tokens[slot, 8] = np.clip(cb * ctx.delta / s_c, -3.0, 3.0)
            tokens[slot, 9] = 1.0
        age = n_tot - 1 - drv.samples.index(s) if s in drv.samples else 0
        tokens[slot, 10] = min(age / max(K, 1), 2.0)
        tokens[slot, 11] = 1.0 if s is center else 0.0
        mask[slot] = True
    return tokens, mask


# --------------------------------------------------------------------------- #
# The network (pure JAX; query token reads out (alpha, beta))
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
        "out_w": dense((cfg.d_model, 2), cfg.d_model),
        "out_b": np.zeros(2, np.float32),
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
    """``(K, F), (K,) -> (2,)`` step in ``[-1, 1]^2`` (JAX-traceable)."""
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
# Grid teacher (true 2D subproblem) and training-data proposer
# --------------------------------------------------------------------------- #
def grid_teacher_target(task: TeacherTask, ctx: IterationContext,
                        n_grid: int = 21) -> Tuple[float, float]:
    """Solve the true 2D subproblem on a dense grid (feasibility-first)."""
    ax = np.linspace(-1.0, 1.0, n_grid)
    A, B = np.meshgrid(ax, ax, indexing="ij")
    P = np.column_stack([A.ravel(), B.ravel()])
    X = (ctx.center.x[None, :]
         + ctx.delta * (P[:, :1] * ctx.e1[None, :] + P[:, 1:] * ctx.e2[None, :]))
    X = np.clip(X, ctx.driver.lb, ctx.driver.ub)
    f_true, c_true = task.values_batch(X)
    return _feasibility_first_pick(P, f_true, c_true)


class RecordingTeacherProposer:
    """Proposer used during training rollouts: records (tokens, target) built
    with the *inference* tokenizer, then executes the teacher step with noise
    so off-policy states are also covered."""

    def __init__(self, task: TeacherTask, cfg: RSPolicyConfig,
                 rng: np.random.Generator, n_grid: int = 21,
                 noise: float = 0.15):
        self.task = task
        self.cfg = cfg
        self.rng = rng
        self.n_grid = n_grid
        self.noise = noise
        self.records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def propose(self, ctx: IterationContext) -> Tuple[float, float]:
        tokens, mask = build_rs_tokens(ctx, self.cfg.context)
        target = grid_teacher_target(self.task, ctx, self.n_grid)
        self.records.append((tokens, mask, np.asarray(target, np.float32)))
        a, b = target
        if self.rng.random() < 0.3:
            a = float(np.clip(a + self.noise * self.rng.standard_normal(), -1, 1))
            b = float(np.clip(b + self.noise * self.rng.standard_normal(), -1, 1))
        return a, b


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
    steps: int = 4000,
    lr: float = 3e-4,
    batch_states: int = 64,
    seed: int = 0,
    log_every: int = 200,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, np.ndarray]:
    """MSE behaviour cloning of the grid teacher in the reduced space."""
    jax, jnp = _jax()

    params = {k: jnp.asarray(v) for k, v in init_params(cfg, seed).items()}
    rng = np.random.default_rng(seed)

    def loss_fn(p, tokens, masks, targets):
        pred = jax.vmap(lambda t, m: forward(p, t, m, cfg))(tokens, masks)
        return jnp.mean((pred - targets) ** 2)

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

    return {k: np.asarray(v_) for k, v_ in params.items()}


def save_params(path, params: Dict[str, np.ndarray], cfg: RSPolicyConfig) -> None:
    meta = {f"cfg_{f.name}": np.asarray(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)}
    np.savez_compressed(path, **params, **meta, rs_version=np.asarray(1))


def load_params(path) -> Tuple[Dict[str, np.ndarray], RSPolicyConfig]:
    data = np.load(path)
    if "rs_version" not in data:
        raise ValueError(f"{path} is not a reduced-space policy weight file.")
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
    """Predict ``(alpha, beta)`` from the history — zero inner evaluations."""

    def __init__(self, params: Dict[str, np.ndarray], cfg: RSPolicyConfig):
        jax, jnp = _jax()
        p = {k: jnp.asarray(v) for k, v in params.items()}
        self.cfg = cfg
        self._fwd = jax.jit(lambda t, m: forward(p, t, m, cfg))

    def propose(self, ctx: IterationContext) -> Tuple[float, float]:
        tokens, mask = build_rs_tokens(ctx, self.cfg.context)
        out = np.asarray(self._fwd(tokens, mask))
        return float(out[0]), float(out[1])


def rs_transformer_minimize(
    evaluate,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    params: Dict[str, np.ndarray],
    policy_cfg: RSPolicyConfig,
    config: Optional[ReducedSpaceConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
) -> ReducedSpaceResult:
    """Reduced-space optimization with the transformer proposer."""
    return ReducedSpaceDriver(
        evaluate, lower_bounds, upper_bounds,
        TransformerRSProposer(params, policy_cfg), config, on_iteration,
    ).run(x0)
