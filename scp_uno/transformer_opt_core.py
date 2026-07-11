# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Transformer learned optimizer — per-variable-token core ("learned MMA").

Second-generation design of the ``TRANSFORMER_OPT`` engine, targeting MMA-level
performance on gradient-cheap problems such as GGP topology optimization:

* **One token per design variable** (not per history sample): the policy
  outputs a full-dimension step every iteration, exactly like MMA. Attention
  over the variable tokens supplies the global coupling (the constraint's
  dual multiplier is a global quantity), and the architecture remains
  dimension-agnostic because variables are a *set* of tokens.
* **Head 0 imitates MMA's step map.** A compact NumPy MMA teacher
  (:mod:`scp_uno.mma_teacher`, single-constraint dual bisection with the
  classical oscillation-adaptive asymptotes) is rolled out on synthetic task
  families — including a *toy-SIMP* family (``sum k_i/(x_i+eps)`` under a
  volume cap) that mirrors the reciprocal/volume structure of compliance
  topology optimization — and head 0 is trained on ``dx_MMA / move``.
  In sequential mode (``eval_heads=1``) the optimizer therefore moves on
  every evaluation, like MMA itself.
* **Heads 1..3 are far-sighted multi-scale proposals**: head ``j`` is trained
  on the trust-clipped direction to the true optimum at reach
  ``sigma_j in {1, 4, 16}`` move lengths. Evaluating several heads per
  iteration gives the multi-point batch mode.
* **Scale/oscillation state as features**: per-variable asymptote widths
  (maintained with the same classical update rule), previous step and
  oscillation sign — i.e. the sufficient statistics MMA itself uses — plus
  log-scaled gradient magnitudes and constraint-activity features, all
  dimensionless.
* Optional **fine-tuning on recorded GEMSEO-MMA trajectories** of the real
  problem family (``scripts/collect_ggp_mma_trajectories.py``) closes the
  domain gap between synthetic teachers and FEM compliance landscapes.

At run time a merit-based backtracking safeguard wraps the learned step, so
one bad proposal cannot derail the run. Requires NumPy + JAX only.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scp_uno.gesbo_core import EvaluateFn
from scp_uno.mma_teacher import AsymptoteTracker, mma_step, sample_teacher_task

LOGGER = logging.getLogger(__name__)

#: reach (in move lengths) of the far-sighted output heads 1..3
HEAD_SIGMAS = (1.0, 1.0, 4.0, 16.0)

N_FEATURES = 10


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PolicyConfig:
    """Architecture of the per-variable-token policy (fixed at training)."""

    n_heads_out: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_attn_heads: int = 4
    d_mlp: int = 256

    @property
    def n_features(self) -> int:
        return N_FEATURES


@dataclasses.dataclass
class TransformerOptConfig:
    """Rollout configuration of the learned optimizer."""

    max_evals: int = 200
    move: float = 0.01          # move limit (fraction of range), as in MMA
    eval_heads: int = 1         # 1 = sequential (MMA-like); k>1 = batch best-of-k
    n_backtracks: int = 2       # halvings tried when the step worsens the merit
    stall_limit: int = 8        # consecutive rejected iterations before stopping
    accept_mode: str = "always"   # "always": advance unconditionally like MMA
                                  # (best-so-far bookkeeping guards the result);
                                  # "watchdog": accept vs the worst of the last
                                  # `nonmonotone_window` merits; "monotone": strict
    nonmonotone_window: int = 10
    penalty: float = 10.0       # l1 merit safety factor over the multiplier estimate
    constraint_tol: float = 1e-6
    asy_incr: float = 1.2
    asy_decr: float = 0.4
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
# Tokenization (identical in training, GGP-trajectory replay and inference)
# --------------------------------------------------------------------------- #
def _slog(v: np.ndarray, scale: float) -> np.ndarray:
    """Signed log magnitude ``sign(v) * log1p(|v|/scale) / 4``, clipped."""
    return np.clip(np.sign(v) * np.log1p(np.abs(v) / max(scale, 1e-300)) / 4.0,
                   -2.5, 2.5)


def build_var_tokens(
    x: np.ndarray,
    grad_f: np.ndarray,
    c_val: Optional[float],
    grad_c: Optional[np.ndarray],
    width: np.ndarray,
    last_step: np.ndarray,     # fractions of range
    osc: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    move: float,
) -> np.ndarray:
    """Per-variable feature tokens ``(d, N_FEATURES)``, all dimensionless.

    Gradient magnitudes are signed-log scaled by their median (so orderings
    across many decades survive), the constraint value is expressed in
    "move-steps to the boundary", and the asymptote width / previous step /
    oscillation triplet is the same sufficient statistic MMA's own update
    uses. The encoding is invariant to affine objective rescaling and to the
    variable count.
    """
    d = len(x)
    rng = np.maximum(ub - lb, 1e-30)
    g0r = np.asarray(grad_f, float) * rng
    # mean (not median) scale: converged designs have many near-zero gradients,
    # a median scale would collapse and saturate the features of the active set
    s0 = float(np.mean(np.abs(g0r))) + 1e-300

    tok = np.zeros((d, N_FEATURES))
    tok[:, 0] = _slog(g0r, s0)
    if c_val is not None and grad_c is not None:
        g1r = np.asarray(grad_c, float) * rng
        s1 = float(np.mean(np.abs(g1r))) + 1e-300
        tok[:, 1] = _slog(g1r, s1)
        tok[:, 2] = np.clip(
            np.sign(c_val) * np.log1p(abs(c_val) / (s1 * move)) / 4.0, -2.5, 2.5
        )
        tok[:, 3] = 1.0
    tok[:, 4] = np.log(np.clip(width / (move * rng), 1e-3, 10.0)) / 2.0
    tok[:, 5] = np.clip(last_step / move, -1.5, 1.5)
    tok[:, 6] = osc
    tok[:, 7] = (x - lb) / rng
    tok[:, 8] = np.minimum((ub - x) / (move * rng), 1.0)
    tok[:, 9] = np.minimum((x - lb) / (move * rng), 1.0)
    return tok


# --------------------------------------------------------------------------- #
# The network (pure JAX, parameters as a flat dict of arrays)
# --------------------------------------------------------------------------- #
def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def init_params(cfg: PolicyConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def dense(shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    p: Dict[str, np.ndarray] = {
        "embed_w": dense((cfg.n_features, cfg.d_model), cfg.n_features),
        "embed_b": np.zeros(cfg.d_model, np.float32),
        "out_w": dense((cfg.d_model, cfg.n_heads_out), cfg.d_model),
        "out_b": np.zeros(cfg.n_heads_out, np.float32),
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


def forward(params, tokens, cfg: PolicyConfig):
    """Policy forward pass: ``(d, F) -> (q, d)`` steps in ``[-1, 1]``.

    Self-attention over the variable tokens (permutation-equivariant: permuting
    variables permutes the steps), per-token linear read-out of the ``q`` heads.
    JAX-traceable; batch with ``jax.vmap``.
    """
    jax, jnp = _jax()

    def layer_norm(h, g, b):
        mu = jnp.mean(h, axis=-1, keepdims=True)
        var = jnp.var(h, axis=-1, keepdims=True)
        return g * (h - mu) / jnp.sqrt(var + 1e-6) + b

    h = tokens @ params["embed_w"] + params["embed_b"]          # (d, D)
    D, nh = cfg.d_model, cfg.n_attn_heads
    dh = D // nh

    for layer in range(cfg.n_layers):
        z = layer_norm(h, params[f"l{layer}_ln1_g"], params[f"l{layer}_ln1_b"])
        qkv = z @ params[f"l{layer}_qkv_w"] + params[f"l{layer}_qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(-1, nh, dh).transpose(1, 0, 2)
        k = k.reshape(-1, nh, dh).transpose(1, 0, 2)
        v = v.reshape(-1, nh, dh).transpose(1, 0, 2)
        att = jax.nn.softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(dh), axis=-1)
        y = (att @ v).transpose(1, 0, 2).reshape(-1, D)
        h = h + y @ params[f"l{layer}_proj_w"] + params[f"l{layer}_proj_b"]
        z = layer_norm(h, params[f"l{layer}_ln2_g"], params[f"l{layer}_ln2_b"])
        z = jax.nn.gelu(z @ params[f"l{layer}_mlp1_w"] + params[f"l{layer}_mlp1_b"])
        h = h + z @ params[f"l{layer}_mlp2_w"] + params[f"l{layer}_mlp2_b"]

    out = layer_norm(h, params["ln_f_g"], params["ln_f_b"])
    steps = out @ params["out_w"] + params["out_b"]              # (d, q)
    return jnp.tanh(steps).T                                     # (q, d)


# --------------------------------------------------------------------------- #
# Training data: teacher rollouts on synthetic tasks
# --------------------------------------------------------------------------- #
def generate_training_batch(
    rng: np.random.Generator,
    cfg: PolicyConfig,
    d: int,
    n_states: int = 48,
    rollout_len: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Roll the MMA teacher out on synthetic tasks of dimension ``d``.

    Returns ``(tokens (S, d, F), targets (S, q, d))``. Target of head 0 is
    the teacher's own step in move units; heads 1..3 get the trust-clipped
    direction to the known optimum at reaches ``HEAD_SIGMAS[1:]``. A fraction
    of the transitions is perturbed so off-trajectory states are covered.
    """
    S_tok: List[np.ndarray] = []
    S_tgt: List[np.ndarray] = []
    lb, ub = np.zeros(d), np.ones(d)

    while len(S_tok) < n_states:
        task = sample_teacher_task(rng, dims=(d,))
        move = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.15)))
        tracker = AsymptoteTracker(
            lb, ub, asy_init=move, asy_min=0.01 * move, asy_max=move
        )
        x = rng.uniform(0.05, 0.95, d)
        tracker.update(x)
        for _ in range(rollout_len):
            f, g, c, gc = task.evaluate(x)
            tok = build_var_tokens(
                x, g, c, gc, tracker.width, tracker.last_step(),
                tracker.oscillation(), lb, ub, move,
            )
            dx = mma_step(x, g, lb, ub, tracker.width, move, c, gc)
            tgt = np.empty((cfg.n_heads_out, d))
            # head 0 in per-variable *width* units: O(1) in every regime, so
            # late tiny-asymptote steps carry as much training signal as early ones
            tgt[0] = np.clip(dx / tracker.width, -1.0, 1.0)
            for j in range(1, cfg.n_heads_out):
                sig = HEAD_SIGMAS[min(j, len(HEAD_SIGMAS) - 1)]
                tgt[j] = np.clip((task.x_star - x) / (sig * move), -1.0, 1.0)
            S_tok.append(tok)
            S_tgt.append(tgt)
            if len(S_tok) >= n_states:
                break
            if rng.random() < 0.15:       # exploration: off-trajectory states
                dx = dx + 0.5 * tracker.width * rng.standard_normal(d)
            x = np.clip(x + dx, lb, ub)
            tracker.update(x)

    return np.asarray(S_tok, np.float32), np.asarray(S_tgt, np.float32)


def replay_trajectory_states(
    X: np.ndarray,           # (n, d) recorded iterates (normalized [0,1] space)
    G0: np.ndarray,          # (n, d) objective gradients
    C: np.ndarray,           # (n,)   constraint values (c <= 0), or empty
    G1: np.ndarray,          # (n, d) constraint gradients, or empty
    move: float,
    cfg: PolicyConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a recorded optimizer trajectory (e.g. GEMSEO-MMA on the GGP
    pipeline) into training states.

    Head-0 targets are the *recorded* steps in move units; far-sighted heads
    target the trajectory's final iterate (the converged design). The
    asymptote-width features are replayed with the same classical rule.
    """
    n, d = X.shape
    lb, ub = np.zeros(d), np.ones(d)
    has_c = len(C) == n
    tracker = AsymptoteTracker(lb, ub, asy_init=move, asy_min=0.01 * move,
                               asy_max=move)
    x_final = X[-1]
    toks, tgts = [], []
    for k in range(n - 1):
        tracker.update(X[k])
        tok = build_var_tokens(
            X[k], G0[k], float(C[k]) if has_c else None,
            G1[k] if has_c else None, tracker.width, tracker.last_step(),
            tracker.oscillation(), lb, ub, move,
        )
        tgt = np.empty((cfg.n_heads_out, d))
        tgt[0] = np.clip((X[k + 1] - X[k]) / tracker.width, -1.0, 1.0)
        for j in range(1, cfg.n_heads_out):
            sig = HEAD_SIGMAS[min(j, len(HEAD_SIGMAS) - 1)]
            tgt[j] = np.clip((x_final - X[k]) / (sig * move), -1.0, 1.0)
        toks.append(tok)
        tgts.append(tgt)
    return np.asarray(toks, np.float32), np.asarray(tgts, np.float32)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_policy(
    cfg: PolicyConfig,
    steps: int = 4000,
    lr: float = 3e-4,
    batch_states: int = 48,
    dims: Sequence[int] = (16, 32, 64, 108, 160),
    seed: int = 0,
    extra_data: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    extra_prob: float = 0.3,
    log_every: int = 200,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, np.ndarray]:
    """Behaviour cloning of the MMA teacher (+ optional recorded trajectories).

    ``extra_data`` is a list of ``(tokens, targets)`` state pools (e.g. from
    :func:`replay_trajectory_states` on GGP runs); with probability
    ``extra_prob`` a training batch is drawn from those pools instead of
    fresh synthetic rollouts. Loss: per-head MSE with double weight on the
    MMA-imitation head 0.
    """
    jax, jnp = _jax()

    params = {k: jnp.asarray(v) for k, v in init_params(cfg, seed).items()}
    rng = np.random.default_rng(seed)

    def loss_fn(p, tokens, targets):
        pred = jax.vmap(lambda t: forward(p, t, cfg))(tokens)    # (B, q, d)
        err = jnp.mean((pred - targets) ** 2, axis=-1)           # (B, q)
        w = jnp.asarray([2.0] + [1.0] * (cfg.n_heads_out - 1))
        return jnp.mean(err @ w) / (2.0 + cfg.n_heads_out - 1)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v_) for k, v_ in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    pools = list(extra_data or [])
    last = float("nan")
    for it in range(1, steps + 1):
        if pools and rng.random() < extra_prob:
            toks, tgts = pools[rng.integers(len(pools))]
            idx = rng.integers(0, len(toks), size=min(batch_states, len(toks)))
            tokens, targets = toks[idx], tgts[idx]
        else:
            d = int(rng.choice(dims))
            tokens, targets = generate_training_batch(
                rng, cfg, d, n_states=batch_states
            )
        val, g = grad_fn(params, jnp.asarray(tokens), jnp.asarray(targets))
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
            LOGGER.info("train step %d: loss=%.5f", it, last)

    return {k: np.asarray(v_) for k, v_ in params.items()}


def save_params(path, params: Dict[str, np.ndarray], cfg: PolicyConfig) -> None:
    meta = {f"cfg_{f.name}": np.asarray(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)}
    np.savez_compressed(path, **params, **meta, cfg_version=np.asarray(2))


def load_params(path) -> Tuple[Dict[str, np.ndarray], PolicyConfig]:
    data = np.load(path)
    version = int(data["cfg_version"]) if "cfg_version" in data else 1
    if version != 2:
        raise ValueError(
            f"policy weights at {path} are version {version}; retrain with "
            "scripts/train_transformer_opt.py (current version: 2)."
        )
    cfg = PolicyConfig(**{
        f.name: int(data[f"cfg_{f.name}"])
        for f in dataclasses.fields(PolicyConfig)
        if f"cfg_{f.name}" in data
    })
    params = {k: data[k] for k in data.files
              if not (k.startswith("cfg_") or k == "cfg_version")}
    return params, cfg


# --------------------------------------------------------------------------- #
# Rollout driver
# --------------------------------------------------------------------------- #
class TransformerOptimizer:
    """Run the trained per-variable policy with a merit backtracking safeguard.

    Evaluation contract as in :class:`scp_uno.gesbo_core.GESBOptimizer`
    (``evaluate: x -> (f, grad_f, cons, jac_cons)``, ``c(x) <= 0``).
    With ``eval_heads=1`` (default) the optimizer moves on every evaluation
    like MMA; with ``eval_heads=k`` it evaluates the first ``k`` heads
    (MMA-imitation + far-sighted multi-scale proposals) and takes the best.
    Multiple constraints are reduced to the most-active one for the policy
    features (the merit safeguard still sees them all).
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
        self.range = np.maximum(self.ub - self.lb, 1e-30)
        self.pcfg = policy_cfg
        self.cfg = config or TransformerOptConfig()
        self.on_iteration = on_iteration

        jax, jnp = _jax()
        p = {k: jnp.asarray(v) for k, v in params.items()}
        self._policy = jax.jit(lambda tokens: forward(p, tokens, self.pcfg))

        self.X: List[np.ndarray] = []
        self.F: List[float] = []
        self.C: List[np.ndarray] = []
        self.n_evals = 0
        # adaptive l1 penalty: ratcheted to penalty * (multiplier estimate), so
        # feasibility-restoring steps are accepted whatever the objective scale
        self._mu = 0.0

    def _update_mu(self, g: np.ndarray, J: np.ndarray) -> None:
        if len(J):
            s0 = float(np.mean(np.abs(g * self.range)))
            s1 = float(np.mean(np.abs(J * self.range[None, :])))
            self._mu = max(self._mu, self.cfg.penalty * s0 / max(s1, 1e-300))

    def _eval(self, x: np.ndarray):
        f, g, c, J = self.evaluate(x)
        g = np.asarray(g, float).flatten()
        c = np.asarray(c, float).flatten()
        J = np.asarray(J, float).reshape(len(c), len(x))
        self.X.append(np.asarray(x, float).copy())
        self.F.append(float(f))
        self.C.append(c)
        self.n_evals += 1
        return len(self.X) - 1, f, g, c, J

    def _merit_of(self, f: float, c: np.ndarray) -> float:
        viol = float(np.sum(np.maximum(0.0, c))) if len(c) else 0.0
        return f + self._mu * viol

    def _violation(self, i: int) -> float:
        return float(np.max(self.C[i])) if len(self.C[i]) else -np.inf

    def _best_index(self) -> int:
        feas = [i for i in range(len(self.X))
                if self._violation(i) <= self.cfg.constraint_tol]
        if feas:
            return min(feas, key=lambda i: self.F[i])
        return min(range(len(self.X)), key=self._violation)

    def run(self, x0: np.ndarray) -> TransformerOptResult:
        cfg = self.cfg
        d = len(self.lb)
        move = cfg.move
        tracker = AsymptoteTracker(
            self.lb, self.ub, asy_init=move, asy_min=0.01 * move,
            asy_max=move, incr=cfg.asy_incr, decr=cfg.asy_decr,
        )
        history: List[dict] = []
        status = "evaluation budget exhausted"

        x = np.clip(np.asarray(x0, float), self.lb, self.ub)
        _, f, g, c, J = self._eval(x)
        tracker.update(x)
        stall = 0
        it = 0
        recent_fc: List[Tuple[float, np.ndarray]] = [(f, c)]   # watchdog window

        while self.n_evals < cfg.max_evals:
            # refresh the adaptive penalty, then the acceptance thresholds under it
            self._update_mu(g, J)
            merit = self._merit_of(f, c)
            if cfg.accept_mode == "always":
                accept_ref = np.inf
            elif cfg.accept_mode == "watchdog" and cfg.nonmonotone_window > 0:
                accept_ref = max(self._merit_of(fw, cw) for fw, cw in recent_fc)
            else:
                accept_ref = merit
            # most-active constraint for the policy features
            if len(c):
                ic = int(np.argmax(c))
                c_val, g_c = float(c[ic]), J[ic]
            else:
                c_val, g_c = None, None
            tokens = build_var_tokens(
                x, g, c_val, g_c, tracker.width, tracker.last_step(),
                tracker.oscillation(), self.lb, self.ub, move,
            )
            steps = np.asarray(self._policy(tokens))             # (q, d)

            # candidate points from the first eval_heads heads
            n_heads = max(1, min(cfg.eval_heads, self.pcfg.n_heads_out))
            best = None                                           # (merit, idx, f, g, c, J)
            for j in range(n_heads):
                if self.n_evals >= cfg.max_evals:
                    break
                if j == 0:      # MMA-imitation head steps in width units
                    raw = tracker.width * steps[0]
                else:           # far-sighted heads step in move units
                    sig = HEAD_SIGMAS[min(j, len(HEAD_SIGMAS) - 1)]
                    raw = sig * move * self.range * steps[j]
                x_c = np.clip(x + raw, self.lb, self.ub)
                idx, f_c, g_new, c_c, J_c = self._eval(x_c)
                m_c = self._merit_of(f_c, c_c)
                if best is None or m_c < best[0]:
                    best = (m_c, idx, f_c, g_new, c_c, J_c)

            if best is None:
                break

            # backtracking safeguard on the best candidate
            n_bt = 0
            while (best[0] > accept_ref and n_bt < cfg.n_backtracks
                   and self.n_evals < cfg.max_evals):
                n_bt += 1
                x_c = x + (self.X[best[1]] - x) * 0.5 ** n_bt
                idx, f_c, g_new, c_c, J_c = self._eval(x_c)
                m_c = self._merit_of(f_c, c_c)
                if m_c < best[0]:
                    best = (m_c, idx, f_c, g_new, c_c, J_c)

            if best[0] <= accept_ref + 1e-12 * (1.0 + abs(accept_ref)):
                x = self.X[best[1]].copy()
                f, g, c, J = best[2], best[3], best[4], best[5]
                tracker.update(x)
                recent_fc.append((f, c))
                if len(recent_fc) > max(cfg.nonmonotone_window, 1):
                    recent_fc.pop(0)
                stall = 0
            else:
                stall += 1
                if stall >= cfg.stall_limit:
                    status = "stalled (no merit improvement)"
                    break

            it += 1
            rec = {
                "iter": it, "n_evals": self.n_evals,
                "merit_center": merit, "best_f": self.F[self._best_index()],
                "stall": stall,
            }
            history.append(rec)
            if self.on_iteration is not None:
                self.on_iteration(rec)

        ib = self._best_index()
        return TransformerOptResult(
            x_opt=self.X[ib].copy(),
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
    """Functional entry point of the learned optimizer."""
    return TransformerOptimizer(
        evaluate, lower_bounds, upper_bounds, params, policy_cfg, config,
        on_iteration,
    ).run(x0)
