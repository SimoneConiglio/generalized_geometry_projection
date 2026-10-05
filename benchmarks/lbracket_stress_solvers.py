# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Open-source optimizers on the density L-bracket with LOCAL stress constraints.

Same problem as ``lbracket_density_uno.py`` (mass minimization, one relaxed von Mises
constraint per element, no aggregation; n design variables = m constraints), solved by:

  full Jacobian (n adjoint solves per iteration, dense m x n matrix)
    ipopt         IPOPT 3.14 (cyipopt), L-BFGS Hessian, MUMPS
    nlopt-mma     NLopt MMA / CCSA (Svanberg 2002)
    paropt-mma    ParOpt MMA (MPI-parallel vectors, dense constraints)
    paropt-tr     ParOpt trust region + interior point, L-BFGS
    uno-*         Uno 2.9 presets (results of lbracket_density_uno.py)
  Jacobian-free (1 primal + 1 adjoint solve per iteration)
    al-mma        aggregation-free augmented Lagrangian + MMA (Senhora et al. 2020,
                  PolyStress, Giraldo-Londono & Paulino 2021)
    tao-almm      PETSc/TAO ALMM (PHR augmented Lagrangian, BQNLS bound-constrained
                  quasi-Newton subsolver), matrix-free constraint Jacobian

Each (method, size) runs in its own process (peak memory per run) and logs
(time, volume, max g) at every primal FE solve.

    python -m benchmarks.lbracket_stress_solvers --methods al-mma nlopt-mma --nelx 40 60
    python -m benchmarks.lbracket_stress_solvers --report
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.lbracket_density_uno import LBracketDensity

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "lbracket_stress_solvers_results.jsonl"

MAX_ITER = 300          # outer iterations, full-Jacobian methods
AL_STEPS, AL_INNER = 150, 10  # PolyStress: 150 AL updates (x 5 MMA steps; 10 here)
TIME_LIMIT = 3600.0


class Tracked:
    """Problem wrapper recording every primal solve (time, volume, max g)."""

    def __init__(self, nelx):
        self.p = LBracketDensity(nelx)
        self.n = self.p.n
        self.t0 = time.perf_counter()
        self.hist = []
        self._last = None

    def g(self, x):
        g = self.p.constraints(x)
        if self._last is None or not np.array_equal(x, self._last):
            self._last = x.copy()
            self.hist.append((time.perf_counter() - self.t0, self.p.volume(x),
                              float(np.max(g))))
        return g

    def expired(self):
        return time.perf_counter() - self.t0 > TIME_LIMIT


# --------------------------------------------------------------------------- #
# Jacobian-free: augmented Lagrangian + MMA (PolyStress)
# --------------------------------------------------------------------------- #
def mma_box_step(x, df, low, upp, it, xold1, xold2, xmin, xmax, move=0.1,
                 asyinit=0.2, asyincr=1.2, asydecr=0.7):
    """One MMA step for a bound-constrained problem: closed form, element-wise."""
    span = xmax - xmin
    if it < 2:
        low = x - asyinit * span
        upp = x + asyinit * span
    else:
        s = (x - xold1) * (xold1 - xold2)
        gam = np.where(s > 0, asyincr, np.where(s < 0, asydecr, 1.0))
        low = x - gam * (xold1 - low)
        upp = x + gam * (upp - xold1)
        low = np.clip(low, x - 10 * span, x - 0.01 * span)
        upp = np.clip(upp, x + 0.01 * span, x + 10 * span)
    alpha = np.maximum.reduce([xmin, low + 0.1 * (x - low), x - move * span])
    beta = np.minimum.reduce([xmax, upp - 0.1 * (upp - x), x + move * span])
    reg = 1e-5 / span
    p = (upp - x) ** 2 * (1.001 * np.maximum(df, 0) + 0.001 * np.maximum(-df, 0) + reg)
    q = (x - low) ** 2 * (0.001 * np.maximum(df, 0) + 1.001 * np.maximum(-df, 0) + reg)
    sp, sq = np.sqrt(p), np.sqrt(q)
    xnew = np.clip((sp * low + sq * upp) / (sp + sq), alpha, beta)
    return xnew, low, upp


def run_al_mma(tp: Tracked, mu0=10.0, alpha=1.05, mu_max=1e4, move=0.05):
    """PolyStress AL; tuned on n = 1 200: 10 inner MMA steps, alpha 1.05, move 0.05
    (PolyStress: 5, 1.1, -) -- best V within 1 % violation 0.3856 vs 0.4180."""
    p, n = tp.p, tp.n
    x = np.ones(n)
    xmin, xmax = np.full(n, p.xmin), np.ones(n)
    lam = np.zeros(n)
    mu = mu0
    dV = p.volume_grad()
    low = upp = None
    xold1 = xold2 = x.copy()
    it = 0
    for k in range(AL_STEPS):
        for _ in range(AL_INNER):
            g = tp.g(x)
            w = np.maximum(lam + mu * g, 0.0) / n          # d(AL penalty)/dg
            df = dV + p.jacobian_T_vec(x, w)
            xnew, low, upp = mma_box_step(x, df, low, upp, it, xold1, xold2, xmin, xmax,
                                          move=move)
            xold2, xold1, x = xold1, x, xnew
            it += 1
        g = tp.g(x)
        lam = np.maximum(lam + mu * g, 0.0)
        mu = min(alpha * mu, mu_max)
        if tp.expired():
            break
    return x, {"iterations": it, "status": f"{k + 1} AL steps"}


# --------------------------------------------------------------------------- #
# Jacobian-free: PETSc/TAO ALMM (matrix-free, slack form g(x) + s = 0, s >= 0)
# --------------------------------------------------------------------------- #
def run_tao_almm(tp: Tracked, max_it=150, sub="bqnls", sub_it=10, extra=None,
                 fscale=None):
    from petsc4py import PETSc

    p, n = tp.p, tp.n
    N = 2 * n                                       # [x, s]
    z = PETSc.Vec().createSeq(N)
    za = z.getArray()
    za[:n] = 1.0
    za[n:] = np.maximum(-tp.g(np.ones(n)), 0.0)
    lb = PETSc.Vec().createSeq(N)
    ub = PETSc.Vec().createSeq(N)
    lb.getArray()[:n] = p.xmin
    lb.getArray()[n:] = 0.0
    ub.getArray()[:n] = 1.0
    ub.getArray()[n:] = 1e3
    fscale = float(n) if fscale is None else fscale      # sum of densities: O(1) gradient
    dV = fscale * p.volume_grad()

    def objgrad(tao, zz, G):
        za = zz.getArray(readonly=True)
        G.getArray()[:n] = dV
        G.getArray()[n:] = 0.0
        return fscale * p.volume(np.array(za[:n]))

    def cons(tao, zz, C):
        za = zz.getArray(readonly=True)
        C.getArray()[:] = tp.g(np.array(za[:n])) + za[n:]

    class Shell:                                    # [J  I]
        def __init__(self):
            self.x = np.ones(n)

        def mult(self, mat, v, y):
            va = v.getArray(readonly=True)
            y.getArray()[:] = p.jacobian_vec(self.x, np.array(va[:n])) + va[n:]

        def multTranspose(self, mat, v, y):
            va = np.array(v.getArray(readonly=True))
            ya = y.getArray()
            ya[:n] = p.jacobian_T_vec(self.x, va)
            ya[n:] = va

    shell = Shell()
    J = PETSc.Mat().createPython([n, N], context=shell)
    J.setUp()

    def jac(tao, zz, A, P):
        shell.x = np.array(zz.getArray(readonly=True)[:n])
        tp.g(shell.x)

    ce = PETSc.Vec().createSeq(n)
    tao = PETSc.TAO().create(PETSc.COMM_SELF)
    tao.setType("almm")
    tao.setSolution(z)
    tao.setVariableBounds(lb, ub)
    tao.setObjectiveGradient(objgrad)
    tao.setEqualityConstraints(cons, ce)
    tao.setJacobianEquality(jac, J, J)
    opts = PETSc.Options()
    opts["tao_almm_type"] = "phr"
    opts["tao_almm_subsolver_tao_type"] = sub
    opts["tao_almm_subsolver_tao_max_it"] = sub_it
    for k, v in (extra or {}).items():
        opts[k] = v
    opts["tao_max_it"] = max_it
    opts["tao_gatol"] = 1e-6
    opts["tao_catol"] = 1e-3
    tao.setFromOptions()
    tao.setMaximumIterations(max_it)
    tao.solve()
    x = np.clip(np.array(tao.getSolution().getArray()[:n]), p.xmin, 1.0)
    return x, {"iterations": int(tao.getIterationNumber()),
               "status": str(tao.getConvergedReason())}


# --------------------------------------------------------------------------- #
# Full Jacobian: IPOPT, NLopt MMA, ParOpt
# --------------------------------------------------------------------------- #
def run_ipopt(tp: Tracked):
    import cyipopt

    p, n = tp.p, tp.n
    scale = float(n)                                # minimize the SUM of densities
    dV = scale * p.volume_grad()
    rows, cols = np.unravel_index(np.arange(n * n), (n, n))

    class NLP:
        def objective(self, x):
            return scale * p.volume(x)

        def gradient(self, x):
            return dV

        def constraints(self, x):
            return tp.g(x)

        def jacobian(self, x):
            tp.g(x)
            return p.jacobian(x).ravel()

        def jacobianstructure(self):
            return rows, cols

        def intermediate(self, *args):
            return not tp.expired()

    nlp = cyipopt.Problem(n=n, m=n, problem_obj=NLP(), lb=np.full(n, p.xmin),
                          ub=np.ones(n), cl=np.full(n, -1e20), cu=np.zeros(n))
    nlp.add_option("hessian_approximation", "limited-memory")
    nlp.add_option("max_iter", MAX_ITER)
    nlp.add_option("tol", 1e-6)
    nlp.add_option("print_level", 3)
    nlp.add_option("linear_solver", "mumps")
    x, info = nlp.solve(np.ones(n))
    return x, {"iterations": int(info.get("iter_count", -1)) if isinstance(info, dict) else -1,
               "status": info["status_msg"].decode() if isinstance(info["status_msg"], bytes)
               else str(info["status_msg"])}


def run_nlopt_mma(tp: Tracked):
    import nlopt

    p, n = tp.p, tp.n
    dV = p.volume_grad()
    opt = nlopt.opt(nlopt.LD_MMA, n)
    iters = [0]

    def f(x, grad):
        if grad.size:
            grad[:] = dV
            iters[0] += 1
        return p.volume(x)

    def c(result, x, grad):
        result[:] = tp.g(x)
        if grad.size:
            grad[:] = p.jacobian(x)

    opt.set_min_objective(f)
    opt.add_inequality_mconstraint(c, np.full(n, 1e-6))
    opt.set_lower_bounds(np.full(n, p.xmin))
    opt.set_upper_bounds(np.ones(n))
    opt.set_maxeval(3 * MAX_ITER)
    opt.set_maxtime(TIME_LIMIT)
    opt.set_ftol_rel(1e-6)
    try:
        x = opt.optimize(np.ones(n))
        status = str(opt.last_optimize_result())
    except Exception as e:           # nlopt raises on roundoff-limited etc.
        x = tp._last if tp._last is not None else np.ones(n)
        status = f"{type(e).__name__}: {e}"
    return x, {"iterations": iters[0], "status": status}


def run_paropt(tp: Tracked, algorithm):
    from mpi4py import MPI
    from paropt import ParOpt

    p, n = tp.p, tp.n
    dV = p.volume_grad()

    class Prob(ParOpt.Problem):
        def __init__(self):
            super().__init__(MPI.COMM_SELF, nvars=n, ncon=n)

        def getVarsAndBounds(self, x, lb, ub):
            x[:] = 1.0
            lb[:] = p.xmin
            ub[:] = 1.0

        def evalObjCon(self, x):
            g = tp.g(np.array(x))
            return 0, p.volume(np.array(x)), -g          # ParOpt: c(x) >= 0

        def evalObjConGradient(self, x, gr, A):
            xa = np.array(x)
            gr[:] = dV
            J = p.jacobian(xa)
            for i in range(n):
                A[i][:] = -J[i]
            return 0

    opts = {"algorithm": algorithm, "output_file": "/dev/null",
            "tr_output_file": "/dev/null", "mma_output_file": "/dev/null"}
    if algorithm == "mma":
        opts.update({"mma_max_iterations": MAX_ITER, "mma_move_limit": 0.1})
    else:
        opts.update({"tr_init_size": 0.05, "tr_max_size": 0.1, "tr_max_iterations": MAX_ITER,
                     "qn_type": "bfgs", "penalty_gamma": 1000.0})
    prob = Prob()                     # keep a reference: ParOpt holds a raw pointer
    opt = ParOpt.Optimizer(prob, opts)
    opt.optimize()
    x = np.array(opt.getOptimizedPoint()[0])
    return x, {"iterations": len(tp.hist), "status": "done"}


METHODS = {
    "al-mma": run_al_mma,
    "tao-almm": run_tao_almm,
    "ipopt": run_ipopt,
    "nlopt-mma": run_nlopt_mma,
    "paropt-mma": lambda tp: run_paropt(tp, "mma"),
    "paropt-tr": lambda tp: run_paropt(tp, "tr"),
}


def run_one(method, nelx):
    tp = Tracked(nelx)
    x, info = METHODS[method](tp)
    wall = time.perf_counter() - tp.t0
    x = np.clip(np.asarray(x, dtype=float), tp.p.xmin, 1.0)
    g = tp.p.constraints(x)
    row = {
        "method": method, "nelx": nelx, "n": tp.n, **info,
        "volume": tp.p.volume(x), "max_g": float(np.max(g)),
        "n_fe": tp.p.n_fe, "n_rhs": tp.p.n_rhs,
        "wall_s": wall, "fe_s": tp.p.t_fe, "opt_s": wall - tp.p.t_fe,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "x": x.tolist(), "hist": tp.hist,
    }
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{method:11s} n={tp.n:6d} V={row['volume']:.4f} max_g={row['max_g']:+.2e} "
          f"it={row['iterations']} FE={row['n_fe']} rhs={row['n_rhs']} "
          f"wall={wall:.0f}s (FE {row['fe_s']:.0f}s, opt {row['opt_s']:.0f}s) "
          f"mem={row['peak_rss_mb']:.0f}MB | {row['status']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--nelx", type=int, nargs="+", default=[40])
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        from benchmarks.lbracket_stress_report import report
        report()
        return
    if args.single:
        run_one(args.methods[0], args.nelx[0])
        return
    for nelx in args.nelx:
        for method in args.methods:
            cmd = [sys.executable, "-m", "benchmarks.lbracket_stress_solvers", "--single",
                   "--methods", method, "--nelx", str(nelx)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=1.5 * TIME_LIMIT)
            except subprocess.TimeoutExpired:
                row = {"method": method, "nelx": nelx, "n": 3 * nelx * nelx // 4,
                       "status": f"killed after {1.5 * TIME_LIMIT:.0f} s"}
                with open(RESULTS, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{method} nelx={nelx} TIMEOUT", flush=True)
                continue
            line = [l for l in r.stdout.splitlines() if l.startswith(method)]
            print(line[-1] if line else f"{method} n(nelx={nelx}) FAILED rc={r.returncode}\n"
                  + r.stderr[-1500:], flush=True)


if __name__ == "__main__":
    main()
