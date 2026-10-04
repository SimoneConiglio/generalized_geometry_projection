# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Uno on the density-based L-bracket with LOCAL stress constraints (no aggregation).

The "original" stress-constrained topology optimization problem (Duysinx & Bendsoe
1998; Le, Norato, Bruns, Ha & Tortorelli 2010), solved with one constraint per element:

    minimize    V(x) = mean(rho~)                               (mass)
    subject to  g_e(x) = rho~_e^q * svm_e(u) / sigma_lim - 1 <= 0   for every element e
                1e-3 <= x <= 1

with rho~ = H x a linear density filter, SIMP stiffness E = Emin + rho~^p (E0 - Emin)
(p = 3), the qp-relaxed stress interpolation of Le et al. (q = 1/2) and svm_e the
centroid von Mises stress of the *solid* material. The number of constraints equals
the number of design variables and the constraint Jacobian is DENSE (every stress
depends on every density through the displacement field): this benchmark measures how
Uno's algorithms scale with that size.

Pure NumPy/SciPy (structured Q4 mesh, plane stress, cached sparse LU): no FEniCS and
no GEMSEO -- GEMSEO's database would store every dense Jacobian. Uno (``unopy``) is
called directly so that its own time can be separated from the FE + sensitivity time.

    python -m benchmarks.lbracket_density_uno --check-gradient --nelx 20
    python -m benchmarks.lbracket_density_uno --nelx 20 40 60 --presets filtersqp ipopt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

OUT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Plane-stress Q4 element (unit square, unit thickness, E = 1)
# --------------------------------------------------------------------------- #
NU = 0.3


def _q4_matrices(h: float):
    """Element stiffness (2x2 Gauss) and centroid strain-displacement matrix.

    Local node order: (0,0), (h,0), (h,h), (0,h); dofs [u0 v0 u1 v1 ...].
    """
    D = 1.0 / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    xi_n = np.array([-1, 1, 1, -1])
    eta_n = np.array([-1, -1, 1, 1])

    def B_at(xi, eta):
        dN_dxi = xi_n * (1 + eta * eta_n) / 4
        dN_deta = eta_n * (1 + xi * xi_n) / 4
        dN_dx, dN_dy = dN_dxi * 2 / h, dN_deta * 2 / h
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dx
        B[1, 1::2] = dN_dy
        B[2, 0::2] = dN_dy
        B[2, 1::2] = dN_dx
        return B

    g = 1 / math.sqrt(3)
    KE = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            B = B_at(xi, eta)
            KE += B.T @ D @ B * (h / 2) ** 2
    return KE, B_at(0.0, 0.0), D


class LBracketDensity:
    """Density-based L-bracket with local von Mises constraints.

    Domain [0, L]^2 minus the upper-right quadrant (re-entrant corner at (L/2, L/2)),
    top edge of the vertical arm clamped, unit vertical load spread over the upper
    fifth of the right face of the horizontal arm (Le et al. 2010 distribute the load
    to avoid a point-load stress singularity).
    """

    def __init__(self, nelx: int, sigma_lim: float = 30.0, rmin: float = 0.04,
                 p: float = 3.0, q: float = 0.5, Emin: float = 1e-9, L: float = 1.0,
                 xmin: float = 1e-3):
        assert nelx % 2 == 0
        self.nelx = nelx
        h = L / nelx
        self.p, self.q, self.Emin, self.sigma_lim = p, q, Emin, sigma_lim
        # Lower density bound: d(rho^q)/drho = q rho^(q-1) is infinite at rho = 0.
        self.xmin = xmin
        self.KE, self.B0, self.D = _q4_matrices(h)
        V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])
        self.V = V
        self.DB = self.D @ self.B0                     # solid centroid stress / u_e

        # Elements of the L (grid cell (i, j): x in [i h, (i+1) h], y in [j h, ...]).
        half = nelx // 2
        cells = [(i, j) for j in range(nelx) for i in range(nelx)
                 if not (i >= half and j >= half)]
        self.cells = np.array(cells)
        self.n = len(cells)
        node = lambda i, j: j * (nelx + 1) + i          # noqa: E731
        conn = np.array([[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
                         for i, j in cells])
        self.edof = np.stack([2 * conn, 2 * conn + 1], axis=2).reshape(self.n, 8)
        used = np.unique(conn)
        self.ndof = 2 * (nelx + 1) ** 2

        # Clamp the top edge of the vertical arm; drop nodes outside the L.
        top = [node(i, nelx) for i in range(half + 1)]
        unused = np.setdiff1d(np.arange((nelx + 1) ** 2), used)
        fixed = np.concatenate([2 * np.array(top), 2 * np.array(top) + 1,
                                2 * unused, 2 * unused + 1])
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)

        # Unit downward load on the right face, y in [0.4 L, 0.5 L].
        self.f = np.zeros(self.ndof)
        jl = [j for j in range(half + 1) if j * h >= 0.4 * L - 1e-12]
        for j in jl:
            self.f[2 * node(nelx, j) + 1] = -1.0 / len(jl)

        # Linear density filter (cone weights, radius rmin * L).
        centers = (self.cells + 0.5) * h
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        pairs = tree.sparse_distance_matrix(tree, rmin * L, output_type="coo_matrix")
        w = np.maximum(0.0, rmin * L - pairs.data)
        H = sps.coo_matrix((w, (pairs.row, pairs.col)), shape=(self.n, self.n)).tocsr()
        H = H + sps.eye(self.n) * rmin * L * (pairs.nnz == 0)      # keep diagonal
        self.H = sps.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H
        self.HT = self.H.T.tocsr()

        self._x = None
        self.n_fe = 0
        self.t_fe = 0.0                                  # primal solves + sensitivities

    # ----------------------------------------------------------------------- #
    def _solve(self, x):
        if self._x is not None and np.array_equal(x, self._x):
            return
        t0 = time.perf_counter()
        # clip: interior-point / trust-region trial points may sit just outside bounds
        rho = np.clip(self.H @ x, 0.5 * self.xmin, 1.0)
        E = self.Emin + rho**self.p * (1 - self.Emin)
        iK = np.repeat(self.edof, 8, axis=1).ravel()
        jK = np.tile(self.edof, (1, 8)).ravel()
        K = sps.coo_matrix(((self.KE.ravel()[None, :] * E[:, None]).ravel(), (iK, jK)),
                           shape=(self.ndof, self.ndof)).tocsc()
        Kf = K[self.free][:, self.free]
        lu = splu(Kf.tocsc())
        u = np.zeros(self.ndof)
        u[self.free] = lu.solve(self.f[self.free])
        ue = u[self.edof]                                # (n, 8)
        s0 = ue @ self.DB.T                              # solid centroid stress (n, 3)
        vm = np.sqrt(np.maximum(np.einsum("ei,ij,ej->e", s0, self.V, s0), 1e-30))
        self._x, self._rho, self._E, self._lu = x.copy(), rho, E, lu
        self._u, self._ue, self._s0, self._vm = u, ue, s0, vm
        self.g = rho**self.q * vm / self.sigma_lim - 1.0
        self._jac = None
        self.n_fe += 1
        self.t_fe += time.perf_counter() - t0

    def volume(self, x):
        return float(np.mean(self.H @ x))

    def volume_grad(self):
        return self.HT @ np.full(self.n, 1.0 / self.n)

    def constraints(self, x):
        self._solve(x)
        return self.g

    def jacobian(self, x, chunk: int = 1024):
        """Dense (n_constraints, n_variables) Jacobian by the adjoint method."""
        self._solve(x)
        if self._jac is not None:
            return self._jac
        t0 = time.perf_counter()
        n, q = self.n, self.q
        rho, vm, s0 = self._rho, self._vm, self._s0
        # d vm_e / d u_e = DB^T V s0 / vm  ->  ds_e/du_e = rho^q / sigma_lim * that
        dsdu = (s0 @ self.V) @ self.DB / vm[:, None]                # (n, 8)
        dsdu *= (rho**q / self.sigma_lim)[:, None]
        dE = self.p * rho ** (self.p - 1) * (1 - self.Emin)
        KEu = self._ue @ self.KE.T                                   # (n, 8)
        free_pos = -np.ones(self.ndof, dtype=np.int64)
        free_pos[self.free] = np.arange(self.free.size)
        J = np.zeros((n, n))
        for start in range(0, n, chunk):
            idx = np.arange(start, min(start + chunk, n))
            rhs = np.zeros((self.ndof, idx.size))
            np.add.at(rhs, (self.edof[idx], np.broadcast_to(np.arange(idx.size)[:, None],
                                                           (idx.size, 8))), dsdu[idx])
            lam = np.zeros((self.ndof, idx.size))
            lam[self.free] = self._lu.solve(rhs[self.free])
            # implicit part: -dE_k lam[edof_k]^T KE u_k   for every element k
            J[idx] = -(np.einsum("kic,ki->ck", lam[self.edof], KEu) * dE[None, :])
        J[np.arange(n), np.arange(n)] += q * rho ** (q - 1) * vm / self.sigma_lim
        self._jac = J @ self.H                                        # chain rule filter
        self.t_fe += time.perf_counter() - t0
        return self._jac


# --------------------------------------------------------------------------- #
def check_gradient(nelx=12, seed=0):
    prob = LBracketDensity(nelx)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 1.0, prob.n)
    J = prob.jacobian(x).copy()
    g0 = prob.constraints(x).copy()
    err = 0.0
    for k in rng.choice(prob.n, 6, replace=False):
        e = 1e-6
        xp = x.copy(); xp[k] += e
        xm = x.copy(); xm[k] -= e
        fd = (prob.constraints(xp) - prob.constraints(xm)) / (2 * e)
        err = max(err, np.max(np.abs(fd - J[:, k])) / max(1e-12, np.max(np.abs(fd))))
    print(f"n={prob.n} m={g0.size}  max relative column error (FD vs adjoint) = {err:.2e}")
    return err


def run_uno(prob: LBracketDensity, preset: str, max_iter: int, options=None,
            hessian_model=None, x0=None, time_limit=None, logger="SILENT",
            objective_scale=1.0):
    import unopy

    n = prob.n
    x0 = np.ones(n) if x0 is None else x0
    prob.n_fe, prob.t_fe = 0, 0.0
    # objective_scale = n minimizes the SUM of densities: an O(1/n) volume gradient
    # makes the identity-initialised L-BFGS SQP step O(1/n) per iteration.
    vgrad = objective_scale * prob.volume_grad()
    hist = []
    t_cb = [0.0]

    def timed(func):
        def wrapper(*args):
            t0 = time.perf_counter()
            try:
                return func(*args)
            finally:
                t_cb[0] += time.perf_counter() - t0
        return wrapper

    def objective(x):
        return objective_scale * prob.volume(np.array(x))

    def objective_gradient(x, out):
        out[:] = vgrad

    def constraints(x, out):
        x = np.array(x)
        out[:] = prob.constraints(x)
        hist.append((time.perf_counter(), prob.volume(x), float(np.max(prob.g))))

    def jacobian(x, out):
        out[:] = prob.jacobian(np.array(x)).ravel()

    model = unopy.Model(unopy.PROBLEM_NONLINEAR, n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds([prob.xmin] * n)
    model.set_variables_upper_bounds([1.0] * n)
    model.set_objective(unopy.MINIMIZE, timed(objective), timed(objective_gradient))
    rows = np.repeat(np.arange(n, dtype=np.int32), n)     # dense pattern, row-major
    cols = np.tile(np.arange(n, dtype=np.int32), n)
    model.set_constraints(n, timed(constraints), [-math.inf] * n, [0.0] * n,
                          n * n, rows, cols, timed(jacobian))
    model.set_initial_primal_iterate(x0.tolist())

    solver = unopy.UnoSolver()
    solver.set_preset(preset)
    solver.set_option("hessian_model", hessian_model or
                      ("zero" if preset == "filterslp" else "LBFGS"))
    solver.set_option("max_iterations", max_iter)
    solver.set_option("logger", logger)
    if time_limit:
        solver.set_option("time_limit", float(time_limit))
    for k, v in (options or {}).items():
        solver.set_option(k, v)
    t0 = time.perf_counter()
    res = solver.optimize(model)
    wall = time.perf_counter() - t0
    x = np.clip(np.array(res.primal_solution)[:n], prob.xmin, 1)
    g = prob.constraints(x)
    return {
        "preset": preset, "options": options or {}, "n": n,
        "objective_scale": objective_scale,
        "status": res.optimization_status.name, "solution": res.solution_status.name,
        "iterations": int(res.number_iterations),
        "volume": prob.volume(x), "max_g": float(np.max(g)),
        "n_constraint_evals": int(res.number_constraint_evaluations),
        "n_jacobian_evals": int(res.number_jacobian_evaluations),
        "wall_s": wall, "callbacks_s": t_cb[0], "uno_s": wall - t_cb[0],
        "fe_s": prob.t_fe,
        "x": x.tolist(),
        "hist": [(t - t0, v, gm) for t, v, gm in hist],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nelx", type=int, nargs="+", default=[20])
    ap.add_argument("--presets", nargs="+", default=["filtersqp"])
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--sigma-lim", type=float, default=30.0)
    ap.add_argument("--options", type=json.loads, default=None,
                    help='Extra Uno options as JSON, e.g. \'{"TR_radius": 0.05}\'.')
    ap.add_argument("--objective-scale", default="1",
                    help="Objective factor: a number, or 'n' (minimize the sum of densities).")
    ap.add_argument("--tag", default="")
    ap.add_argument("--logger", default="SILENT")
    ap.add_argument("--check-gradient", action="store_true")
    args = ap.parse_args()
    if args.check_gradient:
        for nelx in args.nelx:
            check_gradient(nelx)
        return
    rows = []
    out = OUT / f"lbracket_density_uno{args.tag}.json"
    for nelx in args.nelx:
        prob = LBracketDensity(nelx, sigma_lim=args.sigma_lim)
        for preset in args.presets:
            print(f"=== nelx={nelx} n=m={prob.n} preset={preset} options={args.options}",
                  flush=True)
            f_scale = prob.n if args.objective_scale == "n" else float(args.objective_scale)
            r = run_uno(prob, preset, args.max_iter, args.options,
                        time_limit=args.time_limit, logger=args.logger,
                        objective_scale=f_scale)
            r["nelx"] = nelx
            print(f"    {r['status']}/{r['solution']} it={r['iterations']} "
                  f"V={r['volume']:.4f} max_g={r['max_g']:+.3e} "
                  f"wall={r['wall_s']:.1f}s uno={r['uno_s']:.1f}s fe={r['fe_s']:.1f}s "
                  f"jac_evals={r['n_jacobian_evals']}", flush=True)
            rows.append(r)
            json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    main()
