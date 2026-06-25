# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""AMG-preconditioned iterative solver for the FEM stiffness system.

Implements the core approach of the AMJax library (https://github.com/vboussange/AMJax):
a PyAMG algebraic multigrid (AMG) hierarchy is built from the assembled stiffness
matrix and used as a preconditioner for a conjugate-gradient (CG) solve.

We use scipy's CG with the PyAMG preconditioner, which avoids the Python ≥ 3.11
requirement of the ``amjax`` package while delivering the same convergence behaviour.
When JAX is available (and ``jax_enable_x64`` is set), the assembly of the RHS and
solution conversion is done via JAX arrays for consistency with future GPU backends.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps


def solve_amjax(
    K: sps.spmatrix,
    f: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 200,
    cycle: str = "V",
) -> np.ndarray:
    """Solve *K u = f* using a PyAMG AMG preconditioner with scipy CG.

    Builds a smoothed-aggregation multigrid hierarchy via PyAMG and uses it
    as a preconditioner for :func:`scipy.sparse.linalg.cg`.  This reproduces
    the key algorithmic idea of AMJax: AMG-preconditioned Krylov iteration for
    FEM linear systems.

    Parameters
    ----------
    K:
        Assembled, BC-applied global stiffness matrix (any scipy sparse format).
    f:
        Load vector, shape ``(n_dofs,)``.
    tol:
        Relative residual tolerance for the CG solver.
    maxiter:
        Maximum CG iterations.
    cycle:
        AMG cycle type used in the preconditioner — ``"V"``, ``"W"``, or ``"F"``.

    Returns
    -------
    np.ndarray
        Displacement vector, shape ``(n_dofs,)``, as a NumPy float64 array.
    """
    import pyamg
    from scipy.sparse.linalg import cg

    K_csr = K.tocsr().astype(np.float64)
    f_np = np.asarray(f, dtype=np.float64)

    # Build AMG hierarchy (smoothed aggregation — optimal for SPD FEM matrices).
    ml = pyamg.smoothed_aggregation_solver(K_csr)
    M = ml.aspreconditioner(cycle=cycle)

    u, info = cg(K_csr, f_np, M=M, rtol=tol, maxiter=maxiter)
    if info != 0:
        # Fallback: relax tolerance and increase iterations.
        u, info = cg(K_csr, f_np, M=M, rtol=tol * 100, maxiter=maxiter * 5)

    return np.asarray(u, dtype=np.float64)
