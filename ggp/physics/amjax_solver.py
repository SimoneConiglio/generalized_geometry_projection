# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""AMJax-based linear solver for the FEM stiffness system.

Bridges scipy sparse (assembled by GGPPhysicsDiscipline) with AMJax, which
wraps a PyAMG algebraic multigrid hierarchy in JAX so solves are JIT-compiled,
vmappable, and GPU-compatible.

Reference: https://github.com/vboussange/AMJax
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
    """Solve *K u = f* using AMJax as an AMG preconditioner for JAX CG.

    The AMJax :meth:`MultilevelSolver.aspreconditioner` is used with
    :func:`jax.scipy.sparse.linalg.cg` — this is the usage pattern that
    achieves the highest accuracy and the 62× speedup reported in the
    AMJax benchmarks.

    Parameters
    ----------
    K:
        Assembled, BC-applied global stiffness matrix (any scipy sparse format).
    f:
        Load vector, shape ``(n_dofs,)``.
    tol:
        Relative residual tolerance for the JAX CG solver.
    maxiter:
        Maximum CG iterations.
    cycle:
        AMG cycle type used in the preconditioner — ``"V"``, ``"W"``, or ``"F"``.

    Returns
    -------
    np.ndarray
        Displacement vector, shape ``(n_dofs,)``, as a NumPy float64 array.
    """
    import jax
    import jax.numpy as jnp
    import jax.scipy.sparse.linalg as jssl
    from jax.experimental import sparse as jsparse
    import pyamg
    from amjax import MultilevelSolver

    # AMJax / JAX default to float32; enable 64-bit for FEM accuracy.
    jax.config.update("jax_enable_x64", True)

    K_csr = K.tocsr().astype(np.float64)

    # Build algebraic multigrid hierarchy via PyAMG.
    # smoothed_aggregation_solver is the recommended factory for SPD systems
    # (elasticity stiffness matrices are symmetric positive semi-definite after
    # BC application makes them positive definite).
    ml_pyamg = pyamg.smoothed_aggregation_solver(K_csr)

    # Convert hierarchy to JAX-compatible preconditioner.
    ml_jax = MultilevelSolver.from_pyamg(ml_pyamg)
    M = ml_jax.aspreconditioner(cycle=cycle)

    # Assemble JAX sparse matrix and RHS.
    A_jax = jsparse.BCOO.from_scipy_sparse(K_csr)
    b = jnp.array(f, dtype=jnp.float64)

    # Preconditioned CG — JIT-compiled on the first call.
    u, _info = jssl.cg(A_jax, b, M=M, tol=tol, maxiter=maxiter)

    return np.asarray(u, dtype=np.float64)
