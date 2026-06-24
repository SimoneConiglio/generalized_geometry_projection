# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Encode a :class:`Skeleton` into a normalized GGP design vector.

Each edge becomes one component block, using the Free mapper variable layouts:

* 2-D (``Free2DMapper``, 6 vars): ``[Xc, Yc, L, h, Theta, Mc]``
* 3-D (``Free3DMapper``, 8 vars): ``[Xc, Yc, Zc, L, h, Theta, Phi, Mc]``

The 3-D axis convention (``n = (cosP cosT, cosP sinT, sinP)``) is taken from
``ggp/utils/vectorized_mapping_3d.py``; an edge direction ``d`` therefore maps to
``Theta = atan2(d_y, d_x)`` and ``Phi = asin(d_z)``.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .skeleton import Skeleton


def encode_skeleton(
    skeleton: Skeleton,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    thickness: float,
    membership: float,
) -> np.ndarray:
    """Return a normalized ``[0, 1]`` design vector seeding one bar per edge.

    Parameters
    ----------
    skeleton : Skeleton
        Edges to encode; ``num_edges`` must equal ``len(lb) // vars_per_comp``.
    lb, ub : array
        Per-variable physical bounds from the mapper (full design-vector length).
    thickness : float
        Initial bar thickness ``h`` (physical) for every component.
    membership : float
        Initial membership / density ``Mc`` in ``[0, 1]`` for every component.
    """
    dim = skeleton.dim
    vpc = 6 if dim == 2 else 8
    E = skeleton.num_edges
    if E * vpc != len(lb):
        raise ValueError(
            f"Skeleton has {E} edges ({E * vpc} vars) but bounds have {len(lb)}; "
            "num_components must equal the skeleton edge count."
        )

    mids = skeleton.midpoints()
    L = skeleton.lengths()
    d = skeleton.directions()

    phys = np.zeros((E, vpc))
    if dim == 2:
        phys[:, 0] = mids[:, 0]
        phys[:, 1] = mids[:, 1]
        phys[:, 2] = L
        phys[:, 3] = thickness
        phys[:, 4] = np.arctan2(d[:, 1], d[:, 0])
        phys[:, 5] = membership
    else:
        phys[:, 0] = mids[:, 0]
        phys[:, 1] = mids[:, 1]
        phys[:, 2] = mids[:, 2]
        phys[:, 3] = L
        phys[:, 4] = thickness
        phys[:, 5] = np.arctan2(d[:, 1], d[:, 0])           # Theta (azimuth)
        phys[:, 6] = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))  # Phi (elevation)
        phys[:, 7] = membership

    x = (phys.flatten() - lb) / (ub - lb)
    return np.clip(x, 0.0, 1.0)


def fit_thickness_to_volfrac(
    skeleton: Skeleton,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    membership: float,
    evaluate: Callable[[np.ndarray], np.ndarray],
    target_volfrac: float,
    h_bounds: Tuple[float, float],
    n_iter: int = 20,
) -> float:
    """Bisect the bar thickness so the projected volume fraction ≈ target.

    The projected density increases monotonically with thickness, so a simple
    bisection on ``h`` drives ``mean(evaluate(x)) → target_volfrac``.  This is
    what keeps the mesh-seeded start near-feasible instead of near-empty.

    Parameters
    ----------
    evaluate : callable
        Maps a design vector to the aggregated element density field ``rho_E``
        (the geometry mapper forward; no FE solve).
    h_bounds : (lo, hi)
        Physical thickness search interval (clamped to the mapper's ``h`` range).
    """
    lo, hi = h_bounds
    best = 0.5 * (lo + hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        best = mid
        x = encode_skeleton(skeleton, lb, ub, thickness=mid, membership=membership)
        vf = float(np.mean(evaluate(x)))
        if vf < target_volfrac:
            lo = mid
        else:
            hi = mid
    return best
