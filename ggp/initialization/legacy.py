# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Legacy direct initial-guess builders (grid cross-bars + ALM staircase).

These produce a normalized design vector directly from ``num_vars`` (rather than
a :class:`~ggp.initialization.skeleton.Skeleton`), so they live outside the mesh
pattern registry.  The code is moved **verbatim** from the previous
``GGPPipeline._make_init`` so the ``grid`` pattern and the ALM init stay
bit-for-bit identical to before the refactor.
"""
from __future__ import annotations

import numpy as np


def make_grid_init_2d(n: int, **kwargs) -> np.ndarray:
    """2-D Free grid init: paired crossed bars on a 3×3 grid (GGP_main.m)."""
    Lx = kwargs.get("Lx", 60.0)
    Ly = kwargs.get("Ly", 30.0)
    lb = kwargs.get("lb", None)
    ub = kwargs.get("ub", None)
    nc = n // 6
    ncx, ncy = 1, 1
    # Standard Matlab-style grid (same for rectangular and L-shape domains).
    # For the L-shape, only 1 of 9 positions is in the non-design region —
    # the other 8 (including corners like (0,Ly) at ±theta) provide structural
    # connectivity across both arms.  The empty-element override in physics
    # handles the non-design region correctly.
    xp = np.linspace(0.0, Lx, ncx + 2)
    yp = np.linspace(0.0, Ly, ncy + 2)
    xx, yy = np.meshgrid(xp, yp)
    grid_X = xx.flatten()
    grid_Y = yy.flatten()
    half = nc // 2
    theta = np.arctan2(Ly / ncy, Lx / ncx)
    Lc = 2.0 * np.sqrt((Lx / (ncx + 2)) ** 2 + (Ly / (ncy + 2)) ** 2)

    # Paired bars: first nc//2 at +theta, remaining at -theta
    n_grid = len(grid_X)
    idx_pos = np.arange(half) % n_grid
    idx_neg = np.arange(nc - half) % n_grid
    Xc = np.concatenate([grid_X[idx_pos], grid_X[idx_neg]])
    Yc = np.concatenate([grid_Y[idx_pos], grid_Y[idx_neg]])
    Tc = np.concatenate([theta * np.ones(half), -theta * np.ones(nc - half)])

    hc = 2.0   # initial h just above minh=1
    Mc = 0.5   # initial_d

    # Normalize to [0,1] using mapper bounds
    x = np.empty(n)
    if lb is not None and ub is not None:
        x[0::6] = np.clip((Xc - lb[0::6]) / (ub[0::6] - lb[0::6]), 0.0, 1.0)
        x[1::6] = np.clip((Yc - lb[1::6]) / (ub[1::6] - lb[1::6]), 0.0, 1.0)
        x[2::6] = np.clip((Lc - lb[2::6]) / (ub[2::6] - lb[2::6]), 0.0, 1.0)
        x[3::6] = np.clip((hc - lb[3::6]) / (ub[3::6] - lb[3::6]), 0.0, 1.0)
        x[4::6] = np.clip((Tc - lb[4::6]) / (ub[4::6] - lb[4::6]), 0.0, 1.0)
    else:
        # Fallback: rough normalized values
        x[0::6] = Xc / (Lx + 2)
        x[1::6] = Yc / (Ly + 2)
        x[2::6] = Lc / np.sqrt(Lx**2 + Ly**2)
        x[3::6] = 0.015
        x[4::6] = 0.5
    x[5::6] = Mc
    return x


def make_alm_init(n: int, **kwargs) -> np.ndarray:
    """ALM staircase init (interleaved [Xc,L] + h + Mc + [y0, theta0])."""
    np_val     = kwargs.get("np_val", 1)
    nY         = kwargs.get("nY", 1)
    layer_h    = kwargs.get("layer_height", 3.0)
    alpha_deg  = kwargs.get("alpha_deg", 45.0)
    n_xl       = 2 * nY * np_val
    x = np.empty(n)

    # Staircase initialization: each column is a maximum-overhang ascending
    # staircase so that the rightmost column reaches x=Lx at mid-height
    # (the load layer), giving non-zero gradient from iteration 1.
    # Physical Xc bounds: lb=-1, ub=Lx+1 (range = Lx+2).
    Lx         = kwargs.get("Lx", 60.0)
    xc_range   = Lx + 2.0                          # ub - lb = (Lx+1) - (-1)
    delta_norm = np.tan(np.deg2rad(alpha_deg)) * layer_h / xc_range
    load_layer  = nY // 2
    P_bot_right = (Lx + 1.0) / xc_range - load_layer * delta_norm
    P_bot_left  = (0.0 + 1.0) / xc_range
    P_bottom = np.linspace(P_bot_left, P_bot_right, np_val)

    # Assign Xc[k, j] = P_bottom[j] + k * delta_norm (clamped to [0,1])
    # F-order: x_vars index of Xc[k,j] = 2*(j*nY + k)
    for j in range(np_val):
        for k in range(nY):
            x[2*(j*nY + k)]     = float(np.clip(P_bottom[j] + k*delta_norm, 0.0, 1.0))
            x[2*(j*nY + k) + 1] = 0.333   # L normalized ≈ 6 physical

    x[n_xl       : n_xl + np_val] = 1.0   # h = 1 (full height)
    x[n_xl + np_val : n_xl + 2*np_val] = 0.50  # Mc = 0.5
    if n >= n_xl + 2*np_val + 2:
        x[n_xl + 2*np_val]     = 0.5    # y0 = 0
        # theta0 = 0 (normalised 0.5 of the [-pi/2, pi/2] range). theta0=0 is
        # the optimal build orientation for the cantilever: jointly-optimized
        # compliance was tested at theta0 = 0 / -56 / -90 deg -> C = 86 / 108 /
        # 131, i.e. it worsens monotonically as the plane rotates away from 0
        # (vertical layers let the bars form the efficient horizontal load path;
        # beam-axis layering at +/-90 is less efficient).
        x[n_xl + 2*np_val + 1] = 0.5    # theta0 = 0
    return x


def make_grid_init_3d(n: int, **kwargs) -> np.ndarray:
    """3-D Free grid init: cross-bar pairs swept along the box diagonal.

    NB: this is the legacy (degenerate) 3-D seed — kept only for reproducibility
    via ``pattern: grid``.  The default 3-D init is the ``tet3d`` mesh pattern.
    """
    Lx = kwargs.get("Lx", 60.0)
    Ly = kwargs.get("Ly", 30.0)
    Lz = kwargs.get("Lz", 30.0)
    lb = kwargs.get("lb", None)
    ub = kwargs.get("ub", None)
    nc = n // 8

    # Build 3-D grid: sample independently along each axis so the first
    # nc//2 positions are spread across x, y, z simultaneously (diagonal sweep).
    half = nc // 2
    grid_X = np.linspace(0.0, Lx, half + 1)[:-1]
    grid_Y = np.linspace(0.0, Ly, half + 1)[:-1]
    grid_Z = np.linspace(0.0, Lz, half + 1)[:-1]

    # Diagonal bar length spanning ~1/half of each axis
    Lc = 2.0 * np.sqrt((Lx / half) ** 2 + (Ly / half) ** 2 + (Lz / half) ** 2)
    theta = np.arctan2(Ly / half, Lx / half)
    phi = np.arctan2(Lz / half, np.sqrt((Lx / half) ** 2 + (Ly / half) ** 2))

    Xc = np.concatenate([grid_X, grid_X[: nc - half]])
    Yc = np.concatenate([grid_Y, grid_Y[: nc - half]])
    Zc = np.concatenate([grid_Z, grid_Z[: nc - half]])
    Tc = np.concatenate([theta * np.ones(half), -theta * np.ones(nc - half)])
    Pc = np.concatenate([phi * np.ones(half), -phi * np.ones(nc - half)])

    hc = 2.0
    Mc = 0.5

    x = np.empty(n)
    if lb is not None and ub is not None:
        x[0::8] = np.clip((Xc - lb[0::8]) / (ub[0::8] - lb[0::8]), 0.0, 1.0)
        x[1::8] = np.clip((Yc - lb[1::8]) / (ub[1::8] - lb[1::8]), 0.0, 1.0)
        x[2::8] = np.clip((Zc - lb[2::8]) / (ub[2::8] - lb[2::8]), 0.0, 1.0)
        x[3::8] = np.clip((Lc - lb[3::8]) / (ub[3::8] - lb[3::8]), 0.0, 1.0)
        x[4::8] = np.clip((hc - lb[4::8]) / (ub[4::8] - lb[4::8]), 0.0, 1.0)
        x[5::8] = np.clip((Tc - lb[5::8]) / (ub[5::8] - lb[5::8]), 0.0, 1.0)
        x[6::8] = np.clip((Pc - lb[6::8]) / (ub[6::8] - lb[6::8]), 0.0, 1.0)
    else:
        x[0::8] = Xc / Lx
        x[1::8] = Yc / Ly
        x[2::8] = Zc / Lz
        x[3::8] = Lc / np.sqrt(Lx**2 + Ly**2 + Lz**2)
        x[4::8] = 0.02
        x[5::8] = 0.5
        x[6::8] = 0.5
    x[7::8] = Mc
    return x
