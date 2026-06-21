# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Density field visualization utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_density_plot_2d(
    rho_E: np.ndarray,
    eval_coords: np.ndarray,
    output_path: str | Path,
    title: str = "Optimized Design",
) -> None:
    """Save a 2D density field as a grayscale topology image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = eval_coords[:, 0]
    Y = eval_coords[:, 1]

    x_unique = np.sort(np.unique(np.round(X, 6)))
    y_unique = np.sort(np.unique(np.round(Y, 6)))
    nelx, nely = len(x_unique), len(y_unique)

    xi = np.searchsorted(x_unique, np.round(X, 6))
    yi = np.searchsorted(y_unique, np.round(Y, 6))
    Z = np.zeros((nely, nelx))
    Z[yi, xi] = rho_E

    aspect = (Y.max() - Y.min()) / max(X.max() - X.min(), 1e-9)
    fig_w = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect + 0.6))
    ax.imshow(
        1.0 - Z,
        cmap="gray",
        origin="lower",
        aspect="equal",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def save_density_plot_3d(
    rho_E: np.ndarray,
    eval_coords: np.ndarray,
    output_path: str | Path,
    title: str = "3D Optimized Design",
) -> None:
    """Save a mid-plane Z-slice of a 3D density field as a grayscale image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z_coords = eval_coords[:, 2]
    z_unique = np.sort(np.unique(np.round(Z_coords, 6)))
    z_mid = z_unique[len(z_unique) // 2]
    mask = np.abs(np.round(Z_coords, 6) - z_mid) < 1e-6

    coords_slice = eval_coords[mask, :2]
    rho_slice = rho_E[mask]

    X = coords_slice[:, 0]
    Y = coords_slice[:, 1]
    x_unique = np.sort(np.unique(np.round(X, 6)))
    y_unique = np.sort(np.unique(np.round(Y, 6)))
    nelx, nely = len(x_unique), len(y_unique)
    xi = np.searchsorted(x_unique, np.round(X, 6))
    yi = np.searchsorted(y_unique, np.round(Y, 6))
    Z2D = np.zeros((nely, nelx))
    Z2D[yi, xi] = rho_slice

    aspect = (Y.max() - Y.min()) / max(X.max() - X.min(), 1e-9)
    fig_w = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect + 0.6))
    ax.imshow(
        1.0 - Z2D,
        cmap="gray",
        origin="lower",
        aspect="equal",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"{title} (Z = {z_mid:.1f} slice)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
