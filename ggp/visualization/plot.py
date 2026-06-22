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
    # Sharpen the density field: sigmoid with beta=10 centred at 0.5
    # This maps the smooth MNA transition band to a near-binary 0/1 image
    # while preserving the correct material distribution topology.
    beta = 10.0
    Z_sharp = 1.0 / (1.0 + np.exp(-beta * (Z - 0.5)))

    ax.imshow(
        1.0 - Z_sharp,
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
    """Save a maximum-density projection (along Z) of a 3D density field.

    Projects the 3D field onto the XY-plane by taking the maximum density
    across all Z-layers at each (x, y) position, giving a full view of the
    structure regardless of how sparse it is.
    """
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

    # Maximum projection along Z
    Z2D = np.zeros((nely, nelx))
    np.maximum.at(Z2D, (yi, xi), rho_E)

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
    ax.set_title(f"{title} (max-Z projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
