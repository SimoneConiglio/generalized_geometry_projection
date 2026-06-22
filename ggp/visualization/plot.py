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
    iso_level: float = 0.3,
) -> None:
    """Render a 3D isosurface of the density field via marching cubes.

    Produces a perspective-correct solid-body view matching the kind of output
    ParaView shows for topology-optimized structures (light-shaded isosurface at
    rho = iso_level, viewed from a south-east elevated angle).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    X = eval_coords[:, 0]
    Y = eval_coords[:, 1]
    Z = eval_coords[:, 2]

    x_unique = np.sort(np.unique(np.round(X, 6)))
    y_unique = np.sort(np.unique(np.round(Y, 6)))
    z_unique = np.sort(np.unique(np.round(Z, 6)))
    nelx, nely, nelz = len(x_unique), len(y_unique), len(z_unique)

    xi = np.searchsorted(x_unique, np.round(X, 6))
    yi = np.searchsorted(y_unique, np.round(Y, 6))
    zi = np.searchsorted(z_unique, np.round(Z, 6))

    # Build (nelx, nely, nelz) density volume
    vol = np.zeros((nelx, nely, nelz))
    vol[xi, yi, zi] = rho_E

    # Pad with a one-voxel void shell so marching cubes closes the boundary
    vol_pad = np.pad(vol, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = marching_cubes(vol_pad, level=iso_level)

    # Convert voxel indices back to physical coordinates (undo the padding offset)
    dx = (x_unique[-1] - x_unique[0]) / max(nelx - 1, 1)
    dy = (y_unique[-1] - y_unique[0]) / max(nely - 1, 1)
    dz = (z_unique[-1] - z_unique[0]) / max(nelz - 1, 1)
    verts_phys = np.column_stack([
        x_unique[0] + (verts[:, 0] - 1) * dx,
        y_unique[0] + (verts[:, 1] - 1) * dy,
        z_unique[0] + (verts[:, 2] - 1) * dz,
    ])

    mesh = Poly3DCollection(verts_phys[faces], alpha=0.95, linewidth=0)
    mesh.set_facecolor("#4d7db5")
    mesh.set_edgecolor("none")

    fig = plt.figure(figsize=(10, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(mesh)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    # South-east elevated view angle (matches typical ParaView preset)
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
