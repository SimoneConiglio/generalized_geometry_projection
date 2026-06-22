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
    iso_level: float = 0.5,
    fixed_face: str = "left",
    load_point: tuple | None = None,
) -> None:
    """Render a 4-panel 3D isosurface figure (ρ > iso_level) with BC/load markers.

    Panels: isometric view, XY front (z=mid), XZ top (y=mid), YZ side (x=max).
    Fixed face is highlighted in blue; load location shown as a red arrow.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    X = eval_coords[:, 0]
    Y = eval_coords[:, 1]
    Z = eval_coords[:, 2]
    Lx, Ly, Lz = X.max(), Y.max(), Z.max()

    x_unique = np.sort(np.unique(np.round(X, 6)))
    y_unique = np.sort(np.unique(np.round(Y, 6)))
    z_unique = np.sort(np.unique(np.round(Z, 6)))
    nelx, nely, nelz = len(x_unique), len(y_unique), len(z_unique)

    xi = np.searchsorted(x_unique, np.round(X, 6))
    yi = np.searchsorted(y_unique, np.round(Y, 6))
    zi = np.searchsorted(z_unique, np.round(Z, 6))

    vol = np.zeros((nelx, nely, nelz))
    vol[xi, yi, zi] = rho_E

    vol_pad = np.pad(vol, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = marching_cubes(vol_pad, level=iso_level)

    dx = (x_unique[-1] - x_unique[0]) / max(nelx - 1, 1)
    dy = (y_unique[-1] - y_unique[0]) / max(nely - 1, 1)
    dz = (z_unique[-1] - z_unique[0]) / max(nelz - 1, 1)
    vp = np.column_stack([
        x_unique[0] + (verts[:, 0] - 1) * dx,
        y_unique[0] + (verts[:, 1] - 1) * dy,
        z_unique[0] + (verts[:, 2] - 1) * dz,
    ])

    # Default load at mid-right face centre
    if load_point is None:
        load_point = (Lx, Ly / 2, Lz / 2)
    lx, ly, lz = load_point

    def _add_isosurface(ax):
        mesh = Poly3DCollection(vp[faces], alpha=0.90, linewidth=0)
        mesh.set_facecolor("#4d7db5")
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)

    def _add_fixed_face(ax):
        """Blue translucent rectangle for the fixed (left) face."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as P3D
        if fixed_face == "left":
            corners = np.array([[0, 0, 0], [0, Ly, 0], [0, Ly, Lz], [0, 0, Lz]])
        face = P3D([corners], alpha=0.25, linewidth=0)
        face.set_facecolor("#2ca02c")
        ax.add_collection3d(face)

    def _add_load_arrow(ax):
        ax.quiver(lx, ly, lz, 0, 0, -Lz * 0.15,
                  color="red", linewidth=2, arrow_length_ratio=0.4)
        ax.scatter([lx], [ly], [lz], color="red", s=40, zorder=5)

    def _set_limits(ax):
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(title, fontsize=13, y=0.98)

    # Panel 1 — isometric view
    ax1 = fig.add_subplot(221, projection="3d")
    _add_isosurface(ax1); _add_fixed_face(ax1); _add_load_arrow(ax1)
    _set_limits(ax1)
    ax1.view_init(elev=25, azim=-55)
    ax1.set_title("Isometric", fontsize=10)

    # Panel 2 — front view (looking along Z)
    ax2 = fig.add_subplot(222, projection="3d")
    _add_isosurface(ax2); _add_fixed_face(ax2); _add_load_arrow(ax2)
    _set_limits(ax2)
    ax2.view_init(elev=0, azim=-90)
    ax2.set_title("Front (XY plane)", fontsize=10)

    # Panel 3 — top view (looking down Y)
    ax3 = fig.add_subplot(223, projection="3d")
    _add_isosurface(ax3); _add_fixed_face(ax3); _add_load_arrow(ax3)
    _set_limits(ax3)
    ax3.view_init(elev=90, azim=-90)
    ax3.set_title("Top (XZ plane)", fontsize=10)

    # Panel 4 — side view (looking along X from right)
    ax4 = fig.add_subplot(224, projection="3d")
    _add_isosurface(ax4); _add_fixed_face(ax4); _add_load_arrow(ax4)
    _set_limits(ax4)
    ax4.view_init(elev=0, azim=0)
    ax4.set_title("Side (YZ plane)", fontsize=10)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4d7db5", alpha=0.9, label=f"Material (ρ > {iso_level})"),
        Patch(facecolor="#2ca02c", alpha=0.4, label="Fixed face (BC)"),
        Patch(facecolor="red", label="Point load"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
