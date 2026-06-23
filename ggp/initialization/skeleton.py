# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Geometry-agnostic skeleton produced by initial-guess patterns.

A :class:`Skeleton` is just a set of straight edges (line segments) in physical
coordinates.  Each edge later becomes one GGP component (bar): its midpoint is
the component centre, its length the bar length, and its direction the bar
orientation.  Patterns build a skeleton from a coarse tessellation of the design
domain; the encoder turns it into a normalized design vector.

Everything here is pure NumPy — no FEniCS, no formulation knowledge — mirroring
the discipline of the projection mappers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DomainBox:
    """Axis-aligned design domain with optional non-design (void) sub-boxes.

    Parameters
    ----------
    Lx, Ly : float
        Domain extents along x and y.
    Lz : float, optional
        Domain extent along z.  ``None`` marks a 2-D domain.
    non_design : list of (origin, extent)
        Axis-aligned void regions; a point inside any of them is excluded from
        the design space (e.g. the L-shape bracket's empty quadrant).
    """

    Lx: float
    Ly: float
    Lz: Optional[float] = None
    non_design: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return 3 if self.Lz is not None else 2

    @property
    def extents(self) -> Tuple[float, ...]:
        return (self.Lx, self.Ly, self.Lz) if self.dim == 3 else (self.Lx, self.Ly)

    def in_non_design(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: ``True`` where a point lies in any non-design region.

        Parameters
        ----------
        points : (M, dim) array
        """
        points = np.atleast_2d(points)
        mask = np.zeros(len(points), dtype=bool)
        for origin, extent in self.non_design:
            origin = np.asarray(origin, dtype=float)
            extent = np.asarray(extent, dtype=float)
            d = points.shape[1]
            inside = np.all(
                (points >= origin[:d] - 1e-9) & (points <= extent[:d] + 1e-9), axis=1
            )
            mask |= inside
        return mask


@dataclass
class Skeleton:
    """A set of edges (line segments) in physical space.

    Parameters
    ----------
    nodes : (N, dim) array
        Vertex coordinates.
    edges : (E, 2) int array
        Index pairs into ``nodes``.
    """

    nodes: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.edges = np.asarray(self.edges, dtype=int).reshape(-1, 2)

    @property
    def dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def vectors(self) -> np.ndarray:
        """Edge vectors ``P2 - P1`` of shape (E, dim)."""
        return self.nodes[self.edges[:, 1]] - self.nodes[self.edges[:, 0]]

    def midpoints(self) -> np.ndarray:
        """Edge midpoints of shape (E, dim)."""
        return 0.5 * (self.nodes[self.edges[:, 0]] + self.nodes[self.edges[:, 1]])

    def lengths(self) -> np.ndarray:
        """Edge lengths of shape (E,)."""
        return np.linalg.norm(self.vectors(), axis=1)

    def directions(self) -> np.ndarray:
        """Unit edge directions of shape (E, dim)."""
        v = self.vectors()
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def dedup_edges(self) -> "Skeleton":
        """Return a copy with duplicate / reversed edges collapsed."""
        canon = np.sort(self.edges, axis=1)
        uniq = np.unique(canon, axis=0)
        return Skeleton(self.nodes, uniq)

    def drop_zero_length(self, tol: float = 1e-9) -> "Skeleton":
        """Return a copy without degenerate (near-zero-length) edges."""
        keep = self.lengths() > tol
        return Skeleton(self.nodes, self.edges[keep])

    def filter_edges_in_domain(self, box: DomainBox) -> "Skeleton":
        """Drop edges whose midpoint falls inside a non-design (void) region."""
        if not box.non_design:
            return self
        keep = ~box.in_non_design(self.midpoints())
        return Skeleton(self.nodes, self.edges[keep])
