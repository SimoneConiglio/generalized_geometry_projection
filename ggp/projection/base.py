# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Abstract base class for projection mappers.

A :class:`ProjectionMapper` encapsulates the logic that converts a flat
design-variable vector into element-wise density fields (``rho_E``,
``rho_V``) **and** their analytic Jacobians.  Each concrete mapper
(``Free2DMapper``, ``ALM2DMapper``, …) implements only the
geometry-specific maths; the GEMSEO discipline wrapper delegates to
whichever mapper is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class ProjectionMapper(ABC):
    """Maps design variables → density field (and gradients).

    Subclass contract
    -----------------
    * :meth:`forward` — compute ``(rho_E, rho_V)`` from ``x_vars``.
    * :meth:`jacobian` — compute ``{"rho_E": drho_E/dx, "rho_V": drho_V/dx}``.
    * :meth:`num_vars_per_component` — how many design variables per
      geometric primitive (6 for 2-D Free, 8 for 3-D Free, 3 for ALM, …).
    * :meth:`default_bounds` — sensible ``(lb, ub)`` arrays for the design
      space given the domain extents.
    """

    @abstractmethod
    def forward(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute density fields.

        Parameters
        ----------
        x_vars : ndarray, shape (n_vars,)
            Flat design-variable vector (already unscaled).
        eval_coords : ndarray, shape (n_eval, dim)
            Element centroid (or evaluation-point) coordinates.
        power_E : float
            Penalisation exponent for stiffness density.
        power_V : float
            Penalisation exponent for volume density.

        Returns
        -------
        rho_E : ndarray, shape (n_elements,)
            Stiffness density per element.
        rho_V : ndarray, shape (n_elements,)
            Volume density per element.
        """

    @abstractmethod
    def jacobian(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Compute analytic Jacobians.

        Returns
        -------
        dict
            ``{"rho_E": drho_E_dx, "rho_V": drho_V_dx}`` where each
            value has shape ``(n_elements, n_vars)``.
        """

    @abstractmethod
    def num_vars_per_component(self) -> int:
        """Number of design variables per geometric primitive."""

    @abstractmethod
    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lb, ub)`` arrays for the full design-variable vector.

        Parameters
        ----------
        domain_extents : tuple of float
            ``(Lx, Ly)`` for 2-D, ``(Lx, Ly, Lz)`` for 3-D.
        num_components : int
            Number of geometric primitives.

        Returns
        -------
        lb, ub : ndarray, ndarray
            Lower and upper bounds, each of shape ``(n_vars,)``.
        """
