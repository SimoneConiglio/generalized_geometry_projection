# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Base class for mesh / unit-cell initial-guess patterns."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .skeleton import DomainBox, Skeleton


class InitialGuessPattern(ABC):
    """Strategy that tessellates a design domain into a :class:`Skeleton`.

    A pattern turns the design domain into a coarse set of edges; each edge
    later seeds one GGP component.  The number of edges (and hence the number of
    components) is governed by the pattern's ``cell_size`` — implementing the
    "cell size drives the component count" decision.

    Patterns are pure NumPy and stateless; they never touch FEniCS or the
    formulation.  Subclasses register themselves with
    :func:`ggp.initialization.registry.register_pattern`.
    """

    #: spatial dimension the pattern is meant for (2 or 3)
    dim: int = 2

    @abstractmethod
    def generate(self, box: DomainBox, *, cell_size: float, **params) -> Skeleton:
        """Return a :class:`Skeleton` covering *box* at the given *cell_size*."""
        raise NotImplementedError
