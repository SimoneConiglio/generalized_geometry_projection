# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Mesh / unit-cell-pattern initial-guess generation for GGP components.

A pattern tessellates the design domain into a :class:`Skeleton` of edges; each
edge seeds one GGP component (bar).  The encoder turns a skeleton into a
normalized design vector, with the bar thickness fitted so the seeded design
starts near the target volume fraction.

Patterns self-register via :func:`register_pattern`; retrieve them with
:func:`get_pattern` / :func:`list_patterns`.
"""
from __future__ import annotations

from .base import InitialGuessPattern
from .encoder import encode_skeleton, fit_thickness_to_volfrac
from .registry import get_pattern, is_pattern, list_patterns, register_pattern
from .skeleton import DomainBox, Skeleton

# Import built-in patterns so they register on package import.
from . import patterns  # noqa: F401,E402

__all__ = [
    "InitialGuessPattern",
    "Skeleton",
    "DomainBox",
    "encode_skeleton",
    "fit_thickness_to_volfrac",
    "register_pattern",
    "get_pattern",
    "is_pattern",
    "list_patterns",
]
