# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Geometry I/O layer — reads geometry sources into a common representation."""
from .base import GeometryReader, DomainRepresentation
from .registry import register_reader, get_reader, list_readers
from . import fenics_reader
# Optional: only imports numpy at module scope, so a missing cadjoint
# install surfaces when the reader runs, not when ggp is imported.
from . import cadjoint_reader

__all__ = [
    "GeometryReader",
    "DomainRepresentation",
    "register_reader",
    "get_reader",
    "list_readers",
]
