# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Projection layer — maps design variables to density fields."""
from .base import ProjectionMapper
from .registry import register_mapper, get_mapper, list_mappers

__all__ = [
    "ProjectionMapper",
    "register_mapper",
    "get_mapper",
    "list_mappers",
]
