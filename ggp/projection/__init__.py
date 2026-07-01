# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Projection layer — maps design variables to density fields."""
from .base import ProjectionMapper
from .registry import register_mapper, get_mapper, list_mappers
from .free_2d import Free2DMapper
from .free_3d import Free3DMapper
from .alm_2d import ALM2DMapper
from .alm_3d import ALM3DMapper
from .truss_2d import Truss2DMapper, build_ground_structure
from .truss_3d import Truss3DMapper, build_ground_structure_3d

__all__ = [
    "ProjectionMapper",
    "register_mapper",
    "get_mapper",
    "list_mappers",
    "Free2DMapper",
    "Free3DMapper",
    "ALM2DMapper",
    "ALM3DMapper",
    "Truss2DMapper",
    "build_ground_structure",
    "Truss3DMapper",
    "build_ground_structure_3d",
]
