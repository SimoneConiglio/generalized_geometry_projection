# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Discretisation layer — bridges geometry/problem with analysis backend."""
from .fem import FEMDiscretiser, AnalysisDomain

__all__ = [
    "FEMDiscretiser",
    "AnalysisDomain",
]
