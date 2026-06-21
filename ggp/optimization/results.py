# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Results containers for GGP optimization."""
from __future__ import annotations

import dataclasses
from typing import Dict, Any, List, Optional
import numpy as np


@dataclasses.dataclass
class OptimisationResult:
    """Container for the results of a GGP optimization run."""
    problem_name: str
    algorithm: str
    status: str
    iterations: int
    objective_value: float
    max_constraint_violation: float
    design_variables: np.ndarray
    history: Dict[str, List[float]]
    mesh_path: Optional[str] = None
    execution_time_s: Optional[float] = None
    density_field: Optional[np.ndarray] = None
    eval_coords: Optional[np.ndarray] = None
    dim: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        return {
            "problem_name": self.problem_name,
            "algorithm": self.algorithm,
            "status": self.status,
            "iterations": self.iterations,
            "objective_value": self.objective_value,
            "max_constraint_violation": self.max_constraint_violation,
            "execution_time_s": self.execution_time_s,
        }
