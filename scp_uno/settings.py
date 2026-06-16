# Copyright (c) 2026 Charlie Vanaret
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Settings for the SCP framework."""

from __future__ import annotations
from gemseo.algos.opt.base_optimization_library import BaseOptimizerSettings
from pydantic import Field

class SCPSettings(BaseOptimizerSettings):
    """Settings for Sequential Convex Programming algorithms."""
    max_iter: int = Field(50, description="Maximum number of outer iterations.")
    inner_library: str = Field("UNO", description="GEMSEO library for inner subproblem.")
    inner_preset: str = Field("ipopt", description="Uno preset for subproblem.")
    inner_solver: str = Field("MMA", description="Uno subproblem solver / Inner algorithm name.")
    inner_max_iter: int = Field(500, description="Max iterations for Uno.")
    
    # MMA Specific settings (matching baseline)
    max_optimization_step: float = Field(0.05, description="MMA move limit.")
    max_asymptote_distance: float = Field(0.5, description="Max distance for asymptotes.")
    initial_asymptotes_distance: float = Field(0.05, description="Initial distance for asymptotes.")
    min_asymptote_distance: float = Field(0.001, description="Min distance for asymptotes.")
    asymptotes_distance_amplification_coefficient: float = Field(1.2, description="Expansion factor.")
    asymptotes_distance_reduction_coefficient: float = Field(0.7, description="Shrinkage factor.")
    
    xtol_rel: float = Field(1e-12, description="Relative tolerance on design variables.")
    ftol_rel: float = Field(1e-12, description="Relative tolerance on objective.")
    xtol_abs: float = Field(1e-12, description="Absolute tolerance on design variables.")
    ftol_abs: float = Field(1e-12, description="Absolute tolerance on objective.")
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")

class UnoSettings(BaseOptimizerSettings):
    """Settings for the Uno solver wrapper."""
    preset: str = Field("filtersmma", description="Uno configuration preset.")
    solver: str = Field("MMA", description="Uno subproblem solver.")
    hessian: str = Field("identity", description="Hessian model.")
    max_iter: int = Field(500, description="Max iterations.")
    logger: str = Field("INFO", description="Uno logging level.")
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")
