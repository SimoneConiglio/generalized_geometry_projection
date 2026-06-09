import pytest
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.geometry.base_mapper import BaseMapper
from samo_ggp.geometry.ggp_2d_free import GGP2DMapper

def test_geometry_factory():
    mesh = UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=2)
    assert isinstance(mapper, BaseMapper)
    assert isinstance(mapper, GGP2DMapper)

def test_geometry_initial_design():
    mesh = UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=2)
    
    # Expecting 6 parameters per component (Xc, Yc, L, h, theta, Mc)
    x_init = mapper.get_initial_design(10.0, 10.0)
    assert len(x_init) == 12
    
def test_map_to_density():
    # Use dolfin_adjoint version of mesh to ensure tape tracking
    mesh = UnitSquareMesh(10, 10)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=1)

    # Test with component that definitely has material
    # Variables: Xc, Yc, L, h, theta, Mc
    ctrls = [Constant(0.5), Constant(0.5), Constant(0.2), Constant(0.2), Constant(0.0), Constant(1.0)]

    rho_ufl = mapper.map_to_density(ctrls)

    # Project to DG0 (Discontinuous Galerkin 0)
    # DG0 takes the average over each element. Since the underlying function
    # is bounded [0, 1], its average over any subdomain must also be [0, 1].
    # This avoids the Gibbs overshoots/undershoots of CG1 projection.
    V = FunctionSpace(mesh, "DG", 0)
    
    # Project with stop_annotating as we only want to check final density bounds
    with stop_annotating():
        rho_func = project(rho_ufl, V)

        # Values must be strictly between 0 and 1
        # We use get_local() to ensure we check all values
        vals = rho_func.vector().get_local()
        min_val = vals.min()
        max_val = vals.max()

        assert min_val >= 0.0
        assert max_val <= 1.0
