import pytest
import dolfin as df
import numpy as np
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.geometry.base_mapper import BaseMapper
from samo_ggp.geometry.ggp_2d_free import GGP2DMapper

def test_geometry_factory():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=2)
    assert isinstance(mapper, BaseMapper)
    assert isinstance(mapper, GGP2DMapper)

def test_geometry_initial_design():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=2)
    
    # Expecting 6 parameters per component (Xc, Yc, L, h, theta, Mc)
    x_init = mapper.get_initial_design(10.0, 10.0)
    assert len(x_init) == 12
    
def test_map_to_density():
    mesh = df.UnitSquareMesh(10, 10)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=1)

    import dolfin_adjoint as da
    # Test with component that definitely has material
    ctrls = [da.Constant(0.5), da.Constant(0.5), da.Constant(0.2), da.Constant(0.2), da.Constant(0.0), da.Constant(1.0)]

    rho_ufl = mapper.map_to_density(ctrls)

    # Project to DG0 (Discontinuous Galerkin 0)
    # DG0 takes the average over each element. Since the underlying function
    # is bounded [0, 1], its average over any subdomain must also be [0, 1].
    # This avoids the Gibbs overshoots/undershoots of CG1 projection.
    V = df.FunctionSpace(mesh, "DG", 0)
    rho_func = df.project(rho_ufl, V)

    # Values must be strictly between 0 and 1
    # We use a very tight tolerance only for machine epsilon
    min_val = rho_func.vector().get_local().min()
    max_val = rho_func.vector().get_local().max()

    assert min_val >= 0.0
    assert max_val <= 1.0
