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
    mesh = df.UnitSquareMesh(4, 4)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=1)
    
    import dolfin_adjoint as da
    ctrls = [da.Constant(0.5), da.Constant(0.5), da.Constant(0.2), da.Constant(0.2), da.Constant(0.0), da.Constant(1.0)]
    
    rho_ufl = mapper.map_to_density(ctrls)
    
    # Project to verify it compiles correctly
    V = df.FunctionSpace(mesh, "CG", 1)
    rho_func = df.project(rho_ufl, V)
    
    # Values should be roughly between 0 and 1
    min_val, max_val = rho_func.vector().get_local().min(), rho_func.vector().get_local().max()
    assert min_val >= -0.1
    assert max_val <= 1.1
