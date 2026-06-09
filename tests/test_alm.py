import pytest
import dolfin as df
import numpy as np
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.geometry.ggp_2d_alm import GGP2DALMMapper
from samo_ggp.utils.alm_utils import create_alm_overhang_constraints

def test_alm_mapper_factory():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_ALM", mesh=mesh, num_layers=2, components_per_layer=1, layer_height=0.5)
    assert isinstance(mapper, GGP2DALMMapper)

def test_alm_initial_design():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GGP2DALMMapper(mesh, num_layers=10, components_per_layer=1, layer_height=0.1)
    x_init = mapper.get_initial_design(10.0, 1.0)
    # 10 layers * 1 comp/layer * 3 variables = 30
    assert len(x_init) == 30

def test_overhang_constraints_shape():
    A, b = create_alm_overhang_constraints(num_layers=10, comp_per_layer=1, layer_height=1.0, alpha_deg=45.0)
    # 9 layer interfaces * 1 comp * 2 edges = 18 constraints
    assert A.shape == (18, 30)
    assert len(b) == 18

def test_alm_map_to_density():
    mesh = df.UnitSquareMesh(10, 10)
    mapper = GGP2DALMMapper(mesh, num_layers=2, components_per_layer=1, layer_height=0.5)
    
    import dolfin_adjoint as da
    # ctrls: [xc, w, m] * 2
    ctrls = [da.Constant(0.5), da.Constant(0.2), da.Constant(1.0),
             da.Constant(0.5), da.Constant(0.2), da.Constant(1.0)]
    
    rho_ufl = mapper.map_to_density(ctrls)
    V = df.FunctionSpace(mesh, "DG", 0)
    rho_func = df.project(rho_ufl, V)
    
    vals = rho_func.vector().get_local()
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0
