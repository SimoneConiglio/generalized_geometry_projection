import pytest
import dolfin as df
import numpy as np
from ggp.geometry.factory import GeometryFactory
from ggp.geometry.ggp_2d_alm import GGP2DALMMapper
from ggp.utils.alm_utils import create_alm_overhang_constraints

def test_alm_mapper_factory():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GeometryFactory.create_mapper("2D_ALM", mesh=mesh, num_layers=2, components_per_layer=1, layer_height=0.5)
    assert isinstance(mapper, GGP2DALMMapper)

def test_alm_initial_design():
    mesh = df.UnitSquareMesh(2, 2)
    mapper = GGP2DALMMapper(mesh, num_layers=10, components_per_layer=1, layer_height=0.1)
    x_init = mapper.get_initial_design(10.0, 1.0)
    # 10 layers * 1 comp/layer * 2 (X, L) + 10 heights + 1 mass = 31
    assert len(x_init) == 31

def test_overhang_constraints_shape():
    A, b = create_alm_overhang_constraints(num_layers=10, comp_per_layer=1, layer_height=1.0, alpha_deg=45.0)
    # 9 layer interfaces * 1 comp * 2 edges = 18 constraints
    # Columns: 2*10*1 + 2*1 + 2 = 24 (new interleaved layout)
    assert A.shape == (18, 24)
    assert len(b) == 18

def test_3d_overhang_constraints():
    from ggp.utils.alm_utils import create_alm_3d_overhang_constraints, compute_alm_3d_overhang_jacobian
    num_layers = 3
    comp_per_layer = 2
    layer_height = 0.5
    alpha_deg = 45.0
    
    # 3 layers * 2 comp/layer * 6 variables/comp = 36 variables
    x_vars = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 1.0] * 6)
    
    constraints = create_alm_3d_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, x_vars)
    # 2 interfaces * 2 components * 16 constraints = 64
    assert len(constraints) == 64
    
    jacobian = compute_alm_3d_overhang_jacobian(num_layers, comp_per_layer, layer_height, alpha_deg, x_vars)
    assert jacobian.shape == (64, 36)

def test_alm_mapper_map_to_density():
    mesh = df.UnitSquareMesh(5, 5)
    mapper = GGP2DALMMapper(mesh, num_layers=2, components_per_layer=2, layer_height=0.5)
    # 2 layers * 2 components/layer * 3 variables/component = 12 variables
    ctrls = [df.Constant(0.5) for _ in range(12)]
    rho = mapper.map_to_density(ctrls)
    
    V = df.FunctionSpace(mesh, "DG", 0)
    rho_f = df.project(rho, V)
    assert rho_f.vector().get_local().min() >= 0.0


def test_overhang_constraints_nonlinear_theta():
    """create_alm_overhang_constraints with x_vars computes theta-dependent delta."""
    import numpy as np
    from ggp.utils.alm_utils import create_alm_overhang_constraints
    nY, np_val = 3, 2
    n_tot = 2 * nY * np_val + 2 * np_val + 2
    x_vars = np.zeros(n_tot)
    x_vars[-1] = np.deg2rad(10.0)  # theta0 = 10°
    A_nl, b_nl = create_alm_overhang_constraints(
        num_layers=nY, comp_per_layer=np_val, layer_height=1.0, alpha_deg=45.0, x_vars=x_vars
    )
    A_lin, b_lin = create_alm_overhang_constraints(
        num_layers=nY, comp_per_layer=np_val, layer_height=1.0, alpha_deg=45.0
    )
    # Nonlinear deltas differ from linear when theta0 != 0
    assert A_nl.shape == A_lin.shape
    assert not np.allclose(b_nl, b_lin)


def test_bridge_length_constraints():
    """create_bridge_length_constraints produces correct shape and values."""
    from ggp.utils.alm_utils import create_bridge_length_constraints
    nY, np_val, BL = 4, 3, 5.0
    A, b = create_bridge_length_constraints(num_layers=nY, comp_per_layer=np_val, bridge_length=BL)
    # (nY-1) interfaces * np_val components * 2 edges = 18 constraints
    assert A.shape[0] == (nY - 1) * np_val * 2
    assert np.all(b == BL)
