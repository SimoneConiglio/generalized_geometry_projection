import pytest
import dolfin as df
import numpy as np
from samo_ggp.geometry.factory import GeometryFactory

def test_3d_geometry_initial_design():
    mesh = df.BoxMesh(df.Point(0,0,0), df.Point(1,1,1), 2, 2, 2)
    mapper = GeometryFactory.create_mapper("3D_Free", mesh=mesh, num_components=2)
    x_init = mapper.get_initial_design(1.0, 1.0, 1.0)
    # 2 components * 8 variables = 16
    assert len(x_init) == 16

def test_3d_map_to_density():
    mesh = df.BoxMesh(df.Point(0,0,0), df.Point(1,1,1), 5, 5, 5)
    from samo_ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline
    
    num_comp = 2
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_comp, mode='Free')
    
    # [X, Y, Z, L, h, Theta, Phi, Mc] * 2
    x_vars = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 1.0] * 2)
    
    rho = geom_disc._map_logic(x_vars, power=1.0)
    assert len(rho) == mesh.num_cells()
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0
