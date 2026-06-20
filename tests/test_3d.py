import pytest
import dolfin as df
import numpy as np
from ggp.geometry.factory import GeometryFactory

def test_3d_geometry_initial_design():
    mesh = df.BoxMesh(df.Point(0,0,0), df.Point(1,1,1), 2, 2, 2)
    mapper = GeometryFactory.create_mapper("3D_Free", mesh=mesh, num_components=2)
    x_init = mapper.get_initial_design(1.0, 1.0, 1.0)
    # 2 components * 8 variables = 16
    assert len(x_init) == 16

def test_3d_map_to_density_ufl():
    mesh = df.BoxMesh(df.Point(0,0,0), df.Point(1,1,1), 4, 4, 4)
    mapper = GeometryFactory.create_mapper("3D_Free", mesh=mesh, num_components=1)
    # Variables: Xc, Yc, Zc, L, h, theta, phi, Mc
    ctrls = [df.Constant(0.5), df.Constant(0.5), df.Constant(0.5), 
             df.Constant(0.2), df.Constant(0.2), df.Constant(0.0), df.Constant(0.0), df.Constant(1.0)]
    rho_ufl = mapper.map_to_density(ctrls)
    
    V = df.FunctionSpace(mesh, "DG", 0)
    import dolfin_adjoint as da
    with da.stop_annotating():
        rho = df.project(rho_ufl, V)
    
    val = rho.vector().get_local()
    assert val.min() >= -1e-6
    assert val.max() <= 1.0 + 1e-6

def test_3d_map_to_density():
    mesh = df.BoxMesh.create([df.Point(0,0,0), df.Point(1,1,1)], [5, 5, 5], df.CellType.Type.hexahedron)
    from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline
    
    num_comp = 2
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_comp, mode='3D_Free')
    
    # [X, Y, L, h, Theta, Mc] for Free, but 3D_Free requires 8 vars!
    x_vars = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 1.0,  0.5, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 1.0])
    
    # Test _run
    geom_disc.execute({"x_vars": x_vars})
    rho = geom_disc.local_data["rho_V"]
    
    assert len(rho) == mesh.num_cells()
    assert rho.min() >= -1e-6
    assert rho.max() <= 1.0 + 1e-6

def test_3d_map_to_density_jacobian():
    mesh = df.BoxMesh.create([df.Point(0,0,0), df.Point(1,1,1)], [3, 3, 3], df.CellType.Type.hexahedron)
    from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline
    num_comp = 1
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_comp, mode='3D_Free')
    x_vars = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 1.0])
    
    geom_disc.execute({"x_vars": x_vars})
    geom_disc._compute_jacobian()
    
    jac = geom_disc.jac["rho_V"]["x_vars"]
    assert jac.shape == (mesh.num_cells(), 8)

