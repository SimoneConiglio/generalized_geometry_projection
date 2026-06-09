import dolfin as df
from dolfin_adjoint import *
import numpy as np
from samo_ggp.geometry.ggp_2d_free import GGP2DMapper
from samo_ggp.physics.elasticity import LinearElasticitySolver

def test_imports():
    print("Testing samo_ggp package imports...")
    try:
        mesh = df.UnitSquareMesh(10, 10)
        V_u = df.VectorFunctionSpace(mesh, "CG", 1)
        mapper = GGP2DMapper(mesh, num_components=2)
        print("GGP2DMapper: OK")
        
        # Mocking solver requirements
        bc = []
        ds_load = df.Measure("ds", domain=mesh)
        L_rhs_vec = df.Constant((0.0, -1.0))
        solver = LinearElasticitySolver(V_u, bc, ds_load, L_rhs_vec)
        print("LinearElasticitySolver: OK")
        
        print("\nAll internal modules functional.")
    except Exception as e:
        print(f"\nImport/Initialization failed: {e}")
        exit(1)

if __name__ == "__main__":
    test_imports()
