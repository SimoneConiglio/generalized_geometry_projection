from .elasticity_2d import LinearElasticitySolver

class PhysicsFactory:
    @staticmethod
    def create_solver(physics_type, **kwargs):
        if physics_type == "Elasticity_2D":
            return LinearElasticitySolver(**kwargs)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")
