from .ggp_2d_free import GGP2DMapper

class GeometryFactory:
    @staticmethod
    def create_mapper(geometry_type, **kwargs):
        if geometry_type == "2D_Free":
            return GGP2DMapper(**kwargs)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
