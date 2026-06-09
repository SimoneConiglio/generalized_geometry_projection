from .ggp_2d_free import GGP2DMapper
from .ggp_2d_alm import GGP2DALMMapper

class GeometryFactory:
    @staticmethod
    def create_mapper(geometry_type, **kwargs):
        if geometry_type == "2D_Free":
            return GGP2DMapper(**kwargs)
        elif geometry_type == "2D_ALM":
            return GGP2DALMMapper(**kwargs)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
