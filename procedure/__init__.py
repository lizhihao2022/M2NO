from .advection import advection_procedure
from .burgers import burgers_time_procedure
from .darcy import darcy_procedure
from .ns import ns_procedure
from .difftract import diffreact_2d_procedure
from .era5 import era5_procedure


__all__ = [
    "advection_procedure",
    "burgers_time_procedure",
    "darcy_procedure",
    "ns_procedure",
    "diffreact_2d_procedure",
    "era5_procedure",
]
