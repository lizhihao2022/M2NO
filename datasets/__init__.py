from .burgers_time import BurgersTimeDataset
from .darcy import DarcyDataset
from .ns import NavierStokesDataset
from .diff_react_2d import DiffReact2DDataset
from .advection import AdvectionDataset
from .era5 import ERA5Dataset

__all__ = [
    'BurgersTimeDataset',
    'DarcyDataset',
    'NavierStokesDataset',
    'DiffReact2DDataset',
    'AdvectionDataset',
    'ERA5Dataset',
]