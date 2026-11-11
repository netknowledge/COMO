"""
COMO Datasets
=============

Training datasets for molecule-image recognition.

Private package — not included in the pip-publishable ``como`` distribution.

Also contains the render-timeout infrastructure used by ``MoleculeDataset``
(``render_pool`` + ``render_child``: hard-timeout molecule rendering in a
dedicated child process). It is deliberately not re-exported here —
``molecule_dataset.py`` imports it lazily so that importing the dataset
classes stays light.
"""

from .molecule_dataset import MoleculeDataset
from .uspto_mol_dataset import USPTOMolDataset
from .real_image_dataset import RealImageDataset
from .collate import collate_fn

__all__ = [
    'MoleculeDataset',
    'USPTOMolDataset',
    'RealImageDataset',
    'collate_fn',
]
