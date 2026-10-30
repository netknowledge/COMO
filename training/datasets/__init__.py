"""
COMO Datasets
=============

Training datasets for molecule-image recognition.

Private package — not included in the pip-publishable ``como`` distribution.
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
