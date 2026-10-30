"""
COMO USPTO Mol-File Dataset
============================

Dataset that loads patent molecule images + coordinates/edges from a CSV file.

Each row in the CSV has columns:
    file_path   – path to the molecule image
    SMILES      – ground truth SMILES string
    node_coords – stringified list of [x, y] coordinates
    edges       – (optional) stringified list of [u, v, bond_type] triples
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from como.vocab import ComoVocab
from como.augmentations import (
    ScanEdgeLines,
    CropWhiteTrain,
    PadToSquare,
    AdaptiveResize,
)
from como.utils import normalize_nodes
from ..ddp_utils import is_main_process

from .. import drawing_engine as _drawing_engine


def _blank_image(*args, **kwargs):
    return _drawing_engine._blank_image(*args, **kwargs)


class USPTOMolDataset(Dataset):
    """Dataset that loads patent images + coordinates/edges from a CSV file.

    Coordinates from mol files are normalized to [0,1] via
    ``normalize_nodes``, then scaled to image pixel space before
    augmentation. After augmentation they are quantized into vocab bins.
    """

    def __init__(
        self,
        vocab: ComoVocab,
        csv_path: str,
        data_dir: str,
        image_size: Tuple[int, int] = (384, 384),
        img_augment: bool = True,
        geo_augment: bool = True,
    ):
        self.vocab = vocab
        self.image_size = image_size
        self.df = pd.read_csv(csv_path)

        # R-group normalization fallback
        def _quick_normalise(smiles: str) -> str:
            if not isinstance(smiles, str):
                return smiles
            smiles = re.sub(r'\[(\d+)\*\]', r'[R\1]', smiles)
            if '*' not in smiles:
                return smiles
            parts: List[str] = []
            i = 0
            while i < len(smiles):
                if smiles[i] == '[':
                    j = smiles.find(']', i + 1)
                    if j == -1:
                        j = len(smiles) - 1
                    parts.append(smiles[i:j + 1])
                    i = j + 1
                elif smiles[i] == '*':
                    parts.append('[R]')
                    i += 1
                else:
                    parts.append(smiles[i])
                    i += 1
            return ''.join(parts)

        if self.df['SMILES'].str.contains(r'\*', na=False, regex=True).any():
            self.df['SMILES'] = self.df['SMILES'].apply(_quick_normalise)
            if is_main_process():
                print("[USPTOMolDataset] Applied quick R-group fallback "
                      "(consider pre-normalising with scripts/normalise_smiles.py)")

        # Filter out rows with non-vocab SMILES chars
        vocab_chars = set(self.vocab.token2idx.keys())
        def _has_unk(smiles: str) -> bool:
            return any(c not in vocab_chars for c in str(smiles))
        n_before = len(self.df)
        self.df = self.df[~self.df['SMILES'].apply(_has_unk)].reset_index(drop=True)
        n_filtered = n_before - len(self.df)
        if n_filtered > 0 and is_main_process():
            print(f"[USPTOMolDataset] Filtered {n_filtered}/{n_before} rows "
                  f"with non-vocab SMILES chars ({100*n_filtered/n_before:.1f}%)")

        # Resolve image paths
        if not os.path.isabs(self.df['file_path'].iloc[0]):
            self.df['file_path'] = self.df['file_path'].apply(
                lambda p: os.path.join(data_dir, p)
            )

        self.pad_to_square = PadToSquare(fill=255)

        # Geometric augmentations
        geo_list = []
        if geo_augment:
            geo_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=0.5),
                CropWhiteTrain(pad=5, p=1.0),
                A.RandomCropFromBorders(
                    crop_left=0.01, crop_right=0.01,
                    crop_top=0.01, crop_bottom=0.01, p=0.5,
                ),
                A.CropAndPad(
                    percent=(0.0, 0.4), sample_independently=True,
                    keep_size=False, fill=255, fill_mask=255, p=0.2,
                ),
            ]
        self.geo_transforms = A.Compose(
            geo_list,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

        # Image-level noise augmentation
        img_list = []
        if img_augment:
            img_list += [
                A.Downscale(scale_range=(0.5, 0.8),
                            interpolation_pair={'upscale': 3, 'downscale': 3}),
                A.Blur(),
                A.GaussNoise(),
                A.SaltAndPepper(),
                ScanEdgeLines(max_lines=3, p=0.3),
            ]
        self.img_transforms = A.Compose(img_list)

        # Final resize + normalise
        self.post_transforms = A.Compose([
            AdaptiveResize(height=image_size[0], width=image_size[1]),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df.iloc[idx]

        img = cv2.imread(row['file_path'])
        if img is None:
            return self._dummy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        smiles = row['SMILES']
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return self._dummy()

        try:
            raw_coords = np.array(eval(row['node_coords']), dtype=np.float64)
        except Exception:
            return self._dummy()
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 2 or len(raw_coords) == 0:
            return self._dummy()

        coords_norm = normalize_nodes(raw_coords)
        pixel_coords = coords_norm.copy()
        pixel_coords[:, 0] *= w
        pixel_coords[:, 1] *= h
        pixel_coords = pixel_coords.tolist()

        n_atoms = len(raw_coords)
        bond_mat = np.zeros((n_atoms, n_atoms), dtype=int)
        if 'edges' in self.df.columns:
            try:
                edge_list = eval(row['edges'])
                for u, v, t in edge_list:
                    if u < n_atoms and v < n_atoms:
                        if t <= 4:
                            bond_mat[u, v] = t
                            bond_mat[v, u] = t
                        else:
                            bond_mat[u, v] = t
                            bond_mat[v, u] = 11 - t
            except Exception:
                pass

        geo_result = self.geo_transforms(image=img, keypoints=pixel_coords)
        img = geo_result['image']
        coords_px = geo_result['keypoints']

        img, coords_px = self.pad_to_square(img, coords_px)
        img = self.img_transforms(image=img)['image']

        post_result = self.post_transforms(image=img, keypoints=coords_px)
        img_tensor = post_result['image']
        coords_px = post_result['keypoints']

        H, W = self.image_size
        n_bins = self.vocab.n_bins
        binned_coords = []
        for coord in coords_px:
            x = min(max(0, coord[0]), W - 1)
            y = min(max(0, coord[1]), H - 1)
            x_bin = max(0, min(n_bins - 1, int((x / W) * n_bins)))
            y_bin = max(0, min(n_bins - 1, int((y / H) * n_bins)))
            binned_coords.append([x_bin, y_bin])

        try:
            tok_id_seq, atom_indices = self.vocab.smiles_to_sequence(smiles, binned_coords)
        except Exception:
            return self._dummy()

        return {
            'img': img_tensor,
            'smiles': smiles,
            'tok_id_seq': tok_id_seq,
            'atom_indices': atom_indices,
            'bond_mat': bond_mat,
            'success': True,
        }

    def _dummy(self) -> Dict:
        dummy = _blank_image()
        dummy, _ = self.pad_to_square(dummy)
        img_tensor = self.post_transforms(image=dummy, keypoints=[])['image']
        return {
            'img': img_tensor,
            'smiles': '',
            'tok_id_seq': [],
            'atom_indices': [],
            'bond_mat': np.zeros((1, 1), dtype=int),
            'success': False,
        }
