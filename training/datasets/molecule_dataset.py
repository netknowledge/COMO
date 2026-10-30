"""
COMO Molecule Dataset (Synthetic)
==================================

Dataset that renders SMILES into molecule images on-the-fly and produces
chartok_coords token sequences + bond matrices for training.

Uses ``drawing_engine`` for molecule rendering with Indigo backend.
"""

from typing import Dict, List, Optional, Tuple
import random

import os
import fcntl
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from func_timeout import FunctionTimedOut

from como.vocab import ComoVocab
from como.augmentations import (
    ScanEdgeLines,
    CropWhiteTrain,
    PadWhite,
    PadToSquare,
    AdaptiveResize,
)
from ..ddp_utils import is_main_process

from .. import drawing_engine as _drawing_engine


def _generate_image_from_smiles(*args, **kwargs):
    return _drawing_engine.generate_image_from_smiles(*args, **kwargs)


def _generate_image_from_wild_smiles(*args, **kwargs):
    return _drawing_engine.generate_image_from_wild_smiles(*args, **kwargs)


def _blank_image(*args, **kwargs):
    return _drawing_engine._blank_image(*args, **kwargs)


class MoleculeDataset(Dataset):
    """Dataset that renders SMILES into molecule images and produces
    chartok_coords token sequences + bond matrices for training.

    Rendering runs in a child process with a hard timeout (``render_pool``).
    Molecules that exceed the budget are recorded in a bad list and served as
    blank images from then on, so no epoch ever pays for them again.
    """
    
    def __init__(
        self, 
        vocab: ComoVocab,
        smiles_list: List[str],
        shuffle_smiles: bool = True,
        image_size: Tuple[int, int] = (384, 384),
        mol_augment: bool = True,
        img_augment: bool = True,
        geo_augment: bool = True,
        default_drawing_style: bool = False,
        wild_smiles: bool = False,
        render_badlist_path: Optional[str] = None,
    ):
        """Render bad list: ``render_badlist_path`` is a file (one SMILES per
        line) recording molecules whose render blew the hard timeout; they are
        served blank from then on.  Pass the run directory's path so the list
        survives ``--resume``; ``None`` keeps it in memory only.
        """
        self.vocab = vocab
        self.smiles_list = smiles_list
        
        if shuffle_smiles:
            if is_main_process():
                print('Shuffling SMILES list for dataset initialization...')
            rng = np.random.default_rng(42)
            shuffled_indices = rng.permutation(len(self.smiles_list))
            self.smiles_list = [self.smiles_list[i] for i in shuffled_indices]
            if is_main_process():
                print('SMILES list shuffled.')

        self.image_size = image_size
        self.mol_augment = mol_augment
        self.img_augment = img_augment
        self.geo_augment = geo_augment
        self.default_drawing_style = default_drawing_style
        self.wild_smiles = wild_smiles
        # ---- bad-molecule list -------------------------------------------------
        # A handful of SMILES take unbounded time to draw (pathological Indigo
        # layouts).  Each one burns a full render budget and would burn it again
        # every epoch, so once a molecule is known to time out it is served as a
        # blank image without being retried.
        # The file is append-only, one SMILES per line, shared by all workers
        # (reloaded when it grows) and lives in the run dir so resumes keep it.
        self._bad_path = render_badlist_path
        self._bad = set()
        self._bad_size = None
        self._reload_badlist()
        
        self.pad_to_square = PadToSquare(fill=255)

        # 1. Geometric augmentations
        self.geo_transforms_list = []
        if self.geo_augment:
            self.geo_transforms_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=0.5),
                CropWhiteTrain(pad=5, p=1.0),
                A.RandomCropFromBorders(crop_left=0.01, crop_right=0.01,
                                        crop_top=0.01, crop_bottom=0.01, p=0.5),
                PadWhite(pad_ratio=0.4, p=0.2),
            ]
        self.geo_transforms = A.Compose(
            self.geo_transforms_list, 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

        # 2. Blur and noise augmentation
        self.img_transforms_list = []
        if self.img_augment:
            self.img_transforms_list += [
                A.Downscale(scale_range=(0.2, 0.5),
                            interpolation_pair={'upscale': 3, 'downscale': 3}),
                A.Blur(),
                A.GaussNoise(),
                A.SaltAndPepper(),
                ScanEdgeLines(max_lines=3, p=0.3),
            ]
        self.img_transforms = A.Compose(self.img_transforms_list)

        # 3. Final transforms
        self.post_transforms = A.Compose([
            AdaptiveResize(height=self.image_size[0], width=self.image_size[1]),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def _render(self, smiles, wild=False, **kwargs):
        """Render through the timed child process (see ``render_pool``).

        Returns ``None`` if the render exceeded the budget.  The child is
        SIGKILLed on timeout, so a stuck Indigo call can neither poison the next
        render nor leak threads.  Falls back to in-process rendering when the
        pool is disabled (``COMO_RENDER_POOL=0``).
        """
        try:
            from render_pool import ENABLED, RenderPool
        except Exception:
            ENABLED = False
        if not ENABLED:
            fn = _generate_image_from_wild_smiles if wild else _generate_image_from_smiles
            try:
                return fn(smiles, **kwargs)
            except (FunctionTimedOut, Exception):
                return None
        self._reload_badlist()      # learn what other workers already hit
        if smiles in self._bad:
            return None
        pool = getattr(self, '_render_pool', None)
        if pool is None:
            pool = RenderPool(kind='wild' if wild else 'smiles')
            self._render_pool = pool
        out = pool.render(smiles, **kwargs)
        if out is None:
            self._note_bad(smiles)
        return out

    def _reload_badlist(self) -> None:
        """Reload the shared bad list if the file grew (one stat per call)."""
        if not self._bad_path:
            return
        try:
            size = os.path.getsize(self._bad_path)
        except OSError:
            return
        if size == self._bad_size:
            return
        try:
            with open(self._bad_path) as fh:
                self._bad = {ln.strip() for ln in fh if ln.strip()}
            self._bad_size = size
        except OSError:
            pass

    def _note_bad(self, smiles: str) -> None:
        """Mark *smiles* as unrenderable: memory + at most one line in the file.

        Read-check-append happens under an exclusive ``flock``, so workers that
        time out on the same molecule at the same instant still write it once.
        """
        if smiles in self._bad:
            return
        self._bad.add(smiles)
        if not self._bad_path:
            return
        try:
            parent = os.path.dirname(self._bad_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._bad_path, 'a+') as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    on_disk = {ln.strip() for ln in fh if ln.strip()}
                    self._bad.update(on_disk)      # pick up rows written meanwhile
                    if smiles not in on_disk:
                        fh.write(smiles + '\n')    # O_APPEND: no lost/torn writes
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_render_pool', None)   # never pickle a live subprocess
        return state

    def __getitem__(self, idx: int) -> Optional[Dict]:
        anchor_smiles = self.smiles_list[idx]

        img = _blank_image()
        success = False
        graph = None
        output_smiles = None

        rkwargs = dict(n_bins=None,
                       default_drawing_style=self.default_drawing_style,
                       debug=False)
        if not self.wild_smiles:
            rkwargs['mol_augment'] = self.mol_augment
        out = self._render(anchor_smiles, wild=self.wild_smiles, **rkwargs)
        if out is None or (isinstance(out, tuple) and len(out) == 3
                           and out[0] == '__ERR__'):
            img, output_smiles, graph, success = _blank_image(), None, None, False
        else:
            img, output_smiles, graph, success = out[0], out[1], out[2], out[3]
            # Drawings come at native canvas size (megapixels for large molecules)
            # and the augmentation/keypoint pipeline is super-linear in that size:
            # one 200-atom molecule cost 78 s *after* a fast render.  Downscale once
            # here, scaling the graph coordinates with it.
            if success and img is not None and graph and graph.get('coords') is not None:
                cap = int(os.environ.get('COMO_RENDER_MAXDIM', '1024'))
                h, w = img.shape[:2]
                scale = cap / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (max(1, int(round(w * scale))),
                                           max(1, int(round(h * scale)))),
                                      interpolation=cv2.INTER_AREA)
                    graph['coords'] = [[c[0] * scale, c[1] * scale]
                                       for c in graph['coords']]

        if not success or graph is None or output_smiles is None:
            dummy_img = _blank_image()
            dummy_img, _ = self.pad_to_square(dummy_img)
            img_tensor = self.post_transforms(image=dummy_img, keypoints=[])['image']
            return {'img': img_tensor, 'tok_id_seq': [], 'atom_indices': [],
                    'bond_mat': np.zeros((1, 1), dtype=int), 'success': False}

        coords = graph['coords']
        bond_mat = graph['edges']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        geo_result = self.geo_transforms(image=img, keypoints=coords)
        img = geo_result['image']
        coords = geo_result['keypoints']
        img, coords = self.pad_to_square(img, coords)
        img = self.img_transforms(image=img)['image']
        post_result = self.post_transforms(image=img, keypoints=coords)
        img_tensor = post_result['image']
        coords = post_result['keypoints']
        
        H, W = self.image_size
        n_bins = self.vocab.n_bins
        binned_coords = []
        for coord in coords:
            x = min(max(0, coord[0]), W - 1)
            y = min(max(0, coord[1]), H - 1)
            x_bin = int((x / W) * n_bins) 
            y_bin = int((y / H) * n_bins)
            x_bin = max(0, min(n_bins - 1, x_bin))
            y_bin = max(0, min(n_bins - 1, y_bin))
            binned_coords.append([x_bin, y_bin])

        try:
            tok_id_seq, atom_indices = self.vocab.smiles_to_sequence(output_smiles, binned_coords)
        except Exception:
            dummy_img = _blank_image()
            dummy_img, _ = self.pad_to_square(dummy_img)
            img_tensor = self.post_transforms(image=dummy_img, keypoints=[])['image']
            return {'img': img_tensor, 'tok_id_seq': [], 'atom_indices': [],
                    'bond_mat': np.zeros((1, 1), dtype=int), 'success': False}

        return {
            'img': img_tensor,
            'tok_id_seq': tok_id_seq,
            'atom_indices': atom_indices,
            'bond_mat': bond_mat,
            'success': True
        }
