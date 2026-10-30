"""
MolDiscriminator — Contrastive Learning for Molecule Image Similarity
=====================================================================

Siamese network (EfficientNet_V2_S backbone -> 128-D L2-normalized embeddings)
trained with **symmetric InfoNCE / NT-Xent loss** on contrastive pairs:

  view1 = image of molecule M rendered with Indigo or RDKit (augmentation A)
  view2 = same molecule M, different rendering / augmentation (augmentation B)

Negatives come from two sources:
  1. **In-batch negatives** (always): all other samples' views in the batch.
     With batch size B, each anchor gets 2(B-1) in-batch negatives.
  2. **CReM hard negatives** (optional, default on): chemically reasonable
     structural mutations via CReM library. These provide structurally similar
     but distinct molecules — critical for fine-grained visual discrimination.

Downstream use: frozen encoder for visual reward in COMO Stage 2 RL,
where reward = cos_sim(encode(GT_image), encode(render(pred_SMILES))).

Key design choices:
  - Pair-based contrastive learning (SimCLR-style, not triplet-based)
  - Symmetric InfoNCE loss (both views as anchors)
  - CReM hard negatives added to the in-batch negative pool (fixed radius=3)
  - Per-sample random renderer selection (Indigo or RDKit)

Launch training (DDP is managed internally via mp.spawn):
  python train_MolDiscriminator.py
  # Do NOT use torchrun — it would double-spawn DDP workers and deadlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import random
from tqdm import tqdm
import os
import socket
import tempfile
from func_timeout import FunctionTimedOut
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import signal
import gc
import json

from .drawing_engine import (generate_image_from_smiles, _blank_image)
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
import warnings

# ======================== CReM module-level import ========================
try:
    from crem.crem import mutate_mol as _crem_mutate_mol
    _CREM_AVAILABLE = True
except ImportError:
    _crem_mutate_mol = None
    _CREM_AVAILABLE = False

# ======================== Performance Optimizations ========================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Use 'forkserver' to avoid segfaults from Indigo C library in forked workers
_DL_MP_CONTEXT = 'forkserver'


# ======================== CReM Configuration ========================
# Path to CReM fragment database — must be downloaded separately.
# Recommended: chembl33_f5.db from https://doi.org/10.5281/zenodo.16909329
# Set via environment variable or override in training config.
_CREM_DB_PATH = os.environ.get(
    'CREM_DB_PATH',
    os.path.join(os.path.dirname(__file__), 'data', 'chembl33_f5.db'),
)


def _worker_init_fn(worker_id: int):
    """Re-initialize per-worker state for forked DataLoader workers.

    Called once per worker after fork.  Three categories of init:

    1. **Thread safety** — disable OpenCV threading to avoid contention.
    2. **Logging** — suppress RDKit Python-level logs.
    3. **Reproducibility** — seed numpy/random per worker.
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    RDLogger.DisableLog('rdApp.*')
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    """Create a warmup + cosine-annealing LR schedule function."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return lr_lambda


def _resolve_tb_log_dir(
    save_path: Optional[str],
    resume: bool,
) -> str:
    """Determine the TensorBoard log directory."""
    if resume and save_path:
        logs_root = os.path.join(save_path, 'logs')
        if os.path.isdir(logs_root):
            existing = sorted(os.listdir(logs_root))
            if existing:
                return os.path.join(logs_root, existing[-1])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_path:
        return os.path.join(save_path, 'logs', timestamp)
    return f'./runs/{timestamp}'


# ======================== Image Transforms (consistent with COMO) ========================

class AdaptiveResize(A.ImageOnlyTransform):
    """Resize with adaptive interpolation.

    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    """

    def __init__(self, height, width, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        h, w = img.shape[:2]
        interpolation = cv2.INTER_AREA if (h > self.height or w > self.width) else cv2.INTER_LINEAR
        return cv2.resize(img, (self.width, self.height), interpolation=interpolation)

    def get_transform_init_args_names(self):
        return ('height', 'width')


class PadToSquare:
    """
    Simple callable that pads an image to be square and adjusts keypoints.
    Not an Albumentations transform - used separately.
    """
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, image, keypoints=None):
        h, w = image.shape[:2]
        target_dim = max(h, w)

        pad_h = target_dim - h
        pad_w = target_dim - w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if image.ndim == 3 and image.shape[2] == 3:
            fill_value = (float(self.fill), float(self.fill), float(self.fill))
        else:
            fill_value = (float(self.fill),)

        padded_img = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=fill_value
        )

        if keypoints is not None:
            adjusted_kps = []
            for kp in keypoints:
                x, y = kp[0], kp[1]
                adjusted_kps.append([x + left, y + top])
            return padded_img, adjusted_kps

        return padded_img, None


def _sanitize_smiles(smiles: str) -> str:
    """Remove hydrogen-only fragments from multi-fragment SMILES.

    PubChem SMILES occasionally contain standalone H species ([HH], [H-],
    [H+], [2H], etc.) as disconnected fragments.  These cause RDKit C++
    warnings during sanitisation ("not removing hydrogen atom without
    neighbors") and add no chemical information.

    Only fragments where **every** atom is hydrogen are stripped; real
    multi-component molecules (salts, co-crystals, mixtures) are kept
    intact — important for downstream Stage 2 RL which encounters
    multi-fragment molecules in real patent images.
    """
    parts = smiles.split('.')
    kept = []
    for part in parts:
        mol = Chem.MolFromSmiles(part, sanitize=False)
        if mol is None:
            kept.append(part)  # keep unparseable parts as-is
            continue
        # Drop fragment if it contains ONLY hydrogen atoms
        if all(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            continue
        kept.append(part)
    return '.'.join(kept) if kept else smiles


# ======================== CReM Negative Generation ========================

def _generate_crem_negative(
    smiles: str,
    db_name: str,
    radius: int = 3,
    max_replacements: int = 50,
    debug: bool = False,
) -> Optional[str]:
    """Generate a chemically reasonable mutant SMILES using CReM.

    Uses ``crem.mutate_mol`` to produce 1-step structural mutations of the
    input molecule. A random mutant is selected from the results.

    Args:
        smiles: Input SMILES string.
        db_name: Path to the CReM fragment database file.
        radius: Context radius for fragment matching (3=conservative/hard, 1=aggressive/easy).
        max_replacements: Maximum number of candidate mutations to generate.
        debug: Print debug information.

    Returns:
        A mutated SMILES string, or ``None`` if no valid mutation was found.
    """
    if not _CREM_AVAILABLE:
        if debug:
            print('[crem] crem package not installed')
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mutations = list(_crem_mutate_mol(
            mol,
            db_name=db_name,
            radius=radius,
            max_replacements=max_replacements,
            return_mol=False,
            max_inc=2,
            min_inc=-2,
        ))
    except Exception as e:
        if debug:
            print(f'[crem] mutate_mol failed: {e}')
        return None

    if not mutations:
        if debug:
            print(f'[crem] no mutations for radius={radius}')
        return None

    # Try up to 3 random candidates to find a valid, non-identical mutation.
    # A single random pick occasionally lands on an invalid or identical
    # SMILES; sampling a few more avoids unnecessarily dropping the hard neg.
    anchor_canonical = Chem.MolToSmiles(mol)
    candidates = random.sample(mutations, min(3, len(mutations)))
    for mutant_smiles in candidates:
        mutant_mol = Chem.MolFromSmiles(mutant_smiles)
        if mutant_mol is None:
            continue
        canonical = Chem.MolToSmiles(mutant_mol)
        if canonical == anchor_canonical:
            continue
        if debug:
            print(f'[crem] {smiles[:60]} -> {canonical[:60]} (radius={radius}, '
                  f'n_candidates={len(mutations)})')
        return canonical

    if debug:
        print(f'[crem] all {len(candidates)} candidates invalid or identical')
    return None


# ======================== RDKit Fast Render ========================

def _fast_render_smiles(smiles: str) -> Tuple[np.ndarray, bool]:
    """Render SMILES to a numpy RGB image using RDKit Draw (~5 ms/call).

    Returns:
        (image, success) where image is an (H, W, 3) uint8 RGB array,
        or a blank white image on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _blank_image(), False
        img_pil = Draw.MolToImage(mol, size=(300, 300))
        img_np = np.array(img_pil)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        return img_np, True
    except Exception:
        return _blank_image(), False


# ======================== Dataset ========================

class ContrastiveDataset(Dataset):
    """Pair-based contrastive dataset for molecule image similarity learning.

    Each sample produces:
      (view1, view2, hard_neg, hard_neg_valid)

    - ``view1`` and ``view2``: two independently augmented renderings of the
      same molecule (positive pair).
    - ``hard_neg``: CReM-mutated molecule rendering (structurally similar
      but different), or blank if CReM failed / disabled.
    - ``hard_neg_valid``: 1.0 if ``hard_neg`` is a valid hard negative, 0.0
      otherwise. Used as a mask in the loss function.

    In-batch negatives (all other samples' views) are handled by the loss,
    not by the dataset. The dataset only provides the explicit hard negative.
    """

    def __init__(
        self,
        smiles_list: List[str],
        image_size: Tuple[int, int] = (384, 384),
        img_augment: bool = True,
        geo_augment: bool = True,
        default_drawing_style: bool = False,
        shuffle_smiles: bool = True,
        crem_db_path: str = '',
        crem_radius: int = 3,
        use_hard_negatives: bool = True,
        use_random_renderer: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            smiles_list: List of SMILES strings.
            image_size: Target image size (height, width).
            img_augment: Whether to apply image augmentation (blur, noise).
            geo_augment: Whether to apply geometric augmentation (rotation, crop).
            default_drawing_style: Whether to use default drawing style for Indigo.
                When False, randomized drawing styles provide visual diversity
                for positive pairs (different rendering of the same molecule).
            shuffle_smiles: Whether to shuffle the SMILES list at initialization.
            crem_db_path: Path to CReM fragment database file.
            crem_radius: CReM context radius for hard negative generation.
                Higher radius = more conservative mutations = harder negatives.
                Typical values: 1 (aggressive/easy), 3 (conservative/hard).
            use_hard_negatives: Whether to generate CReM hard negatives.
                When False, hard_neg is a blank image with mask=0.
            use_random_renderer: If True, randomly pick Indigo or RDKit per sample.
                If False, always use Indigo (generate_image_from_smiles).
            debug: Print detailed debug information.
        """
        self.smiles_list = smiles_list
        self.crem_db_path = crem_db_path or _CREM_DB_PATH
        self.crem_radius = crem_radius
        self.use_hard_negatives = use_hard_negatives
        self.use_random_renderer = use_random_renderer
        self.debug = debug

        # Sanitize: remove hydrogen-only fragments ([HH], [H-], etc.)
        # that trigger C++ warnings, but keep real multi-fragment molecules
        sanitized = []
        for smi in self.smiles_list:
            if '.' in smi:
                sanitized.append(_sanitize_smiles(smi))
            else:
                sanitized.append(smi)
        self.smiles_list = sanitized

        if shuffle_smiles:
            print('Shuffling SMILES list for dataset initialization...')
            rng = np.random.default_rng(42)
            shuffled_indices = rng.permutation(len(self.smiles_list))
            self.smiles_list = [self.smiles_list[i] for i in shuffled_indices]
            print('SMILES list shuffled.')
        self.image_size = image_size
        self.img_augment = img_augment
        self.geo_augment = geo_augment
        self.default_drawing_style = default_drawing_style

        # Separate pad to square operation (handles keypoints manually)
        self.pad_to_square = PadToSquare(fill=255)

        # 1. Geometric transforms (rotation, crop)
        self.geo_transforms_list = []
        if self.geo_augment:
            self.geo_transforms_list += [
                A.Affine(rotate=(-90, 90), fit_output=True, fill=255, p=1.0),
                A.RandomCropFromBorders(crop_left=0.01, crop_right=0.01, crop_top=0.01, crop_bottom=0.01, p=0.5),
                A.CropAndPad(percent=(0.0, 0.4), sample_independently=True, keep_size=False, fill=255, fill_mask=255, p=0.2),
            ]
        self.geo_transforms = A.Compose(self.geo_transforms_list)
        # 2. Image augmentation (blur, noise)
        self.img_transforms_list = []
        if self.img_augment:
            self.img_transforms_list += [
                A.Downscale(scale_range=(0.5, 0.8), interpolation_pair={'upscale': 3, 'downscale': 3}),
                A.Blur(),
                A.GaussNoise(),
                A.SaltAndPepper()
            ]
        self.img_transforms = A.Compose(self.img_transforms_list)
        # 3. Final transforms (resize, grayscale, normalize -- consistent with COMO)
        self.post_transforms = A.Compose([
            AdaptiveResize(height=self.image_size[0], width=self.image_size[1]),
            A.ToGray(num_output_channels=3),  # Keep 3 channels for pretrained backbone
            A.Normalize(),  # ImageNet normalization
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.smiles_list)

    # ---- Rendering helpers ----

    def _render_with_indigo(self, smiles: str) -> Optional[np.ndarray]:
        """Render using Indigo drawing engine with randomized drawing style.

        Returns an RGB numpy array, or None on failure / timeout.

        Always uses ``mol_augment=False`` because mol_augment modifies the
        molecule itself (adds R-groups, condensed formulas, etc.), which
        would break the positive-pair assumption.  Visual diversity comes
        from ``_generate_style_config`` (random line thickness, fonts,
        layout, comments, aromatize/dearomatize) when
        ``default_drawing_style=False``.
        """
        try:
            img, _, _, success, _, _ = generate_image_from_smiles(
                smiles, mol_augment=False,
                default_drawing_style=self.default_drawing_style, debug=False)
            if success:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (FunctionTimedOut, Exception):
            pass
        return None

    def _render_with_rdkit(self, smiles: str) -> Optional[np.ndarray]:
        """Render using RDKit Draw (fast, deterministic)."""
        img, success = _fast_render_smiles(smiles)
        return img if success else None

    def _render_smiles(self, smiles: str, use_rdkit: bool) -> Optional[np.ndarray]:
        """Render SMILES with the chosen renderer."""
        if use_rdkit:
            return self._render_with_rdkit(smiles)
        return self._render_with_indigo(smiles)

    # ---- Hard negative generation ----

    def _generate_crem_negative_image(
        self, anchor_smiles: str, use_rdkit: bool
    ) -> Optional[np.ndarray]:
        """Generate a hard negative image via CReM mutation."""
        mutant_smiles = _generate_crem_negative(
            anchor_smiles,
            db_name=self.crem_db_path,
            radius=self.crem_radius,
            debug=self.debug,
        )
        if mutant_smiles is None:
            return None
        return self._render_smiles(mutant_smiles, use_rdkit=use_rdkit)

    # ---- Transforms ----

    def _apply_transforms(self, img: np.ndarray) -> torch.Tensor:
        """Apply geo, pad, image, and post transforms to a single image."""
        img = self.geo_transforms(image=img)['image']
        img, _ = self.pad_to_square(img)
        img = self.img_transforms(image=img)['image']
        return self.post_transforms(image=img)['image']

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (view1, view2, hard_neg, hard_neg_valid).

        view1 and view2 are two independent augmented renderings of the same
        molecule. hard_neg is a CReM-mutated molecule rendering (or blank
        placeholder). hard_neg_valid is 1.0 if CReM succeeded, 0.0 otherwise.
        """
        # Retry with alternate indices if rendering fails (R2 fix)
        _MAX_SAMPLE_RETRIES = 5
        view1_img = None
        view2_img = None
        anchor_smiles = None

        for attempt in range(_MAX_SAMPLE_RETRIES):
            actual_idx = (idx + attempt) % len(self.smiles_list)
            anchor_smiles = self.smiles_list[actual_idx]

            # Decide renderer independently for each view to enable
            # cross-renderer pairs (Indigo/RDKit) when use_random_renderer=True
            use_rdkit_v1 = self.use_random_renderer and random.random() < 0.5
            use_rdkit_v2 = self.use_random_renderer and random.random() < 0.5

            # Generate view1 (first rendering with random drawing style)
            view1_img = self._render_smiles(anchor_smiles, use_rdkit=use_rdkit_v1)
            # Generate view2 (second rendering, independent drawing style/renderer)
            view2_img = self._render_smiles(anchor_smiles, use_rdkit=use_rdkit_v2)

            if view1_img is not None and view2_img is not None:
                break

        if view1_img is None or view2_img is None:
            blank = _blank_image()
            return (
                self._apply_transforms(blank),
                self._apply_transforms(blank),
                self._apply_transforms(blank),
                torch.tensor(0.0),
            )

        # Generate hard negative (CReM mutation)
        hn_valid = 0.0
        hn_img = None
        if self.use_hard_negatives:
            use_rdkit_hn = self.use_random_renderer and random.random() < 0.5
            hn_img = self._generate_crem_negative_image(anchor_smiles, use_rdkit_hn)
            if hn_img is not None:
                hn_valid = 1.0
        if hn_img is None:
            hn_img = _blank_image()  # placeholder, masked out in loss

        return (
            self._apply_transforms(view1_img),
            self._apply_transforms(view2_img),
            self._apply_transforms(hn_img),
            torch.tensor(hn_valid),
        )


# ======================== Model ========================

class SiameseNetwork(nn.Module):
    """Siamese network for embedding molecule images.

    Architecture:
        EfficientNet_V2_S backbone (1280-d pooled features)
        -> MLP projection head: Linear(1280, 640) -> ReLU -> Linear(640, 128)
        -> L2-normalisation

    All branches share the same weights (true Siamese).  The ``forward``
    method accepts 1–N image tensors and returns a tuple of L2-normalised
    embeddings suitable for cosine-similarity based losses.
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_v2_s(weights=weights)
        feat_dim = efficientnet.classifier[1].in_features  # 1280
        efficientnet.classifier = nn.Identity()
        self.backbone = efficientnet
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.proj(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode one or more image tensors through the shared backbone."""
        return tuple(self.forward_once(x) for x in inputs)


# ======================== InfoNCE / NT-Xent Loss ========================

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) loss with optional hard negative pool.

    For each anchor in {view1, view2}:
        L = -log( exp(sim(a, p) / t) / sum_negatives exp(sim(a, n_k) / t) )

    Negatives include:
      - In-batch: all other samples' view1 and view2 embeddings (2(B-1) per anchor)
      - Optional: CReM hard negatives pooled from all samples with valid mask

    The loss is symmetric: both view1 and view2 serve as anchors, and the
    two losses are averaged. This directly optimises cosine similarity,
    aligning with the downstream visual reward in COMO Stage 2.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        hard_neg: Optional[torch.Tensor] = None,
        hard_neg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute symmetric InfoNCE loss with in-batch + optional hard negatives.

        Args:
            view1:         (B, D) L2-normalised embeddings (first view)
            view2:         (B, D) L2-normalised embeddings (second view / positive)
            hard_neg:      (B, D) L2-normalised embeddings of CReM-mutated molecules,
                           or None to use only in-batch negatives.
            hard_neg_mask: (B,) boolean mask, True where hard_neg[i] is valid.

        Returns:
            (loss, metrics_dict)
        """
        B = view1.size(0)
        device = view1.device
        idx = torch.arange(B, device=device)

        # ---- Build negative pool ----
        # pool layout: [view1(0..B-1), view2(B..2B-1), hard_neg(2B..3B-1)]
        #
        # For anchor view1_i:  positive = view2_i at pool index (B+i)
        #                      self     = view1_i at pool index (i)
        # For anchor view2_i:  positive = view1_i at pool index (i)
        #                      self     = view2_i at pool index (B+i)
        #
        # Both anchors exclude the SAME pair of indices {i, B+i}: self and
        # positive are removed from the denominator pool because the
        # positive is prepended separately as the first logit (pos_sim).
        has_hn = (hard_neg is not None and hard_neg_mask is not None)
        pool_parts = [view1, view2]
        if has_hn:
            pool_parts.append(hard_neg)
        pool = torch.cat(pool_parts, dim=0)  # (P, D) where P = 2B or 3B
        P = pool.size(0)

        # ---- Exclusion mask (shared by both view1 and view2 anchors) ----
        excl = torch.zeros(B, P, device=device, dtype=torch.bool)
        excl[idx, idx] = True        # view1_i: self / view2_i: its positive
        excl[idx, B + idx] = True    # view1_i: its positive / view2_i: self
        if has_hn:
            # Mask out invalid hard negatives for ALL anchors
            invalid = ~hard_neg_mask
            if invalid.any():
                invalid_cols = invalid.nonzero(as_tuple=True)[0] + 2 * B
                excl[:, invalid_cols] = True

        # ---- Positive similarity (symmetric: v1.v2 = v2.v1) ----
        pos_sim = (view1 * view2).sum(dim=1) / self.temperature  # (B,)

        # ---- Similarities against pool ----
        sim_v1 = torch.mm(view1, pool.t()) / self.temperature  # (B, P)
        sim_v2 = torch.mm(view2, pool.t()) / self.temperature  # (B, P)
        sim_v1.masked_fill_(excl, float('-inf'))
        sim_v2.masked_fill_(excl, float('-inf'))

        # ---- InfoNCE: cross-entropy with positive at index 0 ----
        logits_v1 = torch.cat([pos_sim.unsqueeze(1), sim_v1], dim=1)  # (B, 1+P)
        logits_v2 = torch.cat([pos_sim.unsqueeze(1), sim_v2], dim=1)  # (B, 1+P)
        labels = torch.zeros(B, device=device, dtype=torch.long)
        loss = (F.cross_entropy(logits_v1, labels) + F.cross_entropy(logits_v2, labels)) / 2

        # ---- Diagnostic metrics ----
        with torch.no_grad():
            avg_pos_sim = (view1 * view2).sum(dim=1).mean().item()

            # Hard negative cosine similarity (only valid ones)
            if has_hn and hard_neg_mask.any():
                valid_hn_sim = (view1[hard_neg_mask] * hard_neg[hard_neg_mask]).sum(dim=1)
                avg_hn_sim = valid_hn_sim.mean().item()
                hn_rate = hard_neg_mask.float().mean().item()
            else:
                avg_hn_sim = 0.0
                hn_rate = 0.0

            # Contrastive accuracy: is pos_sim the max logit? (using v1 anchors)
            max_neg_v1 = sim_v1.max(dim=1).values
            acc = (pos_sim > max_neg_v1).float().mean().item()

        metrics = {
            'avg_pos_cosine_sim': avg_pos_sim,
            'avg_hard_neg_cosine_sim': avg_hn_sim,
            'contrastive_accuracy': acc,
            'hard_neg_rate': hn_rate,
        }
        return loss, metrics


# ======================== Helpers ========================

class _EmbeddingWrapper(nn.Module):
    """Wrapper so that nn.DataParallel calls forward_once via forward()."""
    def __init__(self, siamese_model: SiameseNetwork):
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.siamese_model.forward_once(x)


def _find_free_port() -> int:
    """Find a free port on localhost for DDP."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# ======================== DDP Training ========================

def _ddp_train_worker(rank: int, world_size: int, config: Dict[str, Any]):
    """Per-process DDP training worker launched by mp.spawn."""
    RDLogger.DisableLog('rdApp.*')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(config['master_port'])
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    # Prevent NCCL watchdog from killing processes while DataLoader is recovering
    os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '1800')
    nccl_timeout = timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=nccl_timeout)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    is_main = (rank == 0)

    def _graceful_exit(signum, frame):
        raise SystemExit(f'Rank {rank} received signal {signum}')
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    try:
        _ddp_train_worker_body(rank, world_size, config, device, is_main)
    except Exception as e:
        if is_main:
            print(f'[Rank {rank}] Training error: {e}')
        raise
    finally:
        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()


def _ddp_train_worker_body(rank: int, world_size: int, config: Dict[str, Any],
                           device: torch.device, is_main: bool):
    """Core DDP training loop with symmetric InfoNCE and CReM hard negatives."""

    # ---- Reproducibility ----
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    # ---- Unpack config ----
    num_epochs: int        = config['num_epochs']
    batch_size: int        = config['batch_size']
    num_workers: int       = config['num_workers']
    save_path: str         = config['save_path']
    start_epoch: int       = config.get('resume_epoch', 0)
    use_amp: bool          = config.get('use_amp', True)
    warmup_ratio: float    = config.get('warmup_ratio', 0.02)
    temperature: float     = config.get('temperature', 0.07)
    crem_radius: int       = config.get('crem_radius', 3)
    use_hard_negatives: bool = config.get('use_hard_negatives', True)
    early_stopping_patience: Optional[int] = config['early_stopping_patience']

    # ---- Model ----
    model = SiameseNetwork(
        embedding_dim=config['embedding_dim'],
        pretrained=config['pretrained']
    ).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = InfoNCELoss(temperature=temperature)

    # ---- Datasets ----
    train_dataset = ContrastiveDataset(
        smiles_list=config['train_smiles'],
        image_size=config['image_size'],
        default_drawing_style=config['default_drawing_style'],
        crem_db_path=config.get('crem_db_path', _CREM_DB_PATH),
        crem_radius=crem_radius,
        use_hard_negatives=use_hard_negatives,
        use_random_renderer=config.get('use_random_renderer', False),
    )
    val_dataset = ContrastiveDataset(
        smiles_list=config['val_smiles'],
        shuffle_smiles=False,
        image_size=config['image_size'],
        default_drawing_style=config['default_drawing_style'],
        img_augment=False,
        geo_augment=False,
        crem_db_path=config.get('crem_db_path', _CREM_DB_PATH),
        crem_radius=crem_radius,
        use_hard_negatives=use_hard_negatives,
        use_random_renderer=config.get('use_random_renderer', False),
    )

    # ---- Samplers & DataLoaders ----
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    def _create_loaders():
        dl_common: Dict[str, Any] = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'worker_init_fn': _worker_init_fn,
        }
        if num_workers > 0:
            dl_common.update({
                'prefetch_factor': 4,
                'persistent_workers': False,  # Allow stuck workers to be cleaned up
                'multiprocessing_context': _DL_MP_CONTEXT,
                'timeout': 180,  # 3 min: balance between CReM rendering latency and deadlock detection
            })
        tl = DataLoader(train_dataset, sampler=train_sampler,
                        drop_last=True, **dl_common)
        vl = DataLoader(val_dataset, sampler=val_sampler, **dl_common)
        return tl, vl

    train_loader, val_loader = _create_loaders()

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=1e-6)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(warmup_steps, total_steps))

    # ---- TensorBoard ----
    writer = None
    if is_main:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        log_dir = _resolve_tb_log_dir(save_path, resume=(start_epoch > 0))
        writer = SummaryWriter(log_dir=log_dir)
        print(f'[Rank 0] TensorBoard logs: {writer.log_dir}')
        print(f'[Rank 0] Batch size per GPU: {batch_size}, GPUs: {world_size}, '
              f'Effective batch size: {batch_size * world_size}')

    # ---- AMP scaler ----
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if is_main:
        print(f'[Rank 0] Total steps: {total_steps}, Warmup: {warmup_steps} '
              f'({warmup_ratio*100:.1f}%), AMP={use_amp}, tau={temperature}')
        # ---- Log hyperparameters ----
        hparams = {
            'batch_size': batch_size,
            'effective_batch_size': batch_size * world_size,
            'learning_rate': config['learning_rate'],
            'warmup_ratio': warmup_ratio,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'image_size': str(config['image_size']),
            'embedding_dim': config['embedding_dim'],
            'temperature': temperature,
            'use_amp': use_amp,
            'use_hard_negatives': use_hard_negatives,
            'crem_radius': crem_radius,
            'crem_db_path': config.get('crem_db_path', ''),
            'use_random_renderer': config.get('use_random_renderer', False),
            'default_drawing_style': config.get('default_drawing_style', False),
            'pretrained': config['pretrained'],
            'num_workers': num_workers,
            'seed': seed,
            'world_size': world_size,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_steps': total_steps,
            'resume_epoch': start_epoch,
        }
        hparams_path = os.path.join(writer.log_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        hparams_md = '| Param | Value |\n|---|---|\n'
        for k, v in hparams.items():
            hparams_md += f'| {k} | {v} |\n'
        writer.add_text('Hyperparameters', hparams_md, global_step=0)

    # ---- Training-loop state ----
    early_stopping_enabled = early_stopping_patience is not None
    best_val_loss = float('inf')
    vals_no_improve = 0
    global_step = 0

    # ---- Resume checkpoint ----
    if start_epoch > 0:
        resume_state_path = os.path.join(save_path, 'training_state.pth')
        resume_model_path = os.path.join(save_path, f'epoch_{start_epoch}.pth')
        if os.path.exists(resume_state_path):
            ckpt = torch.load(resume_state_path, map_location=device,
                              weights_only=False)
            model.module.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if scaler is not None and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            global_step = ckpt.get('global_step',
                                   start_epoch * len(train_loader))
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            vals_no_improve = ckpt.get('vals_no_improve', 0)
            if is_main:
                print(f'[Rank 0] Resumed full training state '
                      f'(epoch={start_epoch}, global_step={global_step})')
        elif os.path.exists(resume_model_path):
            model.module.load_state_dict(
                torch.load(resume_model_path, map_location=device,
                           weights_only=True))
            steps_to_skip = start_epoch * len(train_loader)
            for _ in range(steps_to_skip):
                scheduler.step()
            global_step = steps_to_skip
            if is_main:
                print(f'[Rank 0] Resumed model weights from {resume_model_path}, '
                      f'scheduler fast-forwarded to step {global_step}')
        else:
            raise FileNotFoundError(
                f'Cannot resume: neither {resume_state_path} nor '
                f'{resume_model_path} found')

    # ==================== Epoch Loop ====================
    for epoch in range(start_epoch, num_epochs):

        # --- Train ---
        epoch_loss = 0.0
        epoch_metrics = {k: 0.0 for k in ['avg_pos_cosine_sim', 'avg_hard_neg_cosine_sim',
                                           'contrastive_accuracy', 'hard_neg_rate']}
        epoch_step_count = 0

        model.train()
        train_sampler.set_epoch(epoch)

        steps_in_epoch = len(train_loader)
        train_bar = tqdm(total=steps_in_epoch,
                         desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
                         disable=not is_main)
        train_iter = iter(train_loader)

        for _step in range(steps_in_epoch):
            # Fetch batch.  DataLoader timeout=60s catches hung C-extension workers.
            # On any error: kill all workers (recreate loader) and skip this step.
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if is_main:
                    warnings.warn(f'DataLoader error at epoch {epoch+1} '
                                  f'step {_step}: {e}')
                batch = None
                # Immediately recreate DataLoader to kill the stuck worker process
                del train_iter
                train_loader, val_loader = _create_loaders()
                train_iter = iter(train_loader)

            # ===== DDP-safe batch validity check =====
            # All ranks must agree to skip; otherwise one rank skips the
            # forward/backward while others do the NCCL all-reduce -> hang.
            batch_valid = batch is not None
            if world_size > 1:
                valid_tensor = torch.tensor(
                    [1 if batch_valid else 0],
                    dtype=torch.int32, device=device)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                batch_valid = valid_tensor.item() == 1

            if not batch_valid:
                train_bar.update(1)
                continue

            view1, view2, hard_neg, hn_valid = batch
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            hard_neg = hard_neg.to(device, non_blocking=True)
            hn_mask = hn_valid.to(device, non_blocking=True).bool()

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    v1_emb, v2_emb, hn_emb = model(view1, view2, hard_neg)
                    loss, metrics = criterion(v1_emb, v2_emb, hn_emb, hn_mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                v1_emb, v2_emb, hn_emb = model(view1, view2, hard_neg)
                loss, metrics = criterion(v1_emb, v2_emb, hn_emb, hn_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            epoch_step_count += 1

            if is_main:
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos': f'{metrics["avg_pos_cosine_sim"]:.3f}',
                    'hn': f'{metrics["hard_neg_rate"]:.0%}',
                })
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                writer.add_scalar('Train/pos_cosine_sim', metrics['avg_pos_cosine_sim'], global_step)
                writer.add_scalar('Train/hard_neg_cosine_sim', metrics['avg_hard_neg_cosine_sim'], global_step)
                writer.add_scalar('Train/contrastive_acc', metrics['contrastive_accuracy'], global_step)
                writer.add_scalar('Train/hard_neg_rate', metrics['hard_neg_rate'], global_step)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
            global_step += 1
            train_bar.update(1)
        train_bar.close()

        train_loss = epoch_loss / max(1, epoch_step_count)
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, epoch_step_count)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_pos_sim_sum = 0.0
        val_hn_sim_sum = 0.0
        val_acc_sum = 0.0
        val_hn_count = 0.0
        count = 0.0

        with torch.no_grad():
            for view1, view2, hard_neg, hn_valid in val_loader:
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                hard_neg = hard_neg.to(device, non_blocking=True)
                hn_mask = hn_valid.to(device, non_blocking=True).bool()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    v1, v2, hn = model(view1, view2, hard_neg)
                vloss, vmetrics = criterion(v1, v2, hn, hn_mask)

                bsz = view1.size(0)
                val_loss_sum += vloss.item() * bsz
                val_pos_sim_sum += vmetrics['avg_pos_cosine_sim'] * bsz
                val_acc_sum += vmetrics['contrastive_accuracy'] * bsz
                # Accumulate hard_neg sim weighted by count of valid hard negs
                n_valid_hn = hn_mask.sum().item()
                val_hn_sim_sum += vmetrics['avg_hard_neg_cosine_sim'] * n_valid_hn
                val_hn_count += n_valid_hn
                count += bsz

        # Gather across ranks
        metrics_t = torch.tensor(
            [val_loss_sum, val_pos_sim_sum, val_hn_sim_sum, val_acc_sum, count, val_hn_count],
            device=device, dtype=torch.float64
        )
        dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
        t_loss, t_pos, t_hn_sim, t_acc, t_cnt, t_hn_cnt = metrics_t.tolist()

        val_loss = t_loss / max(1, t_cnt)
        avg_pos_sim = t_pos / max(1, t_cnt)
        avg_hn_sim = t_hn_sim / max(1, t_hn_cnt)
        val_acc = t_acc / max(1, t_cnt)
        val_hn_rate = t_hn_cnt / max(1, t_cnt)

        if is_main:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: train={train_loss:.4f} | val={val_loss:.4f} | '
                  f'pos_sim={avg_pos_sim:.3f} | hn_sim={avg_hn_sim:.3f} | '
                  f'acc={val_acc:.3f} | hn_rate={val_hn_rate:.1%} | '
                  f'lr={current_lr:.2e}')
            writer.add_scalar('Val/loss', val_loss, global_step)
            writer.add_scalar('Val/pos_cosine_sim', avg_pos_sim, global_step)
            writer.add_scalar('Val/hard_neg_cosine_sim', avg_hn_sim, global_step)
            writer.add_scalar('Val/contrastive_acc', val_acc, global_step)
            writer.add_scalar('Val/hard_neg_rate', val_hn_rate, global_step)

            # --- Checkpointing ---
            if save_path:
                cached_sd = model.module.state_dict()
                torch.save(cached_sd,
                           os.path.join(save_path, f'epoch_{epoch+1}.pth'))
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': cached_sd,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'vals_no_improve': vals_no_improve,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_path, 'training_state.pth'))

            # --- Best-model tracking ---
            if val_loss < best_val_loss:
                vals_no_improve = 0
                best_val_loss = val_loss
                if save_path:
                    torch.save(cached_sd,
                               os.path.join(save_path, 'best.pth'))
                    print(f'  New best model (val_loss={best_val_loss:.4f}).')
            else:
                if early_stopping_enabled:
                    vals_no_improve += 1
                    print(f'  No improvement ({vals_no_improve}/{early_stopping_patience}).')

        # Broadcast early-stopping decision
        should_stop_flag = (
            early_stopping_enabled
            and vals_no_improve >= early_stopping_patience
        )
        should_stop = torch.tensor([1 if should_stop_flag else 0],
                                   device=device, dtype=torch.int)
        dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if is_main:
                print(f'Early stopping at epoch {epoch+1}.')
            break

    # Save final model
    if is_main:
        if save_path:
            torch.save(model.module.state_dict(),
                       os.path.join(save_path, 'final.pth'))
        if writer:
            writer.close()

    del model, optimizer, scheduler, criterion
    del train_loader, val_loader, train_dataset, val_dataset
    del train_sampler, val_sampler
    if scaler is not None:
        del scaler
    gc.collect()
    torch.cuda.empty_cache()


# ======================== MoleculeDiscriminator Wrapper ========================

class MoleculeDiscriminator:
    """Wrapper class for training and inference."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (384, 384),
        embedding_dim: int = 128,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
        pretrained: bool = True,
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.pretrained = pretrained

        print(f'Using device: {self.device}')

        self.model = SiameseNetwork(
            embedding_dim=embedding_dim,
            pretrained=pretrained
        ).to(self.device)
        self.criterion = InfoNCELoss(temperature=temperature)

        # Inference preprocessing (matches training post_transforms)
        self._pad_to_square = PadToSquare(fill=255)
        self._inference_transforms = A.Compose([
            AdaptiveResize(height=self.image_size[0], width=self.image_size[1]),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        smiles_list: List[str],
        num_smiles: Optional[int] = None,
        num_epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 4e-4,
        save_path: Optional[str] = None,
        val_split: float = 0.01,
        default_drawing_style: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        early_stopping_patience: Optional[int] = 5,
        use_ddp: bool = True,
        warmup_ratio: float = 0.02,
        use_amp: bool = True,
        crem_db_path: str = '',
        crem_radius: int = 3,
        use_hard_negatives: bool = True,
        use_random_renderer: bool = False,
        resume_epoch: int = 0,
    ) -> List[float]:
        """Train the discriminator with symmetric InfoNCE loss.

        Args:
            smiles_list: Full SMILES strings list.
            num_smiles: Number of SMILES to sample. None -> use all.
            num_epochs: Number of training epochs.
            batch_size: Batch size per GPU.
            learning_rate: Peak learning rate (after warmup).
            save_path: Directory to save checkpoints.
            val_split: Fraction of data for validation.
            default_drawing_style: Whether to use default Indigo style.
                When False, randomized drawing styles provide visual diversity.
            num_workers: DataLoader workers per process.
            seed: Random seed.
            early_stopping_patience: Epochs without improvement before stop.
                None to disable.
            use_ddp: Use DDP when multiple GPUs available.
            warmup_ratio: Fraction of steps for linear LR warmup.
            use_amp: Use automatic mixed precision.
            crem_db_path: Path to CReM fragment database.
            crem_radius: CReM context radius for hard negatives (default 3,
                higher = more conservative mutations = harder negatives).
            use_hard_negatives: Generate CReM hard negatives (default True).
            use_random_renderer: Randomly pick Indigo or RDKit per sample.
            resume_epoch: Resume from this epoch (0 = fresh start).

        Returns:
            List of per-epoch losses (empty in DDP mode).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ---- Sample SMILES ----
        if num_smiles is not None and num_smiles < len(smiles_list):
            original_n = len(smiles_list)
            rng_sample = np.random.default_rng(seed)
            n_sample = max(2, int(num_smiles))
            sample_idx = rng_sample.choice(original_n, size=n_sample, replace=False)
            smiles_list = [smiles_list[int(i)] for i in sample_idx]
            print(f'Sampled {n_sample}/{original_n} SMILES')

        n = len(smiles_list)
        if n < 2:
            raise ValueError("Need at least 2 SMILES for train/val split.")

        # ---- Train / val split ----
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = max(1, int(n * (1 - val_split)))
        train_idx, val_idx = indices[:split], indices[split:]
        if len(val_idx) == 0:
            val_idx = indices[-1:]
            train_idx = indices[:-1]

        train_smiles = [smiles_list[i] for i in train_idx]
        val_smiles = [smiles_list[i] for i in val_idx]
        print(f'Train: {len(train_smiles)} | Val: {len(val_smiles)}')

        world_size = torch.cuda.device_count()
        if use_ddp and world_size > 1:
            tmp_dir = None
            actual_save = save_path
            if save_path is None:
                tmp_dir = tempfile.mkdtemp(prefix='mol_disc_')
                actual_save = tmp_dir

            config: Dict[str, Any] = {
                'embedding_dim': self.embedding_dim,
                'temperature': self.temperature,
                'pretrained': self.pretrained,
                'image_size': self.image_size,
                'train_smiles': train_smiles,
                'val_smiles': val_smiles,
                'default_drawing_style': default_drawing_style,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'early_stopping_patience': early_stopping_patience,
                'save_path': actual_save,
                'master_port': _find_free_port(),
                'warmup_ratio': warmup_ratio,
                'seed': seed,
                'use_amp': use_amp,
                'crem_db_path': crem_db_path or _CREM_DB_PATH,
                'crem_radius': crem_radius,
                'use_hard_negatives': use_hard_negatives,
                'use_random_renderer': use_random_renderer,
                'resume_epoch': resume_epoch,
            }

            self.model.cpu()
            torch.cuda.empty_cache()

            print(f'Launching DDP training on {world_size} GPUs '
                  f'(port {config["master_port"]})...')
            try:
                ctx = mp.spawn(
                    _ddp_train_worker,
                    args=(world_size, config),
                    nprocs=world_size,
                    join=False,
                )
                # Poll until all workers finish; this lets us catch Ctrl+C
                while not ctx.join(timeout=1):
                    pass
            except KeyboardInterrupt:
                print('\nTraining interrupted by user. Killing workers...')
                for proc in ctx.processes:
                    if proc.is_alive():
                        proc.kill()
                for proc in ctx.processes:
                    proc.join(timeout=5)
            except Exception as e:
                print(f'DDP training failed: {e}')
                raise
            finally:
                for i in range(world_size):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                gc.collect()

            # Reload trained weights
            best_path = os.path.join(actual_save, 'best.pth')
            final_path = os.path.join(actual_save, 'final.pth')
            if os.path.exists(best_path):
                self.load_model(best_path)
                print('Loaded best model after DDP training.')
            elif os.path.exists(final_path):
                self.load_model(final_path)
                print('Loaded final model after DDP training.')

            # Clean up temporary directory
            if tmp_dir is not None and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return []
        else:
            return self._train_single(
                train_smiles=train_smiles,
                val_smiles=val_smiles,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                default_drawing_style=default_drawing_style,
                save_path=save_path,
                num_workers=num_workers,
                early_stopping_patience=early_stopping_patience,
                warmup_ratio=warmup_ratio,
                use_amp=use_amp,
                crem_db_path=crem_db_path,
                crem_radius=crem_radius,
                use_hard_negatives=use_hard_negatives,
                use_random_renderer=use_random_renderer,
            )

    def _train_single(
        self,
        train_smiles: List[str],
        val_smiles: List[str],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        default_drawing_style: bool,
        save_path: Optional[str],
        num_workers: int,
        early_stopping_patience: Optional[int],
        warmup_ratio: float = 0.02,
        use_amp: bool = True,
        crem_db_path: str = '',
        crem_radius: int = 3,
        use_hard_negatives: bool = True,
        use_random_renderer: bool = False,
    ) -> List[float]:
        """Single-GPU / CPU training loop with symmetric InfoNCE."""
        device = self.device

        # ---- Datasets ----
        train_dataset = ContrastiveDataset(
            smiles_list=train_smiles,
            image_size=self.image_size,
            default_drawing_style=default_drawing_style,
            crem_db_path=crem_db_path or _CREM_DB_PATH,
            crem_radius=crem_radius,
            use_hard_negatives=use_hard_negatives,
            use_random_renderer=use_random_renderer,
        )
        val_dataset = ContrastiveDataset(
            smiles_list=val_smiles,
            shuffle_smiles=False,
            image_size=self.image_size,
            default_drawing_style=default_drawing_style,
            img_augment=False,
            geo_augment=False,
            crem_db_path=crem_db_path or _CREM_DB_PATH,
            crem_radius=crem_radius,
            use_hard_negatives=use_hard_negatives,
            use_random_renderer=use_random_renderer,
        )

        # ---- DataLoaders ----
        pin_mem = device.type == 'cuda'
        dl_common: Dict[str, Any] = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_mem,
            'worker_init_fn': _worker_init_fn,
        }
        if num_workers > 0:
            dl_common.update({
                'prefetch_factor': 2,
                'persistent_workers': False,  # Allow stuck workers to be cleaned up
                'multiprocessing_context': _DL_MP_CONTEXT,
                'timeout': 180,  # 3 min: balance between CReM rendering latency and deadlock detection
            })

        def _make_train_loader():
            return DataLoader(train_dataset, shuffle=True,
                              drop_last=True, **dl_common)

        train_loader = _make_train_loader()
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_common)
        print(f'DataLoader: batch_size={batch_size}, workers={num_workers}, '
              f'device={device}')

        # ---- Optimizer & Scheduler ----
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                      weight_decay=1e-6)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, _make_lr_lambda(warmup_steps, total_steps))

        # ---- AMP ----
        use_amp_actual = use_amp and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp_actual else None
        print(f'Total steps: {total_steps}, Warmup: {warmup_steps} '
              f'({warmup_ratio*100:.1f}%), AMP={use_amp_actual}')

        # ---- TensorBoard ----
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        log_dir = _resolve_tb_log_dir(save_path, resume=False)
        writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs: {writer.log_dir}')

        # ---- Log hyperparameters ----
        hparams = {
            'batch_size': batch_size,
            'effective_batch_size': batch_size,
            'learning_rate': learning_rate,
            'warmup_ratio': warmup_ratio,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'image_size': str(self.image_size),
            'embedding_dim': self.embedding_dim,
            'temperature': self.temperature,
            'use_amp': use_amp_actual,
            'use_hard_negatives': use_hard_negatives,
            'crem_radius': crem_radius,
            'crem_db_path': crem_db_path or _CREM_DB_PATH,
            'use_random_renderer': use_random_renderer,
            'default_drawing_style': default_drawing_style,
            'pretrained': self.pretrained,
            'num_workers': num_workers,
            'seed': seed,
            'world_size': 1,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_steps': total_steps,
            'device': str(device),
        }
        hparams_path = os.path.join(log_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        hparams_md = '| Param | Value |\n|---|---|\n'
        for k, v in hparams.items():
            hparams_md += f'| {k} | {v} |\n'
        writer.add_text('Hyperparameters', hparams_md, global_step=0)

        # ---- Training-loop state ----
        early_stopping_enabled = early_stopping_patience is not None
        best_val_loss = float('inf')
        vals_no_improve = 0
        epoch_losses: List[float] = []
        global_step = 0

        # ==================== Epoch Loop ====================
        for epoch in range(num_epochs):

            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            step_count = 0
            steps_in_epoch = len(train_loader)
            train_bar = tqdm(total=steps_in_epoch,
                             desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            train_iter = iter(train_loader)

            for _step in range(steps_in_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                except RuntimeError as e:
                    warnings.warn(f'DataLoader error at epoch {epoch+1} '
                                  f'step {_step}: {e}')
                    del train_iter
                    train_loader = _make_train_loader()
                    train_iter = iter(train_loader)
                    train_bar.update(1)
                    continue

                view1, view2, hard_neg, hn_valid = batch
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                hard_neg = hard_neg.to(device, non_blocking=True)
                hn_mask = hn_valid.to(device, non_blocking=True).bool()

                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        v1_emb, v2_emb, hn_emb = self.model(view1, view2, hard_neg)
                        loss, metrics = self.criterion(v1_emb, v2_emb, hn_emb, hn_mask)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    v1_emb, v2_emb, hn_emb = self.model(view1, view2, hard_neg)
                    loss, metrics = self.criterion(v1_emb, v2_emb, hn_emb, hn_mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item()
                step_count += 1
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos': f'{metrics["avg_pos_cosine_sim"]:.3f}',
                    'hn': f'{metrics["hard_neg_rate"]:.0%}',
                })
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                writer.add_scalar('Train/pos_cosine_sim', metrics['avg_pos_cosine_sim'], global_step)
                writer.add_scalar('Train/hard_neg_cosine_sim', metrics['avg_hard_neg_cosine_sim'], global_step)
                writer.add_scalar('Train/contrastive_acc', metrics['contrastive_accuracy'], global_step)
                writer.add_scalar('Train/hard_neg_rate', metrics['hard_neg_rate'], global_step)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
                global_step += 1
                train_bar.update(1)
            train_bar.close()

            train_loss = epoch_loss / max(1, step_count)
            epoch_losses.append(train_loss)

            # --- Validate ---
            self.model.eval()
            val_loss_sum = 0.0
            val_pos_sum = 0.0
            val_hn_sim_sum = 0.0
            val_acc_sum = 0.0
            val_hn_cnt = 0
            count = 0

            with torch.no_grad():
                for view1, view2, hard_neg, hn_valid in val_loader:
                    view1 = view1.to(device, non_blocking=True)
                    view2 = view2.to(device, non_blocking=True)
                    hard_neg = hard_neg.to(device, non_blocking=True)
                    hn_mask = hn_valid.to(device, non_blocking=True).bool()
                    with torch.amp.autocast('cuda', enabled=use_amp_actual):
                        v1, v2, hn = self.model(view1, view2, hard_neg)
                    vloss, vmetrics = self.criterion(v1, v2, hn, hn_mask)

                    bsz = view1.size(0)
                    val_loss_sum += vloss.item() * bsz
                    val_pos_sum += vmetrics['avg_pos_cosine_sim'] * bsz
                    val_acc_sum += vmetrics['contrastive_accuracy'] * bsz
                    n_valid_hn = hn_mask.sum().item()
                    val_hn_sim_sum += vmetrics['avg_hard_neg_cosine_sim'] * n_valid_hn
                    val_hn_cnt += n_valid_hn
                    count += bsz

            val_loss = val_loss_sum / max(1, count)
            avg_pos_sim = val_pos_sum / max(1, count)
            avg_hn_sim = val_hn_sim_sum / max(1, val_hn_cnt)
            val_acc = val_acc_sum / max(1, count)
            val_hn_rate = val_hn_cnt / max(1, count)

            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}: train={train_loss:.4f} | val={val_loss:.4f} | '
                  f'pos_sim={avg_pos_sim:.3f} | hn_sim={avg_hn_sim:.3f} | '
                  f'acc={val_acc:.3f} | hn_rate={val_hn_rate:.1%} | '
                  f'lr={current_lr:.2e}')
            writer.add_scalar('Val/loss', val_loss, global_step)
            writer.add_scalar('Val/pos_cosine_sim', avg_pos_sim, global_step)
            writer.add_scalar('Val/hard_neg_cosine_sim', avg_hn_sim, global_step)
            writer.add_scalar('Val/contrastive_acc', val_acc, global_step)
            writer.add_scalar('Val/hard_neg_rate', val_hn_rate, global_step)

            # --- Checkpointing ---
            if save_path:
                cached_sd = self.model.state_dict()
                torch.save(cached_sd,
                           os.path.join(save_path, f'epoch_{epoch+1}.pth'))
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': cached_sd,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'vals_no_improve': vals_no_improve,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_path, 'training_state.pth'))

            # --- Best-model tracking ---
            if val_loss < best_val_loss:
                vals_no_improve = 0
                best_val_loss = val_loss
                if save_path:
                    torch.save(cached_sd,
                               os.path.join(save_path, 'best.pth'))
                    print(f'  New best model (val_loss={best_val_loss:.4f}).')
            else:
                if early_stopping_enabled:
                    vals_no_improve += 1
                    print(f'  No improvement ({vals_no_improve}/{early_stopping_patience}).')

            if (early_stopping_enabled
                    and vals_no_improve >= early_stopping_patience):
                print(f'Early stopping at epoch {epoch+1}.')
                break

        if save_path:
            torch.save(self.model.state_dict(),
                       os.path.join(save_path, 'final.pth'))
        writer.close()
        return epoch_losses

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for inference (matches training pipeline)."""
        img, _ = self._pad_to_square(img)
        result = self._inference_transforms(image=img)
        return result['image']

    def predict(
        self,
        img1,
        img2,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """Predict whether two images represent the same compound.

        Args:
            img1: First image (RGB numpy array or preprocessed torch.Tensor).
            img2: Second image (RGB numpy array or preprocessed torch.Tensor).
            threshold: Distance threshold for same/different decision.

        Returns:
            (is_same, distance) where distance is L2 embedding distance.
        """
        self.model.eval()

        if isinstance(img1, torch.Tensor):
            img1_tensor = img1.unsqueeze(0).to(self.device) if img1.dim() == 3 else img1.to(self.device)
        else:
            img1_tensor = self._preprocess_image(img1).unsqueeze(0).to(self.device)

        if isinstance(img2, torch.Tensor):
            img2_tensor = img2.unsqueeze(0).to(self.device) if img2.dim() == 3 else img2.to(self.device)
        else:
            img2_tensor = self._preprocess_image(img2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb1 = self.model.forward_once(img1_tensor)
            emb2 = self.model.forward_once(img2_tensor)

        distance = F.pairwise_distance(emb1, emb2, p=2).item()
        is_same = distance < threshold

        return is_same, distance

    def get_embeddings(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """Get embeddings for a list of images using all available GPUs.

        Args:
            images: List of RGB numpy arrays.
            batch_size: Batch size for processing.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        self.model.eval()

        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            embed_model = nn.DataParallel(
                _EmbeddingWrapper(self.model)
            ).to(self.device)
        else:
            embed_model = _EmbeddingWrapper(self.model).to(self.device)
        embed_model.eval()

        tensors = torch.stack([self._preprocess_image(img) for img in images])

        all_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = tensors[i:i + batch_size].to(self.device)
                emb = embed_model(batch)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def predict_batch(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> List[Tuple[bool, float]]:
        """Batch prediction for multiple image pairs using all available GPUs.

        Args:
            image_pairs: List of (img1, img2) tuples (RGB numpy arrays).
            threshold: Distance threshold for same/different decision.
            batch_size: Processing batch size.

        Returns:
            List of (is_same, distance) tuples.
        """
        img1_list = [pair[0] for pair in image_pairs]
        img2_list = [pair[1] for pair in image_pairs]

        emb1 = self.get_embeddings(img1_list, batch_size=batch_size)
        emb2 = self.get_embeddings(img2_list, batch_size=batch_size)

        emb1_t = torch.from_numpy(emb1)
        emb2_t = torch.from_numpy(emb2)
        distances = F.pairwise_distance(emb1_t, emb2_t, p=2).numpy()

        return [(float(d) < threshold, float(d)) for d in distances]

    def load_model(self, path: str):
        """Load trained model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
