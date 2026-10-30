"""
COMO Training — Reward Functions & MRT Loss
============================================

Minimum Risk Training (MRT) loss computation with composite reward
signals (validity, similarity, exact match) and visual reward encoding.

Also includes:
  - Levenshtein similarity for token-level edit-distance reward
  - RDKit / drawing_engine SMILES rendering for visual rewards
  - External visual reward encoder loading (MolDiscriminator SiameseNetwork)

Private module — not included in the pip-publishable ``como`` distribution.
"""

import math
import os
from typing import Dict, List, Optional, Tuple
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from func_timeout import FunctionTimedOut

from como.vocab import ComoVocab
from como.model import ComoModel
from como.utils import (
    compute_tanimoto_similarity,
    extract_atom_indices_from_tokens,
    truncate_sequences,
)
from como.inference import _result_to_smiles_postprocess, canonicalize_smiles
from training.perf_timers import perf_phase
from como.bond_predictor import _symmetrize_edge_predictions_batched

from . import drawing_engine as _drawing_engine
from como.chemistry import canonicalize_smiles as _canonicalize_smiles


# Re-export canonicalize_smiles for convenience
canonicalize_smiles = _canonicalize_smiles


# ======================== Rendering Helpers ========================

def _blank_image(*args, **kwargs):
    return _drawing_engine._blank_image(*args, **kwargs)


def _generate_image_from_smiles(smiles: str, **kwargs) -> Tuple[np.ndarray, bool]:
    """Render SMILES via drawing_engine (Indigo backend)."""
    return _drawing_engine.generate_image_from_smiles(smiles, **kwargs)


# ======================== Levenshtein Similarity ========================

def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity: 1 − edit_distance / max(|s1|, |s2|).

    Returns a value in [0, 1] where 1 means identical strings.
    Provides token-level feedback (each correct/incorrect character matters),
    unlike Tanimoto which is insensitive to small edits once fingerprints match.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return 1.0 - prev[-1] / max_len


# ======================== SMILES Rendering for Visual Reward ========================

def _fast_render_smiles(smiles: str) -> Tuple[np.ndarray, bool]:
    """Render SMILES to a numpy image using RDKit Draw (~5 ms/call).

    Fast and lightweight, suitable for batch rendering during RL training.
    Returns (H×W×3 uint8, success_bool).
    """
    try:
        from rdkit.Chem import Draw   # rdMolDraw2D needs libXrender; envs without it degrade to blank
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


def _drawing_engine_render_smiles(smiles: str) -> Tuple[np.ndarray, bool]:
    """Render SMILES with drawing_engine for discriminator-style visuals.

    Uses deterministic settings (no molecule augmentation, default style)
    to keep reward variance low while staying closer to the
    MolDiscriminator training domain.
    """
    if not isinstance(smiles, str) or len(smiles) == 0:
        return _blank_image(), False

    try:
        img, _, _, success, _, _ = _generate_image_from_smiles(
            smiles,
            default_drawing_style=True,
            mol_augment=False,
            debug=False,
        )
        if not success:
            return _blank_image(), False
        return img, True
    except (FunctionTimedOut, Exception):
        return _blank_image(), False


def _render_smiles_for_visual_reward(smiles: str, renderer_mode: str) -> Tuple[np.ndarray, bool]:
    """Render SMILES for visual reward with a selectable backend."""
    if renderer_mode == 'rdkit':
        return _fast_render_smiles(smiles)
    if renderer_mode == 'drawing_engine':
        return _drawing_engine_render_smiles(smiles)
    raise ValueError(
        f"Unknown visual_reward_renderer='{renderer_mode}'. "
        "Choose from 'rdkit' or 'drawing_engine'."
    )


# ======================== Visual Reward Encoder Loading ========================

def load_visual_reward_encoder(
    checkpoint_path: str,
    device: torch.device,
    image_size: Tuple[int, int] = (384, 384),
    embedding_dim: int = 128,
) -> Tuple[nn.Module, A.Compose, object]:
    """Load a frozen SiameseNetwork (MolDiscriminator) as visual reward encoder.

    Args:
        checkpoint_path: Path to the SiameseNetwork checkpoint (.pth).
        device: Target device.
        image_size: Image size for preprocessing.
        embedding_dim: Embedding dimension of the SiameseNetwork.

    Returns:
        (encoder, transforms, pad_to_square) — frozen encoder module and
        its preprocessing pipeline.
    """
    from .mol_discriminator import SiameseNetwork, PadToSquare as DiscPadToSquare

    encoder = SiameseNetwork(embedding_dim=embedding_dim, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)

    pad_to_square = DiscPadToSquare(fill=255)
    from .mol_discriminator import AdaptiveResize as DiscAdaptiveResize
    transforms = A.Compose([
        DiscAdaptiveResize(height=image_size[0], width=image_size[1]),
        A.ToGray(num_output_channels=3),
        A.Normalize(),
        ToTensorV2(),
    ])
    return encoder, transforms, pad_to_square


def _encode_with_external_encoder(
    images_np: List[np.ndarray],
    encoder: nn.Module,
    transforms: A.Compose,
    pad_to_square,
    device: torch.device,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Encode a list of numpy images using an external SiameseNetwork encoder."""
    tensors = []
    for img in images_np:
        img_sq, _ = pad_to_square(img)
        tensors.append(transforms(image=img_sq)['image'])
    batch = torch.stack(tensors).to(device)

    _amp_enabled = device.type == 'cuda'
    feat_chunks = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=_amp_enabled):
        for i in range(0, len(tensors), chunk_size):
            chunk = batch[i:i + chunk_size]
            feat_chunks.append(encoder.forward_once(chunk).float())
    feats = torch.cat(feat_chunks, dim=0)
    del batch
    return feats


# ======================== Minimum Risk Training (MRT) Loss ========================

def compute_mrt_loss(
    model: ComoModel,
    img_features: torch.Tensor,
    gt_smiles_list: List[str],
    vocab: ComoVocab,
    device: torch.device,
    max_len: int = 500,
    temperature: float = 1.0,
    n_samples: int = 8,
    mrt_alpha: float = 1.0,
    reward_validity_weight: float = 0.1,
    reward_similarity_weight: float = 0.5,
    reward_exact_match_weight: float = 0.4,
    reward_mode: str = 'tanimoto',
    frozen_encoder: Optional[nn.Module] = None,
    frozen_pos_enc: Optional[nn.Module] = None,
    visual_reward_encoder: Optional[nn.Module] = None,
    visual_reward_transforms: Optional[A.Compose] = None,
    visual_reward_pad_to_square: Optional[object] = None,
    visual_reward_renderer: str = 'rdkit',
    use_amp: bool = False,
    gt_render_cache: Optional[Dict] = None,
    sampling_method: str = 'multinomial',
    adaptive_n: bool = False,
    adaptive_n_gamma: float = 1.5,
    adaptive_n_min: int = 8,
    rollout_fast: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Minimum Risk Training (MRT) loss for molecule image recognition.

    Minimizes expected cost (1 − reward) under the model's sampling
    distribution, approximated by N candidates per image.

    Per-image loss::

        L_MRT = Σ_i  w_i · (1 − R_i)
        w_i = softmax(α · log p(y_i | x))_i

    Pipeline (three phases):
      1. **Sample** — generate N candidate SMILES per image via batched
         multinomial or beam-search decoding (no grad, KV-cached).
      2. **Teacher-forced forward + postprocess + reward** — single
         differentiable decoder pass yields both (a) log p(y_i | x)
         for MRT weights and (b) hidden states reused by the bond
         predictor (detached) for graph-aware postprocessing.
      3. **MRT loss** — softmax-weighted cost, mean over batch.

    Args:
        model: ComoModel (unwrapped, not DDP).
        img_features: ``[B, d_model, H, W]`` encoded features with
            computation graph attached.
        gt_smiles_list: Ground-truth SMILES strings, length B.
        vocab: ComoVocab for tokenisation / detokenisation.
        device: Compute device (must be CUDA when ``use_amp=True``).
        max_len: Max autoregressive decode length.
        temperature: Sampling temperature (>1 → more exploration).
        n_samples: Candidates per image (N ≥ 2).
        mrt_alpha: Sharpness of the softmax over candidate log-probs.
        reward_validity_weight: Weight for valid-SMILES indicator.
        reward_similarity_weight: Weight for main similarity signal.
        reward_exact_match_weight: Weight for exact-match indicator.
        reward_mode: ``'tanimoto'``, ``'edit_distance'``, or ``'visual'``.
        frozen_encoder: Frozen encoder for stable visual reward.
        frozen_pos_enc: Frozen positional encoding.
        visual_reward_encoder: External SiameseNetwork encoder.
        visual_reward_transforms: Preprocessing pipeline for external encoder.
        visual_reward_pad_to_square: Pad-to-square transform.
        visual_reward_renderer: ``'rdkit'`` or ``'drawing_engine'``.
        use_amp: Run teacher-forced forward under AMP fp16.
        gt_render_cache: Optional dict cache for GT SMILES renders.
        sampling_method: ``'multinomial'`` or ``'beam_search'``.
        adaptive_n: Enable adaptive candidate count via ESS.
        adaptive_n_gamma: Scaling factor for adaptive N.
        adaptive_n_min: Minimum candidate count when adaptive_n is True.

    Returns:
        loss: Scalar MRT loss with gradient graph.
        info: Dict of diagnostic metrics.
    """
    B = img_features.size(0)
    total_seqs = B * n_samples

    if adaptive_n and sampling_method != 'beam_search':
        raise ValueError("adaptive_n=True requires sampling_method='beam_search'.")
    if adaptive_n and adaptive_n_min > n_samples:
        raise ValueError(f"adaptive_n_min={adaptive_n_min} > n_samples={n_samples}.")

    img_features_expanded = img_features.repeat_interleave(n_samples, dim=0)

    # ===== Phase 1: Sample N candidate sequences (no grad) =====
    _decode_fast = rollout_fast and device.type == 'cuda'
    with perf_phase('mrt_decode'), torch.no_grad():
        if sampling_method == 'beam_search':
            sampled_seqs_list = []
            for b in range(B):
                feat_b = img_features[b:b+1]
                top_n = model._beam_search_top_n(feat_b, n_samples, max_len, device)
                sampled_seqs_list.extend(top_n)
        elif _decode_fast:
            # fp16 KV pool + autocast: halves the per-step KV bandwidth
            # (the decode loop is memory-bound); sampled-string changes on
            # rounding are within the sampling distribution's own noise
            with torch.amp.autocast('cuda'):
                sampled_seqs_list = model._sample_decode_batch(
                    img_features_expanded, max_len, device, temperature,
                    kv_dtype=torch.float16,
                )
        else:
            sampled_seqs_list = model._sample_decode_batch(
                img_features_expanded, max_len, device, temperature
            )

    # ===== Phase 2: Teacher-forced forward + bond prediction + reward =====
    with perf_phase('mrt_tfwd'):
        max_seq_len = max(len(s) for s in sampled_seqs_list)
        padded_seqs = torch.full((total_seqs, max_seq_len), vocab.pad_idx,
                                 dtype=torch.long, device=device)
        for b, seq in enumerate(sampled_seqs_list):
            padded_seqs[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        tf_input = padded_seqs[:, :-1]
        tf_target = padded_seqs[:, 1:]

        _amp_enabled = use_amp and device.type == 'cuda'
        with torch.amp.autocast('cuda', enabled=_amp_enabled):
            hidden_states, tf_logits = model.sequence_decoder(img_features_expanded, tf_input)

        tf_logits = tf_logits.float()
        log_probs = F.log_softmax(tf_logits, dim=-1)
        token_log_probs = log_probs.gather(2, tf_target.unsqueeze(-1)).squeeze(-1)
        pad_mask = (tf_target != vocab.pad_idx).float()
        seq_log_probs = (token_log_probs * pad_mask).sum(dim=1)

        with torch.no_grad():
            atom_indices, atom_counts = extract_atom_indices_from_tokens(tf_input, vocab)
            max_atoms = int(atom_counts.max().item())
            edge_logits_all = None
            if max_atoms > 0:
                atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
                edge_logits_all = model.bond_predictor(hidden_states.detach(), atom_indices, atom_mask)

        if edge_logits_all is not None:
            edge_preds_all = _symmetrize_edge_predictions_batched(edge_logits_all)
        else:
            edge_preds_all = None

    with perf_phase('mrt_pp'):
        # Serial SMILES postprocessing (the same _result_to_smiles_postprocess
        # the inference path uses). atom_counts + edge matrices are pulled to
        # CPU ONCE per MRT step, replacing 512 per-seq GPU->CPU syncs with two.
        # NB the forkserver PROCESS pool (COMO_FAST_MRT, 2026-10-23 bench g)
        # regressed 4% -> 40%: under torchrun each pool child re-imports
        # __main__ (heavy torch/entry chain) - keep threads-free serial here
        # until the pool pattern is rebuilt around an explicit worker module.
        _atom_counts_cpu = atom_counts.tolist()
        edge_np_all = None
        if edge_preds_all is not None:
            edge_np_all = (edge_preds_all.detach().cpu().numpy()
                           if isinstance(edge_preds_all, torch.Tensor)
                           else np.asarray(edge_preds_all))
        pp_smiles_list: List[Optional[str]] = []
        for b in range(total_seqs):
            result = vocab.sequence_to_smiles(sampled_seqs_list[b])
            if edge_preds_all is not None and _atom_counts_cpu[b] > 0:
                edge_preds = edge_np_all[b]
            else:
                edge_preds = np.zeros((0, 0), dtype=np.int64)
            result_dict = {
                'decode_smiles': result.get('smiles', ''),
                'symbols': result.get('symbols', []),
                'coords': result.get('coords', []),
                'bond_mat': edge_preds,
                'success': len(result.get('smiles', '')) > 0,
            }
            # GT below is canonicalized with ignore_cistrans=True; predictions
            # must match for exact-match/similarity rewards
            pp_smiles_list.append(_result_to_smiles_postprocess(result_dict, ignore_cistrans=True))

    with perf_phase('mrt_prep'):
        gt_canonical = [canonicalize_smiles(s, ignore_cistrans=True)[0] for s in gt_smiles_list]
        gt_canonical_expanded = [c for c in gt_canonical for _ in range(n_samples)]

        is_valid_list = [smi is not None for smi in pp_smiles_list]
        is_exact_list = [smi is not None and smi == gt_canonical_expanded[b]
                         for b, smi in enumerate(pp_smiles_list)]
        n_valid_smiles = sum(is_valid_list)
        n_exact = sum(is_exact_list)

    rewards = torch.zeros(total_seqs, device=device)
    similarity_sum = 0.0
    extra_info: Dict = {}

    if reward_mode == 'tanimoto':
        for b in range(total_seqs):
            tanimoto = compute_tanimoto_similarity(
                pp_smiles_list[b] or '', gt_canonical_expanded[b] or '')
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * tanimoto
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += tanimoto
        extra_info['mean_tanimoto'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'edit_distance':
        with perf_phase('mrt_lev'):
            for b in range(total_seqs):
                edit_sim = _levenshtein_similarity(
                    pp_smiles_list[b] or '', gt_canonical_expanded[b] or '')
                r = (reward_validity_weight * float(is_valid_list[b])
                     + reward_similarity_weight * edit_sim
                     + reward_exact_match_weight * float(is_exact_list[b]))
                rewards[b] = r
                similarity_sum += edit_sim
        extra_info['mean_edit_similarity'] = similarity_sum / max(total_seqs, 1)

    elif reward_mode == 'visual':
        _use_external = (visual_reward_encoder is not None)
        _encode_chunk = 32

        _render_fn = partial(_render_smiles_for_visual_reward, renderer_mode=visual_reward_renderer)
        if visual_reward_renderer == 'drawing_engine':
            _n_render_workers = min(2, os.cpu_count() or 2)
        else:
            _n_render_workers = min(8, os.cpu_count() or 4)

        gt_canonical_for_render = [canonicalize_smiles(s, ignore_cistrans=True)[0] or '' for s in gt_smiles_list]
        _gt_cache = gt_render_cache
        _GT_CACHE_MAX = 10000
        if _gt_cache is not None:
            gt_needed_set = set(gt_canonical_for_render)
            gt_to_render = [s for s in gt_canonical_for_render if s not in _gt_cache]
            if gt_to_render:
                with ThreadPoolExecutor(max_workers=_n_render_workers) as pool:
                    gt_new_results = list(pool.map(_render_fn, gt_to_render))
                n_to_evict = max(0, len(_gt_cache) + len(gt_to_render) - _GT_CACHE_MAX)
                if n_to_evict > 0:
                    evicted = 0
                    for key in list(_gt_cache.keys()):
                        if evicted >= n_to_evict:
                            break
                        if key not in gt_needed_set:
                            del _gt_cache[key]
                            evicted += 1
                for s, r in zip(gt_to_render, gt_new_results):
                    _gt_cache[s] = r
            gt_unique_rendered = [_gt_cache[s] for s in gt_canonical_for_render]
        else:
            with ThreadPoolExecutor(max_workers=_n_render_workers) as pool:
                gt_unique_rendered = list(pool.map(_render_fn, gt_canonical_for_render))

        _blank = _blank_image()
        pred_smiles_for_render = [s or '' for s in pp_smiles_list]
        unique_pred_smiles = list({s for s in pred_smiles_for_render if s != ''})
        if unique_pred_smiles:
            with ThreadPoolExecutor(max_workers=_n_render_workers) as pool:
                unique_pred_results = list(pool.map(_render_fn, unique_pred_smiles))
            _pred_render_map = dict(zip(unique_pred_smiles, unique_pred_results))
        else:
            _pred_render_map = {}
        pred_rendered_np = []
        pred_render_ok = []
        for s in pred_smiles_for_render:
            if s == '' or s not in _pred_render_map:
                pred_rendered_np.append(_blank)
                pred_render_ok.append(False)
            else:
                r = _pred_render_map[s]
                pred_rendered_np.append(r[0])
                pred_render_ok.append(bool(r[1]))
        render_success_count = sum(pred_render_ok)

        if _use_external:
            gt_unique_np = [r[0] for r in gt_unique_rendered]
            gt_feat_unique = _encode_with_external_encoder(
                gt_unique_np, visual_reward_encoder,
                visual_reward_transforms, visual_reward_pad_to_square,
                device, _encode_chunk)
            gt_feat = gt_feat_unique.repeat_interleave(n_samples, dim=0)
            del gt_feat_unique

            pred_feat = _encode_with_external_encoder(
                pred_rendered_np, visual_reward_encoder,
                visual_reward_transforms, visual_reward_pad_to_square,
                device, _encode_chunk)
        else:
            _ref_encoder = frozen_encoder if frozen_encoder is not None else model.image_encoder
            _ref_pos_enc = frozen_pos_enc if frozen_pos_enc is not None else model.pos_enc_2d

            gt_unique_tensors = [model.inference_transforms(image=r[0])['image'] for r in gt_unique_rendered]
            gt_unique_batch = torch.stack(gt_unique_tensors).to(device)
            gt_feat_chunks = []
            _amp_reward = use_amp and device.type == 'cuda'
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=_amp_reward):
                for i in range(0, B, _encode_chunk):
                    chunk = gt_unique_batch[i:i + _encode_chunk]
                    feat = _ref_encoder(chunk)
                    feat = _ref_pos_enc(feat)
                    feat = feat.float().mean(dim=[2, 3])
                    gt_feat_chunks.append(feat)
            gt_feat_unique = torch.cat(gt_feat_chunks, dim=0)
            gt_feat = gt_feat_unique.repeat_interleave(n_samples, dim=0)
            del gt_unique_batch, gt_feat_unique

            pred_rendered_tensors = [model.inference_transforms(image=img)['image'] for img in pred_rendered_np]
            pred_rendered_batch = torch.stack(pred_rendered_tensors).to(device)
            pred_feat_chunks = []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=_amp_reward):
                for i in range(0, total_seqs, _encode_chunk):
                    chunk = pred_rendered_batch[i:i + _encode_chunk]
                    feat = _ref_encoder(chunk)
                    feat = _ref_pos_enc(feat)
                    feat = feat.float().mean(dim=[2, 3])
                    pred_feat_chunks.append(feat)
            pred_feat = torch.cat(pred_feat_chunks, dim=0)
            del pred_rendered_batch

        cos_sims = F.cosine_similarity(gt_feat, pred_feat, dim=1).clamp(min=0.0, max=1.0)
        for b in range(total_seqs):
            visual_sim = cos_sims[b].item() if (is_valid_list[b] and pred_render_ok[b]) else 0.0
            r = (reward_validity_weight * float(is_valid_list[b])
                 + reward_similarity_weight * visual_sim
                 + reward_exact_match_weight * float(is_exact_list[b]))
            rewards[b] = r
            similarity_sum += visual_sim
        extra_info['mean_visual_similarity'] = similarity_sum / max(total_seqs, 1)
        extra_info['render_success'] = render_success_count
        del pred_feat, gt_feat

    else:
        raise ValueError(f"Unknown reward_mode='{reward_mode}'.")

    costs = (1.0 - rewards).detach()

    # ===== Phase 3: MRT loss =====
    seq_log_probs_grouped = seq_log_probs.view(B, n_samples)
    costs_grouped = costs.view(B, n_samples)

    Q = F.softmax(mrt_alpha * seq_log_probs_grouped, dim=1)

    adaptive_n_info: Dict = {}
    if adaptive_n:
        ess = 1.0 / (Q ** 2).sum(dim=1)
        n_x = (adaptive_n_gamma * ess).round().clamp(min=adaptive_n_min, max=n_samples).long()
        col_idx = torch.arange(n_samples, device=device).unsqueeze(0)
        keep_mask = (col_idx < n_x.unsqueeze(1)).float()
        Q_masked = Q * keep_mask
        weights = Q_masked / Q_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adaptive_n_info = {
            'adaptive_n_mean': n_x.float().mean().item(),
            'adaptive_n_min': n_x.min().item(),
            'adaptive_n_max': n_x.max().item(),
            'ess_mean': ess.mean().item(),
        }
    else:
        weights = Q

    mrt_loss = (weights * costs_grouped).sum(dim=1).mean()

    info = {
        'mean_reward': rewards.mean().item(),
        'mean_cost': costs.mean().item(),
        'mean_log_prob': seq_log_probs.mean().item(),
        'sampled_smiles': [s or '' for s in pp_smiles_list[:B]],
        'mean_similarity': similarity_sum / max(total_seqs, 1),
        'exact_matches': n_exact,
        'valid_smiles_count': n_valid_smiles,
        'reward_mode': reward_mode,
    }
    info.update(extra_info)
    info.update(adaptive_n_info)

    return mrt_loss, info
