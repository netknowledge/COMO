"""
COMO Utilities
==============

Shared utility functions used across the COMO codebase:
  - Tanimoto similarity computation
  - SMILES atom-mapping removal
  - Atom index extraction from chartok_coords sequences
  - Sequence truncation
  - Coordinate normalization
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


# ======================== Tanimoto Similarity ========================

def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            gen = GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
        except ImportError:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ======================== SMILES Helpers ========================

def remove_atom_mapping(smiles: str) -> str:
    """Remove atom mapping numbers from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


# ======================== Atom Index Extraction ========================

def extract_atom_indices_from_tokens(
    tgt_tokens: torch.Tensor, 
    vocab: 'ComoVocab',  # type: ignore[name-defined]  # noqa: F821
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized extraction of atom positions from chartok_coords sequences.

    In the format [... atom_chars, X_BIN, Y_BIN ...], each atom is anchored
    at its Y_BIN token.  This function returns those positions.

    Args:
        tgt_tokens: [B, T] token-index tensor.
        vocab: ComoVocab instance (used for Y_BIN range).
    Returns:
        atom_indices: [B, max_N] positions of Y_BIN tokens (zero-padded).
        atom_counts:  [B] number of atoms per sequence.
    """
    B, T = tgt_tokens.shape
    device = tgt_tokens.device
    
    # Vectorized: find all Y_BIN tokens at once
    is_y_bin = (tgt_tokens >= vocab.y_bin_start_idx) & (tgt_tokens <= vocab.y_bin_end_idx)
    
    # Count atoms per sequence
    atom_counts = is_y_bin.sum(dim=1)  # [B]
    max_atoms = int(atom_counts.max().item())
    
    if max_atoms == 0:
        return torch.zeros((B, 1), dtype=torch.long, device=device), torch.zeros(B, dtype=torch.long, device=device)
    
    # Pre-allocate output tensor
    atom_indices = torch.zeros((B, max_atoms), dtype=torch.long, device=device)
    
    # Vectorized extraction using nonzero
    y_bin_positions = torch.nonzero(is_y_bin, as_tuple=False)  # [num_y_bins, 2]
    
    if y_bin_positions.numel() > 0:
        batch_indices = y_bin_positions[:, 0]
        positions = y_bin_positions[:, 1]
        
        ones = torch.ones(y_bin_positions.size(0), dtype=torch.long, device=device)
        batch_cumsum = torch.zeros(B + 1, dtype=torch.long, device=device)
        batch_cumsum.scatter_add_(0, batch_indices + 1, ones)
        batch_cumsum = batch_cumsum.cumsum(0)
        
        global_idx = torch.arange(y_bin_positions.size(0), device=device)
        in_batch_idx = global_idx - batch_cumsum[batch_indices]
        
        atom_indices[batch_indices, in_batch_idx] = positions
    
    return atom_indices, atom_counts


# ======================== Sequence Truncation ========================

def truncate_sequences(
    tgt_tokens: torch.Tensor,
    tgt_padding_mask: torch.Tensor,
    atom_indices: torch.Tensor,
    atom_mask: torch.Tensor,
    max_seq_len: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncate sequences to max_seq_len and update atom indices/mask accordingly."""
    if tgt_tokens.size(1) <= max_seq_len:
        effective_max = tgt_tokens.size(1) - 1
        valid_atom_mask = atom_indices < effective_max
        atom_mask = atom_mask & valid_atom_mask
        atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
        return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()
    
    tgt_tokens = tgt_tokens[:, :max_seq_len].contiguous()
    tgt_padding_mask = tgt_padding_mask[:, :max_seq_len].contiguous()
    
    effective_max = max_seq_len - 1
    valid_atom_mask = atom_indices < effective_max
    atom_mask = atom_mask & valid_atom_mask
    atom_indices = torch.where(valid_atom_mask, atom_indices, torch.zeros_like(atom_indices)).contiguous()
    
    return tgt_tokens, tgt_padding_mask, atom_indices, atom_mask.contiguous()


# ======================== Coordinate Normalization ========================

def normalize_nodes(nodes: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """Normalize node coordinates to [0, 1] based on bounding box.

    Args:
        nodes: (N, 2) array of coordinates.
        flip_y: If True, flip Y axis (image convention: y=0 at top).
    """
    x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)
