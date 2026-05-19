"""
COMO Bond Predictor
===================

Pairwise MLP that classifies bond types between atom pairs.
Takes hidden states from the SequenceDecoder at atom positions and produces
per-edge logits for N bond classes.

Public API:
  - ``BondPredictor``                         — pairwise bond-type classifier
  - ``_symmetrize_edge_predictions``          — symmetrize single-sample edges
  - ``_symmetrize_edge_predictions_batched``  — batched symmetrization
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BondPredictor(nn.Module):
    """
    Pairwise MLP bond-type predictor.

    Takes hidden states from the SequenceDecoder at atom positions,
    forms all ordered (i, j) pairs via concatenation [h_i, h_j], and
    classifies into *n_bond_classes* bond types.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_bond_classes: int = 7,
    ):
        """
        Args:
            d_model: Hidden dimension from SequenceDecoder.
            n_bond_classes: Number of bond types (0–6: none, single, double,
                triple, aromatic, wedge-solid, wedge-dash).
        """
        super().__init__()
        self.d_model = d_model
        self.n_bond_classes = n_bond_classes
        
        # MLP for pairwise bond prediction
        # Input: concatenated pair features [2*d_model]
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_bond_classes)
        )
        
        # Cache for diagonal mask to avoid recreation
        self._diag_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    
    def _get_diag_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """Get cached diagonal mask or create new one."""
        cache_key = (N, device)
        if cache_key not in self._diag_mask_cache:
            self._diag_mask_cache[cache_key] = torch.eye(N, device=device, dtype=torch.bool)
        return self._diag_mask_cache[cache_key]
    
    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
        atom_indices: torch.Tensor,
        atom_mask: torch.Tensor,
        N: int,
        T: int,
        dim: int,
    ) -> torch.Tensor:
        """Process a chunk of the batch through pair MLP + masking."""
        B_chunk = hidden_states.size(0)
        device = hidden_states.device

        expanded_indices = atom_indices.unsqueeze(-1).expand(B_chunk, N, dim).contiguous()
        atom_hidden = torch.gather(hidden_states, 1, expanded_indices)  # [B_chunk, N, d]

        atom_i = atom_hidden.unsqueeze(2)  # [B_chunk, N, 1, d]
        atom_j = atom_hidden.unsqueeze(1)  # [B_chunk, 1, N, d]
        pair_features = torch.cat(
            [atom_i.expand(-1, -1, N, -1), atom_j.expand(-1, N, -1, -1)], dim=3
        )  # [B_chunk, N, N, 2d]

        edge_logits = self.mlp(pair_features).permute(0, 3, 1, 2)  # [B_chunk, n_bond_classes, N, N]

        valid_pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # [B_chunk, N, N]
        diag_mask = self._get_diag_mask(N, device)
        valid_pair_mask = valid_pair_mask & ~diag_mask.unsqueeze(0)
        edge_logits = edge_logits.masked_fill(~valid_pair_mask.unsqueeze(1), -1e4)

        return edge_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        atom_indices: torch.Tensor,
        atom_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive chunked pairwise computation.

        When the pairwise tensor [B, N, N, 2d] would be very large, the batch
        is split into smaller chunks along dim-0 to cap peak GPU memory.  This
        does NOT change the numerics — gradient flows through ``torch.cat``.
        
        Args:
            hidden_states: [B, T, d_model]
            atom_indices: [B, N] - positions of atom embeddings
            atom_mask: [B, N] - True for valid atoms
        
        Returns:
            edge_logits: [B, n_bond_classes, N, N]
        """
        B, T, dim = hidden_states.shape
        N = atom_indices.size(1)
        device = hidden_states.device
        
        # Ensure tensors are contiguous and on correct device
        atom_indices = atom_indices.to(device).contiguous()
        atom_mask = atom_mask.to(device).contiguous()
        
        # Clamp indices defensively
        atom_indices = atom_indices.clamp(0, T - 1)

        # Adaptive chunking: cap peak pair-tensor memory per chunk.
        # Elements per sample ≈ N*N*2d (concat) + N*N*d (linear hidden) = N*N*3d.
        elements_per_sample = N * N * dim * 3  # conservative estimate
        # 256M elements ≈ 512 MB fp16 — keeps worst-case (N=100) to ~5 chunks
        max_elements = 256 * 1024 * 1024
        chunk_size = max(1, max_elements // max(elements_per_sample, 1))
        chunk_size = min(chunk_size, B)

        if chunk_size >= B:
            # Fast path: entire batch fits comfortably
            return self._forward_chunk(hidden_states, atom_indices, atom_mask, N, T, dim)

        # Chunked path: iterate over sub-batches
        edge_logits_list = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_logits = self._forward_chunk(
                hidden_states[start:end],
                atom_indices[start:end],
                atom_mask[start:end],
                N, T, dim,
            )
            edge_logits_list.append(chunk_logits)
        return torch.cat(edge_logits_list, dim=0)


# ======================== Symmetrize Edge Predictions ========================

def _symmetrize_edge_predictions(edge_logits: torch.Tensor) -> np.ndarray:
    r"""Symmetrize bond-type predictions via bidirectional probability averaging.

    Follows MolScribe's ``get_edge_prediction`` logic:
      - Bond types 0–4 (no-bond, single, double, triple, aromatic) are symmetric:
        $p_{ij}^k \leftarrow (p_{ij}^k + p_{ji}^k) / 2$
      - Bond types 5 & 6 (solid-wedge / dash-wedge) are directional, so they
        are cross-averaged:
        $p_{ij}^5 \leftarrow (p_{ij}^5 + p_{ji}^6) / 2$  (and vice-versa)

    Args:
        edge_logits: [n_bond_classes, N, N] raw logits from BondPredictor
                     (single sample, no batch dim).
    Returns:
        edge_preds: [N, N] numpy int array of predicted bond types.
    """
    # Convert logits to probabilities: [N, N, n_bond_classes]
    prob = F.softmax(edge_logits.permute(1, 2, 0).float(), dim=2)

    # Symmetric bond types (0–4): average p[i,j] and p[j,i]
    sym = prob[:, :, :5]
    prob[:, :, :5] = (sym + sym.transpose(0, 1)) / 2

    # Directional wedge bonds (5 & 6): cross-average
    old_5 = prob[:, :, 5].clone()
    old_6 = prob[:, :, 6].clone()
    prob[:, :, 5] = (old_5 + old_6.T) / 2
    prob[:, :, 6] = (old_6 + old_5.T) / 2

    return prob.argmax(dim=2).cpu().numpy()


def _symmetrize_edge_predictions_batched(edge_logits: torch.Tensor) -> np.ndarray:
    r"""Batched symmetrization: one GPU op + one CPU transfer.

    Same logic as :func:`_symmetrize_edge_predictions` but operates on a
    full ``[B, 7, N, N]`` batch.  Replaces B individual per-sample calls
    (each triggering a GPU→CPU sync) with a single batched operation.

    Args:
        edge_logits: ``[B, n_bond_classes, N, N]`` raw logits.
    Returns:
        edge_preds: ``[B, N, N]`` numpy int array of predicted bond types.
    """
    # [B, N, N, 7]
    prob = F.softmax(edge_logits.permute(0, 2, 3, 1).float(), dim=3)

    # Symmetric bond types (0–4)
    sym = prob[..., :5]
    prob[..., :5] = (sym + sym.transpose(1, 2)) / 2

    # Directional wedge bonds (5 & 6): cross-average
    old_5 = prob[..., 5].clone()
    old_6 = prob[..., 6].clone()
    prob[..., 5] = (old_5 + old_6.transpose(1, 2)) / 2
    prob[..., 6] = (old_6 + old_5.transpose(1, 2)) / 2

    return prob.argmax(dim=3).cpu().numpy()
