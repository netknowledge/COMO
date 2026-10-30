"""
COMO Training — Loss Functions
===============================

Training-time loss computation: token cross-entropy with label smoothing
and weighted bond-type cross-entropy.  These functions are used across all
training loops (MLE, RL finetuning, joint training).

Private module — not included in the pip-publishable ``como`` package.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from como.vocab import ComoVocab
from training.ddp_utils import is_main_process


def get_bond_class_weights():
    """Get predefined class weights for bond matrix.

    Weights are designed to balance the heavy class imbalance in bond-type
    prediction: no-bond (class 0) is very common, while triple bonds and
    wedge bonds are rare.
    """
    weights = torch.tensor([
        1.0,   # 0: no bond (very common, lower weight)
        10.0,  # 1: single bond
        10.0,  # 2: double bond
        10.0,  # 3: triple bond (rare)
        10.0,  # 4: aromatic bond
        10.0,  # 5: solid wedge (stereo, rare)
        10.0   # 6: dash wedge (stereo, rare)
    ], dtype=torch.float32)
    
    if is_main_process():
        print(f'Using predefined bond class weights: {weights.numpy()}')
    
    return weights


def compute_losses(
    token_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    tgt_tokens: torch.Tensor,
    bond_matrices_list: List[np.ndarray],
    vocab: ComoVocab,
    bond_class_weights: torch.Tensor,
    token_label_smoothing: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute token-prediction and bond-prediction losses.

    Token loss uses label smoothing for regularization; bond loss uses
    per-class weights to handle extreme class imbalance (most pairs are
    non-bonded).

    Args:
        token_logits: ``[B, T, V]`` decoder output logits.
        edge_logits: ``[B, 7, N, N]`` bond predictor logits.
        tgt_tokens: ``[B, T]`` ground-truth token indices.
        bond_matrices_list: List of ``[N_i, N_i]`` ground-truth bond matrices.
        vocab: ComoVocab instance (for PAD index).
        bond_class_weights: ``[7]`` per-class weights.
        token_label_smoothing: Label smoothing factor for token CE.

    Returns:
        Dict with keys ``token_loss``, ``bond_loss``, ``total_loss``.
    """
    B, T, vocab_size = token_logits.shape
    device = token_logits.device
    
    # ===== 1. Token loss (cross-entropy with label smoothing) =====
    token_logits_flat = token_logits.reshape(-1, vocab_size)
    tgt_tokens_flat = tgt_tokens.reshape(-1)
    
    token_loss = F.cross_entropy(
        token_logits_flat, 
        tgt_tokens_flat, 
        ignore_index=vocab.pad_idx,
        label_smoothing=token_label_smoothing,
        reduction='mean'
    )
    
    # ===== 2. Bond loss (weighted cross-entropy, diagonal ignored) =====
    B, N_bond_type, N_atom_pred, _ = edge_logits.shape
    
    bond_targets = torch.full((B, N_atom_pred, N_atom_pred), -100, dtype=torch.long, device=device)
    
    for b, bond_mat in enumerate(bond_matrices_list):
        N_atom_gt = bond_mat.shape[0]
        
        if N_atom_gt == 0 or N_atom_gt > N_atom_pred:
            continue
            
        bond_mat_tensor = torch.from_numpy(bond_mat.astype(np.int64)).to(device)
        bond_targets[b, :N_atom_gt, :N_atom_gt] = bond_mat_tensor
        bond_targets[b].fill_diagonal_(-100)

    if (bond_targets != -100).any():
        avg_bond_loss = F.cross_entropy(
            edge_logits,
            bond_targets,
            weight=bond_class_weights.to(device),
            ignore_index=-100,
            reduction='mean'
        )
    else:
        avg_bond_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = token_loss + avg_bond_loss
    
    return {
        'token_loss': token_loss,
        'bond_loss': avg_bond_loss,
        'total_loss': total_loss
    }
