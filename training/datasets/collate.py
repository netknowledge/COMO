"""
COMO Collate Function
=====================

Batching function for chartok_coords format training.

Pads variable-length token sequences and atom indices, stacks images.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from como.vocab import ComoVocab


def collate_fn(batch_list: List[Dict], vocab: ComoVocab,
               max_atoms_limit: int = 100, max_seq_len: int = 600) -> Optional[Dict]:
    """Collate function for chartok_coords format.

    Sequences are pre-tokenized in the dataset; this function pads and
    batches them into fixed-size tensors for GPU training.

    Args:
        batch_list: List of dataset items.
        vocab: ComoVocab instance (for PAD index).
        max_atoms_limit: Maximum number of atoms per molecule.
        max_seq_len: Maximum sequence length.
    Returns:
        Batched dict with keys: images, tgt_tokens, tgt_padding_mask,
        atom_indices, atom_mask, max_atoms, bond_matrices_list.
    """
    # 1. Stack images
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0:
        return None

    # Filter out failed/invalid samples
    filtered_batch = []
    for item in batch_list:
        if not item.get('success', True) or len(item['tok_id_seq']) == 0:
            continue
        n_atoms = len(item['atom_indices'])
        if n_atoms <= 0 or n_atoms > max_atoms_limit:
            continue
        if len(item['tok_id_seq']) > max_seq_len:
            continue
        filtered_batch.append(item)

    if len(filtered_batch) == 0:
        return None

    batch_list = filtered_batch
    batch_size = len(batch_list)
    images = torch.stack([item['img'] for item in batch_list])

    # 2. Pad pre-tokenized sequences
    tokenized_seqs = [item['tok_id_seq'] for item in batch_list]
    max_len = max(len(seq) for seq in tokenized_seqs)
    tgt_tokens = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long)
    tgt_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, tokens in enumerate(tokenized_seqs):
        seq_len = len(tokens)
        tgt_tokens[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        tgt_padding_mask[i, :seq_len] = False

    # 3. Pad atom indices
    atom_indices_list = [item['atom_indices'] for item in batch_list]
    max_atoms = max(len(indices) for indices in atom_indices_list)
    max_atoms = max(max_atoms, 1)

    atom_indices = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    for i, indices in enumerate(atom_indices_list):
        n_atoms = len(indices)
        if n_atoms > 0:
            valid_indices = [idx for idx in indices if idx < max_len]
            n_valid = len(valid_indices)
            if n_valid > 0:
                atom_indices[i, :n_valid] = torch.tensor(valid_indices, dtype=torch.long)
                atom_mask[i, :n_valid] = True

    # 4. Collect bond matrices
    bond_matrices_list = []
    for item in batch_list:
        bond_mat = item['bond_mat']
        n = bond_mat.shape[0]
        if n > max_atoms_limit:
            bond_mat = bond_mat[:max_atoms_limit, :max_atoms_limit]
        bond_matrices_list.append(bond_mat)

    return {
        'images': images,
        'tgt_tokens': tgt_tokens,
        'tgt_padding_mask': tgt_padding_mask,
        'atom_indices': atom_indices,
        'atom_mask': atom_mask,
        'max_atoms': max_atoms,
        'bond_matrices_list': bond_matrices_list
    }
