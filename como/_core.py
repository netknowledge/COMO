"""
COMO Model — Inference & Evaluation
====================================

Canonical COMO implementation (inference-only).

End-to-end molecule-image recognition model:
  Image  →  SequenceDecoder (Transformer, autoregressive)  →  SMILES + atom coordinates
         →  BondPredictor   (MLP, pairwise)                →  bond matrix

Sequence format: **chartok_coords**
  [SOS, <SMILES chars>, X_BIN, Y_BIN, <SMILES chars>, X_BIN, Y_BIN, …, EOS]
  Each atom's characters are followed by its binned (x, y) image coordinates.

Three SMILES output modes for evaluation / inference:
  • decoder     – raw SMILES from the decoded token sequence
  • graph       – SMILES reconstructed from predicted atoms + bonds
  • postprocess – decoder SMILES + chirality restoration via predicted coords/edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.distributed as dist
import torchvision.models as models
import torch.multiprocessing as mp

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

from typing import Tuple, List, Optional, Dict
import numpy as np
import random
import string
import os
from functools import partial
import math
import re
import gc
import warnings
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

__all__ = [
    "ComoVocab",
    "ComoModel",
    "validate",
    "evaluate_benchmarks",
]

# ======================== Performance Optimizations ========================
# Enable cuDNN benchmark: safe because image_size is fixed (384x384)
# Gives ~10-20% speedup on convolutions by auto-tuning kernel selection
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Use TensorFloat-32 for faster matmul operations on Ampere+ GPUs (no accuracy loss for training)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
# Disable cv2 threading globally to avoid conflicts with DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# drawing_engine is lazily imported only when needed (training / MRT reward rendering).
# Inference and evaluation do NOT require it.
_drawing_engine = None

def _get_drawing_engine():
    """Lazy-load drawing_engine (and indigo) only when needed."""
    global _drawing_engine
    if _drawing_engine is None:
        from . import _drawing as _drawing_engine
    return _drawing_engine

def generate_image_from_smiles(*args, **kwargs):
    return _get_drawing_engine().generate_image_from_smiles(*args, **kwargs)

def generate_image_from_wild_smiles(*args, **kwargs):
    return _get_drawing_engine().generate_image_from_wild_smiles(*args, **kwargs)

def _blank_image(*args, **kwargs):
    return _get_drawing_engine()._blank_image(*args, **kwargs)
from ._chemistry import (
    _convert_graph_to_smiles, _verify_chirality, canonicalize_smiles,
    _replace_functional_group, _expand_functional_group, canonicalize_tautomer,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from SmilesPE.pretokenizer import atomwise_tokenizer

# Three SMILES prediction modes (used in validation / evaluation / inference)
SMILES_MODE_DECODER = 'decoder'        # SMILES directly from decoder sequence
SMILES_MODE_GRAPH   = 'graph'          # SMILES reconstructed from predicted atoms + bonds
SMILES_MODE_POSTPROCESS = 'postprocess' # decoder SMILES + chirality correction via predicted coords/edges


# ======================== Minimal stubs (replacing DDP imports) ================

def is_main_process() -> bool:
    """Always True in inference-only mode (no distributed training)."""
    return True

# ======================== Vocabulary & Tokenizer ========================

class ComoVocab:
    """Canonical COMO chartok_coords vocabulary.

    Token layout: [special] + [SMILES chars] + [X_BIN_0 … X_BIN_{n-1}] + [Y_BIN_0 … Y_BIN_{n-1}]
    Coordinates are quantized into *n_bins* bins per axis.
    """
    
    def __init__(self, n_bins: int = 64):
        """
        Args:
            n_bins: Number of bins for coordinate quantization (0 to n_bins-1)
        """
        self.n_bins = n_bins
        
        # Special tokens (Removed wrapper tokens)
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        
        special_tokens = [self.PAD, self.SOS, self.EOS, self.UNK]
        
        # SMILES character set
        uppercase = string.ascii_uppercase  # A-Z
        lowercase = string.ascii_lowercase  # a-z
        digits = string.digits  # 0-9
        symbols = ['[', ']', '(', ')', '=', '#', '@', '+', '-', '/', '\\', '.', '%']
        
        smiles_chars = list(uppercase) + list(lowercase) + list(digits) + symbols
        
        # Separate X and Y coordinate bins (0 to n_bins-1)
        x_coord_bins = [f'<X_BIN_{i}>' for i in range(n_bins)]
        y_coord_bins = [f'<Y_BIN_{i}>' for i in range(n_bins)]
        
        # Build vocab: [special] + [smiles_chars] + [X_bins] + [Y_bins]
        self.tokens = special_tokens + smiles_chars + x_coord_bins + y_coord_bins
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        self.pad_idx = self.token2idx[self.PAD]
        self.sos_idx = self.token2idx[self.SOS]
        self.eos_idx = self.token2idx[self.EOS]
        self.unk_idx = self.token2idx[self.UNK]

        # Pre-compute bin token ID ranges for fast checking
        self.n_smiles_chars = len(smiles_chars)
        self.n_special = len(special_tokens)
        
        # X bins range
        self.x_bin_start_idx = self.n_special + self.n_smiles_chars
        self.x_bin_end_idx = self.x_bin_start_idx + n_bins - 1
        
        # Y bins range
        self.y_bin_start_idx = self.x_bin_end_idx + 1
        self.y_bin_end_idx = self.y_bin_start_idx + n_bins - 1
        
        # Combined range for backward compatibility
        self.bin_start_idx = self.x_bin_start_idx
        self.bin_end_idx = self.y_bin_end_idx
        
    def __len__(self):
        return len(self.tokens)

    def is_coord_token(self, idx: int) -> bool:
        """Check if a token index represents any coordinate bin (X or Y)."""
        return self.bin_start_idx <= idx <= self.bin_end_idx
    
    def is_x_coord_token(self, idx: int) -> bool:
        """Check if a token index represents an X coordinate bin."""
        return self.x_bin_start_idx <= idx <= self.x_bin_end_idx
    
    def is_y_coord_token(self, idx: int) -> bool:
        """Check if a token index represents a Y coordinate bin."""
        return self.y_bin_start_idx <= idx <= self.y_bin_end_idx
    
    def is_symbol(self, idx: int) -> bool:
        """Check if token index represents a SMILES symbol (not special, not coordinate)."""
        return self.n_special <= idx < self.x_bin_start_idx

    @staticmethod
    def is_atom_token(token: str) -> bool:
        """Check if a SMILES token (from atomwise_tokenizer) represents an atom."""
        return token.isalpha() or token.startswith("[") or token == '*'

    def smiles_to_sequence(self, smiles: str, coords: Optional[List] = None) -> Tuple[List[int], List[int]]:
        """
        Tokenize SMILES with interleaved coordinates (chartok_coords style).
        
        Format: [SOS, ..., atom_chars, X_BIN, Y_BIN, bond_char, atom_chars, X_BIN, Y_BIN, ..., EOS]
        Coordinates are inserted after each atom token's characters.
        
        Args:
            smiles: SMILES string
            coords: List of [x_bin, y_bin] for each atom, in SMILES atom order.
                    Values should already be in range [0, n_bins-1].
        Returns:
            labels: List of token indices
            indices: List of positions of Y_BIN tokens (one per atom, for bond predictor)
        """
        tokens = atomwise_tokenizer(smiles)
        labels = [self.sos_idx]
        indices = []
        atom_idx = -1

        for token in tokens:
            # Tokenize each character of the SMILES token
            for c in token:
                if c in self.token2idx:
                    labels.append(self.token2idx[c])
                else:
                    labels.append(self.unk_idx)

            # If this token is an atom, append X_BIN and Y_BIN
            if self.is_atom_token(token):
                atom_idx += 1
                if coords is not None and atom_idx < len(coords):
                    x_bin = max(0, min(self.n_bins - 1, int(coords[atom_idx][0])))
                    y_bin = max(0, min(self.n_bins - 1, int(coords[atom_idx][1])))
                    labels.append(self.token2idx[f'<X_BIN_{x_bin}>'])
                    labels.append(self.token2idx[f'<Y_BIN_{y_bin}>'])
                else:
                    # Fallback: use bin 0 (should not happen in normal training)
                    labels.append(self.token2idx['<X_BIN_0>'])
                    labels.append(self.token2idx['<Y_BIN_0>'])
                indices.append(len(labels) - 1)  # Position of Y_BIN

        labels.append(self.eos_idx)
        return labels, indices

    def sequence_to_smiles(self, sequence: List[int]) -> Dict:
        """
        Detokenize a chartok_coords sequence back to SMILES + coords + symbols.
        
        Returns:
            dict with keys:
                'smiles': reconstructed SMILES string
                'symbols': list of atom symbol strings
                'coords': list of [x_bin, y_bin] per atom
                'indices': list of Y_BIN positions in the sequence (for bond predictor)
                'success': bool
        """
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0

        if len(sequence) > 0 and sequence[0] == self.sos_idx:
            i = 1

        while i < len(sequence):
            label = sequence[i]
            if label == self.eos_idx or label == self.pad_idx:
                break
            # Skip coordinate tokens (they are consumed when following an atom)
            if self.is_x_coord_token(label) or self.is_y_coord_token(label):
                i += 1
                continue
            # Skip special tokens
            if label in (self.pad_idx, self.sos_idx, self.unk_idx):
                i += 1
                continue

            token_str = self.idx2token.get(label, '')

            # --- Bracket atom: [...]  ---
            if token_str == '[':
                j = i + 1
                while j < len(sequence):
                    jt = self.idx2token.get(sequence[j], '')
                    if not self.is_symbol(sequence[j]):
                        break
                    if jt == ']':
                        j += 1
                        break
                    j += 1
                atom_token = ''.join(self.idx2token.get(sequence[k], '') for k in range(i, j))
                smiles += atom_token
                # Read following coords
                if (j + 1 < len(sequence)
                        and self.is_x_coord_token(sequence[j])
                        and self.is_y_coord_token(sequence[j + 1])):
                    x_val = int(self.idx2token[sequence[j]][7:-1])
                    y_val = int(self.idx2token[sequence[j + 1]][7:-1])
                    coords.append([x_val, y_val])
                    symbols.append(atom_token)
                    indices.append(j + 1)
                    i = j + 2
                else:
                    i = j

            # --- Regular atom (uppercase letter, or lowercase aromatic atom) ---
            elif token_str.isalpha():
                j = i + 1
                # Check for two-letter atoms: Cl, Br (uppercase), se, te (aromatic)
                if (j < len(sequence) and self.is_symbol(sequence[j])):
                    next_ch = self.idx2token.get(sequence[j], '')
                    if ((token_str == 'C' and next_ch == 'l')
                            or (token_str == 'B' and next_ch == 'r')
                            or (token_str == 's' and next_ch == 'e')
                            or (token_str == 't' and next_ch == 'e')):
                        j = i + 2
                atom_token = ''.join(self.idx2token.get(sequence[k], '') for k in range(i, j))
                smiles += atom_token
                # Read following coords
                if (j + 1 < len(sequence)
                        and self.is_x_coord_token(sequence[j])
                        and self.is_y_coord_token(sequence[j + 1])):
                    x_val = int(self.idx2token[sequence[j]][7:-1])
                    y_val = int(self.idx2token[sequence[j + 1]][7:-1])
                    coords.append([x_val, y_val])
                    symbols.append(atom_token)
                    indices.append(j + 1)
                    i = j + 2
                else:
                    i = j

            # --- Non-atom symbol (bond chars: =, #, (, ), digits, etc.) ---
            else:
                smiles += token_str
                i += 1

        success = len(symbols) > 0
        return {
            'smiles': smiles,
            'symbols': symbols,
            'coords': coords,
            'indices': indices,
            'success': success
        }

    def get_output_mask(self, token_idx: int) -> List[bool]:
        """
        Get output constraint mask for the next token given current token.
        Returns a list of bools where True = disallowed.
        
        Rules:
            After X_BIN  → only Y_BIN is allowed
            After Y_BIN  → no coordinate tokens allowed (symbols or EOS)
            Otherwise    → no Y_BIN allowed (must go through X first)
        """
        mask = [False] * len(self)
        if self.is_x_coord_token(token_idx):
            # After X_BIN → only Y_BIN
            for i in range(len(self)):
                if not self.is_y_coord_token(i):
                    mask[i] = True
        elif self.is_y_coord_token(token_idx):
            # After Y_BIN → no coords, no PAD/SOS
            for i in range(self.x_bin_start_idx, self.y_bin_end_idx + 1):
                mask[i] = True
            mask[self.pad_idx] = True
            mask[self.sos_idx] = True
        else:
            # After symbol/SOS → no Y_BIN (must produce X first if starting coords)
            for i in range(self.y_bin_start_idx, self.y_bin_end_idx + 1):
                mask[i] = True
            mask[self.pad_idx] = True
            mask[self.sos_idx] = True
        return mask


# ======================== Image Encoder ========================

class ImageEncoder(nn.Module):
    """
    Image encoder with ResNet-50 or Swin Transformer backbone.
    Outputs spatial feature maps for Transformer Decoder.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        d_model: int = 512
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            # Remove avgpool and fc
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feat_dim = 2048

        elif backbone.startswith('swin'):
            if backbone == 'swin_t':
                swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_s':
                swin = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_b':
                swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            elif backbone == 'swin_v2_t':
                swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_s':
                swin = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 768
            elif backbone == 'swin_v2_b':
                swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)
                self.feat_dim = 1024
            else:
                raise ValueError(f"Unknown Swin variant: {backbone}")
            
            self.backbone = swin.features
            self.norm = swin.norm
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Project to d_model
        self.proj = nn.Conv2d(self.feat_dim, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB images
        Returns:
            [B, d_model, H', W'] feature maps
        """
        if self.backbone_name == 'resnet50':
            x = self.backbone(x)
        else:
            x = self.backbone(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.proj(x)
        return x

# ======================== Positional Encoding ========================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for image features."""
    
    def __init__(self, d_model: int, max_h: int = 100, max_w: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # d_model must be divisible by 4 (split among y_sin, y_cos, x_sin, x_cos)
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
        
        pe = torch.zeros(max_h, max_w, d_model)
        
        y_pos = torch.arange(0, max_h).unsqueeze(1).float()  # [max_h, 1]
        x_pos = torch.arange(0, max_w).unsqueeze(1).float()  # [max_w, 1]
        
        # Each axis gets d_model/4 frequency components
        dim_per_axis = d_model // 4
        div_term = torch.exp(torch.arange(0, dim_per_axis) * -(np.log(10000.0) / dim_per_axis))
        
        # Compute sinusoidal encodings per axis
        y_sin = torch.sin(y_pos * div_term)  # [max_h, dim_per_axis]
        y_cos = torch.cos(y_pos * div_term)  # [max_h, dim_per_axis]
        x_sin = torch.sin(x_pos * div_term)  # [max_w, dim_per_axis]
        x_cos = torch.cos(x_pos * div_term)  # [max_w, dim_per_axis]
        
        # Assemble [max_h, max_w, d_model] encoding table
        for i in range(max_h):
            for j in range(max_w):
                pe[i, j, 0*dim_per_axis:1*dim_per_axis] = y_sin[i]
                pe[i, j, 1*dim_per_axis:2*dim_per_axis] = y_cos[i]
                pe[i, j, 2*dim_per_axis:3*dim_per_axis] = x_sin[j]
                pe[i, j, 3*dim_per_axis:4*dim_per_axis] = x_cos[j]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, H, W]
        Returns:
            x + positional encoding: [B, d_model, H, W]
        """
        B, C, H, W = x.shape
        pe_tensor: torch.Tensor = self.pe
        pe = pe_tensor[:H, :W, :].permute(2, 0, 1).unsqueeze(0)  # [1, d_model, H, W]
        return x + pe


# ======================== KV-Cache Helpers ========================

def _mha_with_kv_cache(
    mha: nn.MultiheadAttention,
    q_input: torch.Tensor,
    kv_new_input: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    cached_v: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-head attention with KV-cache for incremental decoding.

    Projects only the *new* tokens for K/V, concatenates with cached K/V from
    previous steps, and computes attention for the query input.

    Args:
        mha: ``nn.MultiheadAttention`` module (must use packed ``in_proj_weight``).
        q_input:  ``[B, Lq, d_model]`` query input.
        kv_new_input: ``[B, Lnew, d_model]`` new key/value input to project and
                      append to cache, **or** ``None`` to reuse the cache as-is
                      (useful for cross-attention after the first step).
        cached_k: ``[B, n_heads, Lprev, d_head]`` or ``None``.
        cached_v: ``[B, n_heads, Lprev, d_head]`` or ``None``.

    Returns:
        output:  ``[B, Lq, d_model]`` attention output.
        new_k:   ``[B, n_heads, Lprev+Lnew, d_head]`` updated key cache.
        new_v:   ``[B, n_heads, Lprev+Lnew, d_head]`` updated value cache.
    """
    B = q_input.size(0)
    d = mha.embed_dim
    n_heads = mha.num_heads
    d_head = d // n_heads

    W = mha.in_proj_weight  # [3d, d]
    b = mha.in_proj_bias    # [3d] or None

    # --- Q projection (always from q_input) ---
    q = F.linear(q_input, W[:d], b[:d] if b is not None else None)
    q = q.view(B, -1, n_heads, d_head).transpose(1, 2)  # [B, heads, Lq, d_head]

    # --- K / V projection + cache concatenation ---
    if kv_new_input is not None:
        k_new = F.linear(kv_new_input, W[d:2*d], b[d:2*d] if b is not None else None)
        v_new = F.linear(kv_new_input, W[2*d:],  b[2*d:]  if b is not None else None)
        k_new = k_new.view(B, -1, n_heads, d_head).transpose(1, 2)
        v_new = v_new.view(B, -1, n_heads, d_head).transpose(1, 2)
        k = torch.cat([cached_k, k_new], dim=2) if cached_k is not None else k_new
        v = torch.cat([cached_v, v_new], dim=2) if cached_v is not None else v_new
    else:
        assert cached_k is not None, "Must provide kv_new_input or cached_k/v"
        k, v = cached_k, cached_v

    # --- Scaled dot-product attention ---
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)                              # [B, heads, Lq, d_head]
    out = out.transpose(1, 2).contiguous().view(B, -1, d)    # [B, Lq, d]
    out = F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    return out, k, v


# ======================== Sequence Decoder (Transformer) ========================

class SequenceDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for chartok_coords token prediction.

    Consumes encoder memory (image features) and generates the full
    SMILES-with-coordinates token sequence.  Causal-mask caching and
    optional gradient checkpointing are supported for efficiency.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token embedding (shared for all vocab tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for sequence
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head: predict next token in vocabulary
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Cache for causal masks to avoid recreation every forward pass
        self._causal_mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
    
    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Get cached causal mask or create and cache a new one."""
        cache_key = (T, device)
        if cache_key not in self._causal_mask_cache:
            self._causal_mask_cache[cache_key] = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device),
                diagonal=1
            )
        return self._causal_mask_cache[cache_key]
    
    def _decoder_forward(self, tgt_emb: torch.Tensor, memory: torch.Tensor, 
                         tgt_mask: torch.Tensor, tgt_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Wrapper for decoder forward pass, used for gradient checkpointing."""
        return self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
    
    def forward(
        self,
        img_features: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_features: [B, d_model, H, W] from ImageEncoder
            tgt_tokens: [B, T] target token indices (teacher forcing)
            tgt_key_padding_mask: [B, T] padding mask (for variable length)
        
        Returns:
            hidden_states: [B, T, d_model] - decoder hidden states
            logits: [B, T, vocab_size] - predictions for each position
        """
        B, C, H, W = img_features.shape
        T = tgt_tokens.size(1)
        device = tgt_tokens.device
        
        # Flatten image features to sequence: [B, H*W, d_model]
        memory = img_features.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]
        
        # Embed target tokens
        tgt_emb = self.token_embedding(tgt_tokens)  # [B, T, d_model]
        
        # Add positional encoding (optimized: avoid expand by using broadcasting)
        positions = torch.arange(T, device=device)
        tgt_emb = tgt_emb + self.pos_encoder(positions)  # broadcasts [T, d] to [B, T, d]

        # Get cached causal mask
        tgt_mask = self._get_causal_mask(T, device)
        
        # Transformer Decoder with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointing saves memory by recomputing activations during backward
            output = checkpoint(
                self._decoder_forward,
                tgt_emb, memory, tgt_mask, tgt_key_padding_mask,
                use_reentrant=False
            )
        else:
            output = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )  # [B, T, d_model]
        
        # Predict next token
        logits = self.output_head(output)  # [B, T, vocab_size]
        
        return output, logits

    def forward_step_cached(
        self,
        memory: torch.Tensor,
        new_token_ids: torch.Tensor,
        step_idx: int,
        cache: Optional[List[Dict[str, Optional[torch.Tensor]]]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Optional[torch.Tensor]]]]:
        """One-step forward through the decoder with a KV-cache.

        Instead of re-encoding the full growing sequence at every step
        (*O(T²)* per step, *O(T³)* total), this method processes only the
        **new token** and reuses projected K/V tensors from earlier steps
        (*O(T)* per step, *O(T²)* total).

        Args:
            memory: ``[B, S, d_model]`` flattened encoder features (compute
                    once via ``img_features.flatten(2).permute(0,2,1)``).
            new_token_ids: ``[B]`` token indices for the current step.
            step_idx: 0-based decoding step index.
            cache: list of per-layer dicts with keys ``self_k``, ``self_v``,
                   ``cross_k``, ``cross_v`` (each ``[B, heads, T, d_head]``
                   or ``None``).  Pass ``None`` on the first call.

        Returns:
            logits: ``[B, vocab_size]`` next-token logits.
            new_cache: updated cache (same structure, one dict per layer).
        """
        B = new_token_ids.size(0)
        device = new_token_ids.device

        # Embed the single new token  →  [B, 1, d_model]
        pos = torch.full((B, 1), step_idx, dtype=torch.long, device=device)
        h = self.token_embedding(new_token_ids.unsqueeze(1)) + self.pos_encoder(pos)

        num_layers = len(self.transformer_decoder.layers)
        if cache is None:
            cache = [
                {'self_k': None, 'self_v': None, 'cross_k': None, 'cross_v': None}
                for _ in range(num_layers)
            ]

        new_cache: List[Dict[str, Optional[torch.Tensor]]] = []
        for i, layer in enumerate(self.transformer_decoder.layers):
            lc = cache[i]

            # --- Self-attention (post-norm, Q=new token, K/V grows) ---
            sa_out, sa_k, sa_v = _mha_with_kv_cache(
                layer.self_attn, h, h, lc['self_k'], lc['self_v'])
            h = layer.norm1(h + layer.dropout1(sa_out))

            # --- Cross-attention (K/V=memory, projected once at step 0) ---
            cross_new = memory if lc['cross_k'] is None else None
            ca_out, ca_k, ca_v = _mha_with_kv_cache(
                layer.multihead_attn, h, cross_new, lc['cross_k'], lc['cross_v'])
            h = layer.norm2(h + layer.dropout2(ca_out))

            # --- Feed-forward ---
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm3(h + layer.dropout3(ff))

            new_cache.append({
                'self_k': sa_k, 'self_v': sa_v,
                'cross_k': ca_k, 'cross_v': ca_v,
            })

        # Final norm (present only if TransformerDecoder was built with one)
        if self.transformer_decoder.norm is not None:
            h = self.transformer_decoder.norm(h)

        logits = self.output_head(h.squeeze(1))  # [B, vocab_size]
        return logits, new_cache


# ======================== Bond Predictor ========================

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
            n_bond_classes: Number of bond types (0–6: none, single, double, triple, aromatic, wedge-solid, wedge-dash).
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


# ======================== Helper: Symmetrize Edge Predictions ========================

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

def extract_atom_indices_from_tokens(
    tgt_tokens: torch.Tensor, 
    vocab: ComoVocab
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized extraction of atom positions from chartok_coords sequences.

    In the format [... atom_chars, X_BIN, Y_BIN ...], each atom is anchored
    at its Y_BIN token.  This function returns those positions.

    Args:
        tgt_tokens: [B, T] token-index tensor.
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

# ======================== CropWhite for Inference ========================

class AdaptiveLongestMaxSize(A.ImageOnlyTransform):
    """LongestMaxSize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    """
    
    def __init__(self, max_size, p=1.0):
        super().__init__(p=p)
        self.max_size = max_size

    def apply(self, img, **params):
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest == self.max_size:
            return img
        scale = self.max_size / longest
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        interpolation = cv2.INTER_AREA if longest > self.max_size else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    def get_transform_init_args_names(self):
        return ('max_size',)


class AdaptiveResize(A.DualTransform):
    """Resize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    Supports keypoint tracking (DualTransform).
    """

    def __init__(self, height, width, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]
        interpolation = cv2.INTER_AREA if (h > self.height or w > self.width) else cv2.INTER_LINEAR
        return {
            "scale_x": self.width / w,
            "scale_y": self.height / h,
            "interpolation": interpolation,
        }

    def apply(self, img, scale_x=1, scale_y=1, interpolation=cv2.INTER_LINEAR, **params):
        return cv2.resize(img, (self.width, self.height), interpolation=interpolation)

    def apply_to_keypoints(self, keypoints, scale_x=1, scale_y=1, **params):
        if keypoints.size == 0:
            return keypoints
        result = keypoints.copy()
        result[:, 0] *= scale_x
        result[:, 1] *= scale_y
        return result

    def get_transform_init_args_names(self):
        return ('height', 'width')


class CropWhiteInference(A.ImageOnlyTransform):
    """Crop white borders from images during inference.
    
    Finds the bounding box of non-white pixels and crops the image,
    keeping a small padding. This removes wasted white space so that
    the molecule content fills more of the final resized image.
    """
    
    def __init__(self, value=(255, 255, 255), pad=5, p=1.0):
        super().__init__(p=p)
        self.value = value
        self.pad = pad
    
    def apply(self, img, **params):
        height, width = img.shape[:2]
        # Find non-white pixels
        if img.ndim == 3:
            non_white = (img != self.value).any(axis=2)
        else:
            non_white = (img != 255)
        
        if not non_white.any():
            return img
        
        # Find bounding box of non-white region
        rows = non_white.any(axis=1)
        cols = non_white.any(axis=0)
        top = rows.argmax()
        bottom = height - rows[::-1].argmax()
        left = cols.argmax()
        right = width - cols[::-1].argmax()
        
        # Crop with padding
        top = max(0, top - self.pad)
        bottom = min(height, bottom + self.pad)
        left = max(0, left - self.pad)
        right = min(width, right + self.pad)
        
        return img[top:bottom, left:right]
    
    def get_transform_init_args_names(self):
        return ('value', 'pad')

# ======================== COMO Model ========================

class ComoModel(nn.Module):
    """Legacy compatibility name for the canonical COMO model.

    Architecture:
      ImageEncoder  (Swin / ResNet)  →  2-D positional encoding
      SequenceDecoder (Transformer)  →  chartok_coords token logits + hidden states
      BondPredictor   (MLP)          →  pairwise bond-type logits
    """
    
    def __init__(
        self,
        vocab: ComoVocab,
        image_size: Tuple[int, int] = (384, 384),
        backbone: str = 'resnet50',
        pretrained: bool = False,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 256*4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.image_size = image_size
        
        # Image Encoder
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            d_model=d_model
        )
        self.pos_enc_2d = PositionalEncoding2D(d_model, max_h=20, max_w=20)
        
        # Sequence Decoder (autoregressive Transformer)
        self.sequence_decoder = SequenceDecoder(
            vocab_size=len(vocab),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Bond Predictor (pairwise MLP)
        self.bond_predictor = BondPredictor(
            d_model=d_model,
            n_bond_classes=7,
        )
        
        # Inference transform: crop white borders, resize with padding
        self.inference_transform_list = [
            CropWhiteInference(pad=5),
            AdaptiveLongestMaxSize(max_size=self.image_size[0]),  # INTER_LINEAR↑ / INTER_AREA↓
            A.PadIfNeeded(
                min_height=self.image_size[0], 
                min_width=self.image_size[1], 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=255
            ),
            A.ToGray(num_output_channels=3),
            A.Normalize(),
            ToTensorV2(),
        ]
        self.inference_transforms = A.Compose(self.inference_transform_list)
    
    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        atom_indices: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        max_atoms: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit atom mask."""
        # Defensive shape check for images
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(
                f"Expected images with shape [B, 3, H, W], got {images.shape}. "
                f"Ensure ToGray uses num_output_channels=3."
            )
        
        # Image encoding
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        B, C, H, W = img_features.shape
        
        # Sequence decoding (teacher-forced)
        T = tgt_tokens.size(1)

        max_pos = self.sequence_decoder.pos_encoder.num_embeddings
        if T > max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {max_pos}. "
                f"This should be handled before calling forward()."
            )

        hidden_states, token_logits = self.sequence_decoder(
            img_features=img_features,
            tgt_tokens=tgt_tokens,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # ===== Bond prediction =====
        if atom_indices is None or atom_mask is None or max_atoms is None:
            # Inference mode: generate atom indices and mask from tgt_tokens
            atom_indices, atom_counts = extract_atom_indices_from_tokens(tgt_tokens, self.vocab)
            max_atoms = int(atom_counts.max().item())
            
            if max_atoms == 0:
                edge_logits_padded = torch.zeros((B, 7, 1, 1), device=images.device)
                return token_logits, edge_logits_padded, hidden_states
            
            atom_mask = torch.zeros((B, max_atoms), dtype=torch.bool, device=images.device)
            for i, count in enumerate(atom_counts):
                if count > 0:
                    atom_mask[i, :count] = True
        
        # ===== Align atom_indices tensor to expected max_atoms dimension =====
        current_size = atom_indices.size(1)
        
        if current_size != max_atoms:
            if current_size < max_atoms:
                padding_size = max_atoms - current_size
                atom_indices = F.pad(atom_indices, (0, padding_size), value=0)
                atom_mask = F.pad(atom_mask, (0, padding_size), value=False)
            else:
                atom_indices = atom_indices[:, :max_atoms]
                atom_mask = atom_mask[:, :max_atoms]
        
        atom_indices = atom_indices.to(images.device)
        atom_mask = atom_mask.to(images.device)
        
        # Predict bonds
        if max_atoms == 0:
            edge_logits_padded = torch.zeros((B, 7, 1, 1), device=images.device)
        else:
            T_hidden = hidden_states.size(1)
            atom_indices = atom_indices.clamp(0, T_hidden - 1)
            
            edge_logits_padded = self.bond_predictor(
                hidden_states, 
                atom_indices,
                atom_mask
            )
        
        return token_logits, edge_logits_padded, hidden_states

    def load_model(self, path: str, device: Optional[torch.device] = None):
        """Load trained model weights."""
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        
        if is_main_process():
            print(f'Model loaded from: {path}')

# ======================== Inference Functions ========================

    def predict_step(self, img_features: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits given image features and the sequence decoded so far."""
        _, logits = self.sequence_decoder(
            img_features=img_features,
            tgt_tokens=current_tokens,
            tgt_key_padding_mask=None
        )
        return logits[:, -1, :]

    def _apply_constraints(self, logits: torch.Tensor, last_token: int) -> torch.Tensor:
        """Apply structural constraints for chartok_coords format.

        Rules:
          - After X_BIN  → only Y_BIN is allowed
          - After Y_BIN  → SMILES chars + EOS (no coords)
          - After SOS    → SMILES chars only (no coords, no EOS)
          - After SMILES char → SMILES chars + X_BIN + EOS (no Y_BIN)
        """
        logits = logits.clone()
        vocab = self.vocab

        is_last_x_bin = vocab.is_x_coord_token(last_token)
        is_last_y_bin = vocab.is_y_coord_token(last_token)

        if is_last_x_bin:
            # After X_BIN → only Y_BIN allowed
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits.masked_fill_(mask, float('-inf'))

        elif is_last_y_bin:
            # After Y_BIN → SMILES chars + EOS; no coords
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')

        elif last_token == vocab.sos_idx:
            # After SOS → SMILES chars only (no coords, no EOS)
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.eos_idx, vocab.unk_idx]] = float('-inf')

        else:
            # After a SMILES char → allow SMILES chars + X_BIN + EOS; no Y_BIN
            logits[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')

        return logits

    def _apply_constraints_batch(self, logits: torch.Tensor,
                                 last_tokens: torch.Tensor,
                                 finished: torch.Tensor) -> torch.Tensor:
        """Vectorised chartok_coords constraints for a whole batch.

        Applies the same four rules as :meth:`_apply_constraints` but
        operates on the full ``[B, V]`` logits tensor at once using
        boolean index masking.  This avoids the per-sample ``.item()``
        calls that would otherwise force a GPU→CPU sync at **every**
        decoding step for **every** sequence.

        Args:
            logits: ``[B, V]`` raw logits from the decoder step.
            last_tokens: ``[B]`` previous token ids (on GPU).
            finished: ``[B]`` bool mask; True for sequences that have
                already emitted EOS (their logits are left untouched).

        Returns:
            ``[B, V]`` logits with disallowed positions set to ``-inf``.
        """
        vocab = self.vocab
        B, V = logits.shape

        NEG_INF = float('-inf')

        # Classify each sequence's last token into one of four categories [B]
        is_x = (last_tokens >= vocab.x_bin_start_idx) & (last_tokens <= vocab.x_bin_end_idx)
        is_y = (last_tokens >= vocab.y_bin_start_idx) & (last_tokens <= vocab.y_bin_end_idx)
        is_sos = (last_tokens == vocab.sos_idx)
        is_char = ~is_x & ~is_y & ~is_sos & ~finished

        # ------ After X_BIN → only Y_BIN allowed ------
        if is_x.any():
            # Mask: everything except Y_BIN range
            x_mask = torch.ones(V, dtype=torch.bool, device=logits.device)
            x_mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits[is_x] = logits[is_x].masked_fill(x_mask.unsqueeze(0), NEG_INF)

        # ------ After Y_BIN → no coords, no PAD/SOS/UNK ------
        if is_y.any():
            y_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            y_mask[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            y_mask[vocab.pad_idx] = True
            y_mask[vocab.sos_idx] = True
            y_mask[vocab.unk_idx] = True
            logits[is_y] = logits[is_y].masked_fill(y_mask.unsqueeze(0), NEG_INF)

        # ------ After SOS → no coords, no PAD/SOS/EOS/UNK ------
        if is_sos.any():
            sos_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            sos_mask[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            sos_mask[vocab.pad_idx] = True
            sos_mask[vocab.sos_idx] = True
            sos_mask[vocab.eos_idx] = True
            sos_mask[vocab.unk_idx] = True
            logits[is_sos] = logits[is_sos].masked_fill(sos_mask.unsqueeze(0), NEG_INF)

        # ------ After SMILES char → no Y_BIN, no PAD/SOS/UNK ------
        if is_char.any():
            char_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            char_mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            char_mask[vocab.pad_idx] = True
            char_mask[vocab.sos_idx] = True
            char_mask[vocab.unk_idx] = True
            logits[is_char] = logits[is_char].masked_fill(char_mask.unsqueeze(0), NEG_INF)

        return logits

    def _greedy_decode(self, feat: torch.Tensor, max_len: int, device: torch.device) -> List[int]:
        """Greedy decoding for single image."""
        seq = [self.vocab.sos_idx]
        
        for _ in range(max_len):
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self.predict_step(feat, tgt)[0]
            
            last_token = seq[-1]
            logits = self._apply_constraints(logits, last_token)
            
            next_token = torch.argmax(logits).item()
            seq.append(next_token)
            if next_token == self.vocab.eos_idx:
                break
        
        return seq

    def _beam_search_decode(self, feat: torch.Tensor, beam_size: int, max_len: int, device: torch.device) -> List[int]:
        """Beam search decoding for single image."""
        beams = [([self.vocab.sos_idx], 0.0)]
        completed = []
        
        for _ in range(max_len):
            if not beams:
                break
                
            candidates = []
            for seq, score in beams:
                if seq[-1] == self.vocab.eos_idx:
                    completed.append((seq, score))
                    continue
                
                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.predict_step(feat, tgt)[0]
                
                last_token = seq[-1]
                logits = self._apply_constraints(logits, last_token)
                
                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                for prob, idx in zip(topk_probs.tolist(), topk_ids.tolist()):
                    if prob > float('-inf'):
                        candidates.append((seq + [idx], score + prob))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            if len(completed) >= beam_size:
                break
            if completed and beams and beams[0][1] < max(completed, key=lambda x: x[1])[1]:
                break
        
        completed.extend(beams)
        return max(completed, key=lambda x: x[1])[0] if completed else [self.vocab.sos_idx]

    def _beam_search_top_n(
        self,
        feat: torch.Tensor,
        n_top: int,
        max_len: int,
        device: torch.device,
    ) -> List[List[int]]:
        """Per-image beam search returning the top-N sequences by log-prob.

        Uses KV-cached decoding (``forward_step_cached``) for efficiency.
        All ``n_top`` beams are batched together for a single image, so the
        GPU processes ``[n_top, ...]`` tensors at each step.

        Structural constraints are applied via ``_apply_constraints_batch``.

        Args:
            feat: ``[1, d_model, H, W]`` encoded features for **one** image.
            n_top: Beam width and number of returned sequences.
            max_len: Maximum decoding length.
            device: Compute device.

        Returns:
            List of ``n_top`` token-id lists, sorted best-first by
            cumulative log-probability. If fewer than ``n_top`` sequences
            complete, the remainder is filled from the best incomplete beams.
        """
        vocab = self.vocab
        beam_size = n_top

        # Expand single-image features to beam_size copies: [1,...] → [beam_size,...]
        memory = feat.expand(beam_size, -1, -1, -1).flatten(2).permute(0, 2, 1)  # [beam_size, S, d_model]

        # Initialize: all beams start with SOS
        tokens = torch.full((beam_size,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)  # [beam_size, 1]
        scores = torch.zeros(beam_size, device=device)  # cumulative log-probs
        finished = torch.zeros(beam_size, dtype=torch.bool, device=device)
        cache = None

        completed: List[Tuple[List[int], float]] = []  # (seq, score)

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )  # logits: [beam_size, vocab_size]

            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            log_probs = F.log_softmax(logits, dim=-1)  # [beam_size, V]
            V = log_probs.size(-1)

            # Candidate scores: [beam_size, V]
            candidate_scores = scores.unsqueeze(1) + log_probs
            # For finished beams, only keep their current score (no expansion)
            if finished.any():
                candidate_scores[finished] = float('-inf')
                # Preserve finished beam scores via a sentinel column (pad token)
                candidate_scores[finished, vocab.pad_idx] = scores[finished]

            # Flatten to [beam_size * V], take top beam_size
            flat_scores = candidate_scores.view(-1)
            topk_scores, topk_indices = torch.topk(flat_scores, beam_size)

            beam_indices = topk_indices // V  # which beam each came from
            token_indices = topk_indices % V  # which token was chosen

            # Check for newly completed beams (hit EOS)
            new_eos = (token_indices == vocab.eos_idx) & ~finished[beam_indices]
            if new_eos.any():
                for k in new_eos.nonzero(as_tuple=False).squeeze(-1).tolist():
                    src_beam = beam_indices[k].item()
                    full_seq = seqs[src_beam].tolist() + [vocab.eos_idx]
                    completed.append((full_seq, topk_scores[k].item()))

            # Reindex beams: reorder seqs, cache, finished flags
            seqs = torch.cat([seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)
            scores = topk_scores
            finished = finished[beam_indices] | (token_indices == vocab.eos_idx)

            # Reindex KV-cache
            new_cache = []
            for layer_cache in cache:
                new_cache.append({
                    'self_k': layer_cache['self_k'][:, :, :, :].index_select(0, beam_indices) if layer_cache['self_k'] is not None else None,
                    'self_v': layer_cache['self_v'][:, :, :, :].index_select(0, beam_indices) if layer_cache['self_v'] is not None else None,
                    'cross_k': layer_cache['cross_k'],  # same for all beams (shared memory)
                    'cross_v': layer_cache['cross_v'],
                })
            cache = new_cache

            tokens = token_indices  # next step input

            # Early stop: enough completed sequences
            if len(completed) >= beam_size:
                # Check if best incomplete beam can't beat worst completed
                best_incomplete = scores[~finished].max().item() if (~finished).any() else float('-inf')
                worst_completed = min(c[1] for c in completed)
                if best_incomplete <= worst_completed:
                    break

        # Add remaining incomplete beams (trim trailing PAD)
        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(beam_size):
            if not finished[b]:
                seq = seqs[b].tolist()
                while seq and seq[-1] == pad:
                    seq.pop()
                completed.append((seq, scores[b].item()))

        # Sort by score (descending) and return top n_top
        completed.sort(key=lambda x: x[1], reverse=True)
        result = []
        for seq, _score in completed[:n_top]:
            # Trim to EOS if present, remove trailing PAD
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)

        # Pad with SOS-only sequences if we somehow got fewer than n_top
        while len(result) < n_top:
            result.append([vocab.sos_idx])

        return result

    def _postprocess_sequence(self, seq: List[int], feat: torch.Tensor, device: torch.device) -> Dict:
        """Decode a chartok_coords token sequence → SMILES + atom coords, then predict bonds.

        Returns a dict with keys: token_ids, decode_smiles, symbols, coords, bond_mat, success.
        """
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        
        atom_indices, atom_counts = extract_atom_indices_from_tokens(seq_tensor, self.vocab)
        atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
        
        hidden_states, _ = self.sequence_decoder(feat, seq_tensor)
        edge_logits = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        edge_preds = _symmetrize_edge_predictions(edge_logits[0])
        
        # Decode sequence to SMILES + atom symbols/coords
        result = self.vocab.sequence_to_smiles(seq)
        
        return {
            'token_ids': seq,
            'decode_smiles': result.get('smiles', ''),
            'symbols': result.get('symbols', []),
            'coords': result.get('coords', []),
            'bond_mat': edge_preds,
            'success': len(result.get('smiles', '')) > 0
        }

    def _greedy_decode_batch(self, feats: torch.Tensor, max_len: int, device: torch.device) -> List[List[int]]:
        """Batched greedy decoding with KV-cached Transformer steps.

        Uses :meth:`SequenceDecoder.forward_step_cached` so that each step
        only processes the **new token** — O(T) per step, O(T²) total —
        instead of re-encoding the full growing sequence.

        Structural constraints (chartok_coords format) are applied via
        :meth:`_apply_constraints_batch`, which operates on the entire
        ``[B, V]`` logits tensor per step with no CPU↔GPU sync.
        """
        B = feats.size(0)
        vocab = self.vocab

        # Pre-flatten encoder features → memory [B, S, d_model]
        memory = feats.flatten(2).permute(0, 2, 1)

        tokens = torch.full((B,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)  # [B, 1]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )  # logits: [B, vocab_size]

            # Batched structural constraints (fully on GPU, no .item() calls)
            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            tokens = torch.argmax(logits, dim=-1)  # [B]

            # For already-finished sequences, emit PAD
            tokens = torch.where(finished,
                                 torch.full_like(tokens, vocab.pad_idx),
                                 tokens)

            finished = finished | (tokens == vocab.eos_idx)
            seqs = torch.cat([seqs, tokens.unsqueeze(1)], dim=1)

        # Convert to list-of-lists, trimming after EOS / removing trailing PAD
        result = []
        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(B):
            seq = seqs[b].tolist()
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)

        return result

    def _sample_decode_batch(
        self,
        feats: torch.Tensor,
        max_len: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """Batched multinomial sampling with KV-cached decoding.

        Mirrors :meth:`_greedy_decode_batch` but uses temperature-scaled
        multinomial sampling instead of argmax.  Uses KV-cached single-token
        steps — O(T) per step, O(T²) total — instead of re-encoding the
        full growing sequence each time.

        Structural constraints (chartok_coords format) are applied via
        :meth:`_apply_constraints_batch`, which operates on the entire
        ``[B, V]`` logits tensor per step with no CPU↔GPU sync.

        Intended for the RL sampling phase (called under ``torch.no_grad()``).

        Args:
            feats: ``[B, d_model, H, W]`` encoded image features.
            max_len: Maximum decoding length.
            device: Compute device.
            temperature: Softmax temperature (>1 → more exploration).

        Returns:
            List of token-id lists (length B), each starting with SOS and
            ending with EOS (or truncated at *max_len*).
        """
        B = feats.size(0)
        vocab = self.vocab

        # Pre-flatten encoder features → memory [B, S, d_model]
        memory = feats.flatten(2).permute(0, 2, 1)

        tokens = torch.full((B,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)                  # [B, 1] running record
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )  # logits: [B, vocab_size]

            # Batched structural constraints (fully on GPU, no .item() calls)
            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            # Temperature-scaled multinomial sampling
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs, 1).squeeze(-1)   # [B]

            tokens = torch.where(
                finished,
                torch.full_like(tokens, vocab.pad_idx),
                tokens,
            )
            finished = finished | (tokens == vocab.eos_idx)
            seqs = torch.cat([seqs, tokens.unsqueeze(1)], dim=1)

        # Convert to list-of-lists, trim to EOS
        result = []
        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(B):
            seq = seqs[b].tolist()
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)

        return result

    def _postprocess_sequences_batch(self, all_seqs: List[List[int]],
                                      img_features: torch.Tensor,
                                      device: torch.device) -> List[Dict]:
        """Batched bond prediction + SMILES reconstruction for decoded sequences.
        
        Pads all sequences, runs one decoder + bond-predictor forward pass,
        then does per-sample CPU post-processing.
        """
        B = len(all_seqs)
        vocab = self.vocab
        
        # Pad sequences to same length
        max_len = max(len(s) for s in all_seqs)
        padded = torch.full((B, max_len), vocab.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(all_seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        
        padding_mask = (padded == vocab.pad_idx)  # True = padded position
        
        # Extract atom indices (already supports batched input)
        atom_indices, atom_counts = extract_atom_indices_from_tokens(padded, vocab)
        max_atoms = int(atom_counts.max().item())
        
        # Batched decoder forward + bond predictor (single GPU pass)
        edge_logits_all = None
        if max_atoms > 0:
            atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
            hidden_states, _ = self.sequence_decoder(img_features, padded, tgt_key_padding_mask=padding_mask)
            edge_logits_all = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        
        # Per-sample CPU post-processing
        results = []
        for b in range(B):
            seq = all_seqs[b]
            result = vocab.sequence_to_smiles(seq)
            
            if edge_logits_all is not None and atom_counts[b] > 0:
                edge_preds = _symmetrize_edge_predictions(edge_logits_all[b])
            else:
                edge_preds = np.zeros((0, 0), dtype=np.int64)
            
            results.append({
                'token_ids': seq,
                'decode_smiles': result.get('smiles', ''),
                'symbols': result.get('symbols', []),
                'coords': result.get('coords', []),
                'bond_mat': edge_preds,
                'success': len(result.get('smiles', '')) > 0,
            })
        
        return results

    @torch.no_grad()
    def generate(self, images: torch.Tensor, beam_size: int = 1, max_len: int = 500, 
                 device: Optional[torch.device] = None) -> List[Dict]:
        """Auto-regressively decode token sequences and predict bond matrices for a batch of images.
        
        When beam_size=1 and B>1, uses batched greedy decoding for much higher
        GPU utilization (~10-30x faster than sequential single-image decoding).
        """
        self.eval()
        
        if device is None:
            device = images.device
        
        B = images.size(0)
        
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        # --- Fast path: batched greedy decode ---
        if beam_size == 1 and B > 1:
            all_seqs = self._greedy_decode_batch(img_features, max_len, device)
            return self._postprocess_sequences_batch(all_seqs, img_features, device)
        
        # --- Fallback: per-image decode (beam search or single image) ---
        results = []
        for b in range(B):
            feat = img_features[b:b+1]
            
            if beam_size == 1:
                seq = self._greedy_decode(feat, max_len, device)
            else:
                seq = self._beam_search_decode(feat, beam_size, max_len, device)
            
            result = self._postprocess_sequence(seq, feat, device)
            results.append(result)
        
        return results

    def _preprocess_tensor(self, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Preprocess tensor input, ensuring correct size and dimensions."""
        img_tensor = img_tensor.to(device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        _, _, h, w = img_tensor.shape
        if (h, w) != self.image_size:
            img_tensor = F.interpolate(img_tensor, size=self.image_size, mode='bilinear', align_corners=False)
        return img_tensor

    def _preprocess_image(self, image_source, device: torch.device) -> torch.Tensor:
        """Preprocess raw image input (path/numpy/PIL)."""
        if isinstance(image_source, str):
            img_pil = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                img_pil = Image.fromarray(image_source).convert('RGB')
            elif image_source.shape[2] == 3:
                img_pil = Image.fromarray(image_source[:, :, ::-1])
            elif image_source.shape[2] == 4:
                img_pil = Image.fromarray(image_source[:, :, :3][:, :, ::-1])
            else:
                raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")
        elif isinstance(image_source, Image.Image):
            img_pil = image_source.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
        
        return self.inference_transforms(image=np.array(img_pil))['image'].unsqueeze(0).to(device)

    def predict(self, image_source, device: Optional[torch.device] = None, beam_size: int = 3, max_len: int = 500,
                return_preprocessed: bool = False, smiles_mode: Optional[str] = None) -> Dict:
        """End-to-end prediction for a single image.
        
        Args:
            image_source: file path, numpy array, PIL Image, or pre-processed tensor.
            smiles_mode: if set, add 'pred_smiles' key using the given mode.
                One of 'decoder', 'graph', 'postprocess', or None (no conversion).
        """
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(image_source, torch.Tensor):
            img_tensor = self._preprocess_tensor(image_source, device)
        else:
            img_tensor = self._preprocess_image(image_source, device)
        
        result = self.generate(images=img_tensor, beam_size=beam_size, max_len=max_len, device=device)[0]
        
        if return_preprocessed:
            result['preprocessed_img'] = img_tensor.squeeze(0).cpu()
        
        if smiles_mode is not None:
            result['pred_smiles'] = _result_to_smiles(result, mode=smiles_mode)
        
        return result

    def predict_batch(self, image_sources: List, 
                      device: Optional[torch.device] = None, 
                      beam_size: int = 3,
                      max_len: int = 500, 
                      smiles_mode: Optional[str] = None) -> List[Dict]:
        """Batch prediction on a single device.
        
        Args:
            image_sources: list of file paths, numpy arrays, PIL Images, or tensors.
            smiles_mode: if set, add 'pred_smiles' key to each result.
                One of 'decoder', 'graph', 'postprocess', or None.
        """
        if device is None:
            if list(self.parameters()):
                device = next(self.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        if isinstance(device, str):
            device = torch.device(device)

        tensors = []
        cpu_device = torch.device('cpu') 
        
        for src in image_sources:
            if isinstance(src, torch.Tensor):
                tensors.append(self._preprocess_tensor(src, cpu_device))
            else:
                tensors.append(self._preprocess_image(src, cpu_device))
        
        if not tensors:
            return []

        img_batch = torch.cat(tensors, dim=0).to(device)
        results = self.generate(images=img_batch, beam_size=beam_size, max_len=max_len, device=device)
        
        if smiles_mode is not None:
            for r in results:
                r['pred_smiles'] = _result_to_smiles(r, mode=smiles_mode)
        
        return results


# ======================== Utility Functions ========================

def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


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

# ======================== Evaluation & Multi-GPU Inference ========================

def _load_benchmark_gt(benchmark_dir: str, csv_path: str,
                       max_samples: Optional[int] = None) -> List[Dict]:
    """Load benchmark CSV, canonicalize GT SMILES, return list of dicts."""
    label_df = pd.read_csv(csv_path)
    if max_samples is not None and max_samples < len(label_df):
        label_df = label_df.sample(n=max_samples, random_state=42)

    data = []
    for _, row in label_df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(benchmark_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        try:
            gt_smi = remove_atom_mapping(row['SMILES'])
            gt_smi, ok = canonicalize_smiles(gt_smi, ignore_cistrans=True)
        except Exception:
            gt_smi, ok = None, False
        data.append({'image_id': img_id, 'img_path': img_path,
                     'gt_smiles': gt_smi, 'gt_ok': ok})
    return data


def _result_to_smiles_decoder(result: Dict) -> Optional[str]:
    """Mode 1: SMILES directly from decoder sequence → canonicalize."""
    if not result or not result.get('success'):
        return None
    try:
        smiles = result.get('decode_smiles', '')
        if not smiles:
            return None
        can_smi, ok = canonicalize_smiles(smiles, ignore_cistrans=True)
        return can_smi if can_smi else None
    except Exception:
        return None


def _result_to_smiles_graph(result: Dict) -> Optional[str]:
    """Mode 2: SMILES entirely reconstructed from predicted atoms + bonds."""
    if not result or not result.get('success'):
        return None
    try:
        symbols = result.get('symbols', [])
        coords = result.get('coords', [])
        bond_mat = result.get('bond_mat')
        if not symbols or not coords or bond_mat is None:
            return None
        smi, _, _, ok = _convert_graph_to_smiles(
            coords=coords, symbols=symbols, edges=bond_mat)
        if not ok or not smi or smi == '<invalid>':
            return None
        smi, ok = canonicalize_smiles(smi, ignore_cistrans=True)
        return smi if ok else None
    except Exception:
        return None


def _result_to_smiles_postprocess(result: Dict) -> Optional[str]:
    """Mode 3 (MolScribe-style): decoder SMILES + postprocessing.
    
    Follows MolScribe's _postprocess_smiles workflow:
      1. Replace R-groups / unknown tokens with isotope-labeled wildcards
      2. Strip stereo from SMILES, build mol (sanitize=False)
      3. Restore chirality/E-Z via _verify_chirality with predicted coords/edges
      4. Expand functional groups back using the mappings
      5. Return canonical SMILES
    """
    if not result or not result.get('success'):
        return None

    smiles = result.get('decode_smiles', '')
    if not isinstance(smiles, str) or smiles == '':
        return None

    coords = result.get('coords', [])
    symbols = result.get('symbols', [])
    edges = result.get('bond_mat')

    try:
        pred_smiles = smiles
        # Step 1: replace R-groups / abbreviations with placeholders
        pred_smiles, mappings = _replace_functional_group(pred_smiles)

        # Step 2: if we have graph info, strip stereo and restore via coordinates
        if coords and symbols and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, edges)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)

        # Step 3: expand functional groups back
        pred_smiles, mol = _expand_functional_group(mol, mappings)

        if pred_smiles and pred_smiles != '<invalid>':
            # Canonicalize (with ignore_cistrans) to match GT preprocessing
            can_smi, ok = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
            return can_smi if ok and can_smi else None
        return None
    except Exception:
        # Fallback: try plain canonicalize of raw decoder SMILES
        try:
            can_smi, ok = canonicalize_smiles(smiles, ignore_cistrans=True)
            return can_smi if ok else None
        except Exception:
            return None


def _result_to_smiles(result: Dict, mode: str = SMILES_MODE_POSTPROCESS) -> Optional[str]:
    """Dispatcher: convert prediction result → canonical SMILES.
    
    Args:
        result: prediction dict with keys 'smiles', 'symbols', 'coords', 'bond_mat', 'success'.
        mode: one of SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS.
    """
    if mode == SMILES_MODE_DECODER:
        return _result_to_smiles_decoder(result)
    elif mode == SMILES_MODE_GRAPH:
        return _result_to_smiles_graph(result)
    elif mode == SMILES_MODE_POSTPROCESS:
        return _result_to_smiles_postprocess(result)
    else:
        raise ValueError(f"Unknown smiles_mode: {mode!r}. "
                         f"Choose from '{SMILES_MODE_DECODER}', '{SMILES_MODE_GRAPH}', '{SMILES_MODE_POSTPROCESS}'.")

def _compute_benchmark_metrics(gt_data: List[Dict],
                               pred_smiles_list: List[Optional[str]],
                               with_records: bool = False,
                               tautomer_standardize: bool = True) -> Dict:
    """Compute exact match accuracy and avg Tanimoto from GT/pred SMILES lists.

    Args:
        gt_data: Ground truth data list.
        pred_smiles_list: Predicted SMILES list.
        with_records: Whether to include per-sample records DataFrame.
        tautomer_standardize: If True, also compute tautomer-normalized exact match.
    """
    exact_match, failed_gt, failed_pred = 0, 0, 0
    tautomer_match = 0
    tanimoto_scores = []
    records = [] if with_records else None

    for gt, pred_smi in zip(gt_data, pred_smiles_list):
        if not gt['gt_ok']:
            failed_gt += 1
            if records is not None:
                records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                                'pred_smiles': None, 'match': False, 'tautomer_match': False, 'tanimoto': 0.0})
            continue
        if pred_smi is None:
            failed_pred += 1
            tanimoto_scores.append(0.0)
            if records is not None:
                records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                                'pred_smiles': None, 'match': False, 'tautomer_match': False, 'tanimoto': 0.0})
            continue

        match = pred_smi == gt['gt_smiles']
        if match:
            exact_match += 1

        # Tautomer-normalized matching
        taut_match = match  # If exact match, no need to check tautomers
        if tautomer_standardize and not match:
            gt_taut, gt_ok = canonicalize_tautomer(gt['gt_smiles'])
            pred_taut, pred_ok = canonicalize_tautomer(pred_smi)
            if gt_ok and pred_ok and gt_taut == pred_taut:
                taut_match = True
        if taut_match:
            tautomer_match += 1

        tan = compute_tanimoto_similarity(gt['gt_smiles'], pred_smi)
        tanimoto_scores.append(tan)
        if records is not None:
            records.append({'image_id': gt['image_id'], 'gt_smiles': gt['gt_smiles'],
                            'pred_smiles': pred_smi, 'match': match, 'tautomer_match': taut_match, 'tanimoto': tan})

    valid = len(gt_data) - failed_gt
    acc = exact_match / valid * 100 if valid > 0 else 0.0
    taut_acc = tautomer_match / valid * 100 if valid > 0 else 0.0
    avg_tan = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0

    out = {'exact_match_acc': acc, 'avg_tanimoto': avg_tan,
           'total': len(gt_data), 'valid': valid, 'failed_predictions': failed_pred}
    if tautomer_standardize:
        out['tautomer_match_acc'] = taut_acc
    if records is not None:
        out['records_df'] = pd.DataFrame(records)
    return out

# ---------- DDP training validation (all ranks participate via all_reduce) ----------

class _ValImageDataset(Dataset):
    """Lightweight dataset for batched validation image loading.

    Returns (image_tensor, index) pairs.  Images that fail to load are
    replaced with zero tensors so that the entire batch is not lost.
    """

    def __init__(self, items: List[Dict], transforms, image_size: Tuple[int, int] = (384, 384)):
        self.items = items
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.items[idx]['img_path']).convert('RGB')
            tensor = self.transforms(image=np.array(img))['image']
        except Exception:
            # Fallback: black image — will produce garbage output, counted as failed
            tensor = torch.zeros(3, *self.image_size)
        return tensor, idx


def validate(
    model: nn.Module,
    benchmark_dir: str,
    benchmark_csv_path: str,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    global_step: int,
    beam_size: int = 1,
    max_samples: Optional[int] = None,
    val_batch_size: int = 128,
    benchmark_name: str = '',
) -> Dict[str, float]:
    """Evaluate on benchmark using ALL DDP ranks; metrics aggregated via all_reduce.

    Uses **mini-batched inference** with parallel image loading for high GPU
    utilisation (~80-95% vs ~20-25% with single-image processing).

    Speed gains come from three layers:
      1. DataLoader with num_workers for overlapped CPU image decoding / transforms.
      2. Batched Swin-B encoder forward pass (128 images at once).
      3. Batched greedy decoding — one decoder step processes all B sequences in
         parallel instead of B separate loops with batch=1.

    Reports accuracy for all three SMILES modes:
      - decoder:     SMILES directly from sequence decoding
      - graph:       SMILES reconstructed from predicted atoms + bonds
      - postprocess: decoder SMILES + chirality correction via coords/edges
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    rank, world_size = get_rank(), get_world_size()

    gt_data = _load_benchmark_gt(benchmark_dir, benchmark_csv_path, max_samples)
    # For DDP training val, skip entries with bad GT to keep things clean
    gt_data = [d for d in gt_data if d['gt_ok']]

    tag = benchmark_name or os.path.splitext(os.path.basename(benchmark_csv_path))[0]
    if is_main_process():
        print(f'\nEvaluating on {tag} ({len(gt_data)} valid, {world_size} GPUs)')
    if not gt_data:
        return {'exact_match_acc': 0.0, 'avg_tanimoto': 0.0, 'valid_samples': 0, 'failed_predictions': 0}

    my_data = gt_data[rank::world_size]

    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]
    # Per-mode counters: exact, failed, tan_sum
    local_stats = {m: [0, 0, 0.0] for m in modes}  # [exact, failed, tan_sum]
    local_count = 0

    # --- Parallel image loading via DataLoader ---
    val_dataset = _ValImageDataset(
        my_data, actual_model.inference_transforms, actual_model.image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        n_batches = len(val_loader)
        it = (tqdm(val_loader, total=n_batches, desc=f'Epoch {epoch} [Val-{tag} R{rank}]')
              if is_main_process() else val_loader)

        for img_batch, indices in it:
            img_batch = img_batch.to(device, non_blocking=True)
            B_cur = img_batch.size(0)
            local_count += B_cur

            # --- Batched forward: encoder + greedy decode + bond predictor ---
            try:
                batch_results = actual_model.generate(
                    images=img_batch, beam_size=beam_size, device=device)
            except Exception:
                batch_results = [None] * B_cur

            # --- Per-sample SMILES evaluation (CPU-bound, ~2-5 ms each) ---
            for local_b in range(B_cur):
                idx = indices[local_b].item()
                gt_smi = my_data[idx]['gt_smiles']
                result = batch_results[local_b]

                for mode in modes:
                    try:
                        pred_smi = _result_to_smiles(result, mode=mode) if result else None
                    except Exception:
                        pred_smi = None
                    if pred_smi is None:
                        local_stats[mode][1] += 1  # failed
                        continue
                    if pred_smi == gt_smi:
                        local_stats[mode][0] += 1  # exact
                    local_stats[mode][2] += compute_tanimoto_similarity(gt_smi, pred_smi)

    # Pack: [count, dec_exact, dec_failed, dec_tan, graph_exact, graph_failed, graph_tan, post_exact, post_failed, post_tan]
    flat = [float(local_count)]
    for mode in modes:
        flat.extend([float(local_stats[mode][0]), float(local_stats[mode][1]), local_stats[mode][2]])
    stats = torch.tensor(flat, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_count = int(stats[0])
    results_out = {}

    if is_main_process():
        print(f'Epoch {epoch} [Val-{tag}] ({total_count} samples):')

    for i, mode in enumerate(modes):
        exact = int(stats[1 + i * 3])
        failed = int(stats[2 + i * 3])
        tan_sum = stats[3 + i * 3].item()
        acc = exact / total_count if total_count else 0.0
        avg_tan = tan_sum / total_count if total_count else 0.0

        results_out[f'{mode}/exact_match_acc'] = acc
        results_out[f'{mode}/avg_tanimoto'] = avg_tan
        results_out[f'{mode}/failed'] = failed

        if is_main_process():
            print(f'  [{mode:11s}] Exact={acc:.4f} ({exact}/{total_count})  '
                  f'Tanimoto={avg_tan:.4f}  Failed={failed}')
            if writer:
                writer.add_scalar(f'Val_{tag}/{mode}_exact_match_acc', acc, global_step)
                writer.add_scalar(f'Val_{tag}/{mode}_avg_tanimoto', avg_tan, global_step)
                writer.add_scalar(f'Val_{tag}/{mode}_failed', failed, global_step)

    # Also store default (postprocess) under canonical keys for backward compat
    results_out['exact_match_acc'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/exact_match_acc', 0.0)
    results_out['avg_tanimoto'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/avg_tanimoto', 0.0)
    results_out['valid_samples'] = total_count
    results_out['failed_predictions'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/failed', 0)

    return results_out

# ---------- Multi-GPU inference via mp.spawn (for standalone / notebook use) ----------

def _inference_worker(rank: int, world_size: int, model: ComoModel,
                      data_paths: List[str], return_dict: dict,
                      beam_size: int = 1, mini_batch_size: int = 128):
    """Worker for mp.spawn — runs inference on one GPU slice."""
    cv2.setNumThreads(0)
    try:
        device = torch.device(f'cuda:{rank}')
        chunk = math.ceil(len(data_paths) / world_size)
        my_paths = data_paths[rank * chunk:(rank + 1) * chunk]
        if not my_paths:
            return
        model.to(device); model.eval()
        results = []
        it = range(0, len(my_paths), mini_batch_size)
        if rank == 0:
            it = tqdm(it, desc=f"GPU-{rank}", total=math.ceil(len(my_paths) / mini_batch_size))
        for i in it:
            results.extend(model.predict_batch(my_paths[i:i + mini_batch_size],
                                               device=device, beam_size=beam_size))
        return_dict[rank] = results
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Rank {rank} failed: {e}")


def predict_multigpu(model: ComoModel, image_paths: List[str],
                     beam_size: int = 1) -> List[Dict]:
    """Distribute inference across all GPUs via mp.spawn. Returns ordered results."""
    world_size = torch.cuda.device_count()
    if world_size == 0:
        return model.predict_batch(image_paths, beam_size=beam_size)

    print(f"Distributing {len(image_paths)} images across {world_size} GPUs...")
    model.cpu(); model.share_memory()
    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(_inference_worker,
                 args=(world_size, model, image_paths, return_dict, beam_size),
                 nprocs=world_size, join=True)
    except Exception as e:
        print(f"mp.spawn failed: {e}"); return []

    results, chunk = [], math.ceil(len(image_paths) / world_size)
    for rank in range(world_size):
        if rank in return_dict:
            results.extend(return_dict[rank])
        else:
            results.extend([{'success': False}] * min(chunk, len(image_paths) - len(results)))
    return results

def evaluate_benchmarks(
    model: ComoModel,
    benchmarks: List[Dict],
    beam_size: int = 1,
    postproc_workers: int = 32,
    tautomer_standardize: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate model on multiple benchmarks using all GPUs (mp.spawn).
    Reports all three SMILES modes (decoder / graph / postprocess) per benchmark.

    Args:
        model: Loaded ComoModel.
        benchmarks: List of dicts, each with keys 'name', 'benchmark_dir', 'csv_path'.
        beam_size: Beam width for decoding (1 = greedy).
        postproc_workers: Thread-pool size for parallel SMILES post-processing.
        tautomer_standardize: If True, also compute tautomer-normalized exact match (default: True).

    Returns:
        Dict mapping benchmark name → {<mode>/exact_match_acc, <mode>/avg_tanimoto, …}
    """
    from concurrent.futures import ProcessPoolExecutor
    all_results = {}
    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]

    for b in benchmarks:
        name = b['name']
        print(f"\n{'='*50}\nBenchmark: {name}\n{'='*50}")

        gt_data = _load_benchmark_gt(b['benchmark_dir'], b['csv_path'])
        image_paths = [d['img_path'] for d in gt_data]
        print(f"  Images: {len(image_paths)}")

        raw_results = predict_multigpu(model, image_paths, beam_size=beam_size)
        min_len = min(len(raw_results), len(gt_data))
        raw_results, gt_data = raw_results[:min_len], gt_data[:min_len]

        benchmark_stats = {}
        for mode in modes:
            print(f"  Post-processing [{mode}] ...")
            converter = partial(_result_to_smiles, mode=mode)
            chunksize = max(1, len(raw_results) // (postproc_workers * 4))
            with ProcessPoolExecutor(max_workers=postproc_workers) as ex:
                pred_smiles = list(tqdm(ex.map(converter, raw_results, chunksize=chunksize),
                                        total=len(raw_results), desc=f"  {mode}"))

            stats = _compute_benchmark_metrics(gt_data, pred_smiles,
                                               with_records=(mode == SMILES_MODE_POSTPROCESS),
                                               tautomer_standardize=tautomer_standardize)
            taut_info = f"  Tautomer: {stats.get('tautomer_match_acc', 0):.2f}%" if tautomer_standardize else ""
            print(f"  [{name}/{mode:11s}] Exact Match: {stats['exact_match_acc']:.2f}%{taut_info} "
                  f"Tanimoto: {stats['avg_tanimoto']:.4f}  Failed: {stats['failed_predictions']}")
            for k, v in stats.items():
                benchmark_stats[f'{mode}/{k}'] = v

        benchmark_stats['total'] = len(gt_data)
        all_results[name] = benchmark_stats

    # Summary table
    taut_header = f"{'Tautomer':>10}" if tautomer_standardize else ""
    print(f"\n{'='*90 if tautomer_standardize else '='*80}")
    print(f"{'Benchmark':<12} {'Mode':<13} {'Exact Match':>12} {taut_header}{'Tanimoto':>10} {'Failed':>8}")
    print(f"{'-'*90 if tautomer_standardize else '-'*80}")
    for name, d in all_results.items():
        for mode in modes:
            acc = d.get(f'{mode}/exact_match_acc', 0)
            taut = d.get(f'{mode}/tautomer_match_acc', 0)
            tan = d.get(f'{mode}/avg_tanimoto', 0)
            fail = d.get(f'{mode}/failed_predictions', 0)
            taut_col = f"{taut:>10.2f}%" if tautomer_standardize else ""
            print(f"{name:<12} {mode:<13} {acc:>11.2f}% {taut_col}{tan:>10.4f} {fail:>8}")
    print(f"{'='*90 if tautomer_standardize else '='*80}")
    return all_results


