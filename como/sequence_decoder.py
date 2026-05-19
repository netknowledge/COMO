"""
COMO Sequence Decoder
=====================

Autoregressive Transformer Decoder with KV-cache support for efficient
incremental decoding of chartok_coords token sequences.

Public API:
  - ``SequenceDecoder``    — autoregressive decoder with causal masking
  - ``_mha_with_kv_cache`` — multi-head attention with key/value caching
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ======================== KV-Cache Helper ========================

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


# ======================== Sequence Decoder ========================

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
