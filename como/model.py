"""
COMO Model
==========

The canonical COMO (Chemical Optical Molecular recognitiOn) model.

Architecture::

    ImageEncoder  (Swin / ResNet)  →  2-D positional encoding
    SequenceDecoder (Transformer)  →  chartok_coords token logits + hidden states
    BondPredictor   (MLP)          →  pairwise bond-type logits

End-to-end molecule-image recognition: images → SMILES + atom coordinates + bonds.

This module is designed to be pip-publishable: it contains the model
definition and inference methods only.  Training code lives in the
separate ``training/`` package.
"""

from typing import Dict, List, Optional, Tuple
import os
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from .vocab import ComoVocab
from .image_encoder import ImageEncoder, PositionalEncoding2D
from .sequence_decoder import SequenceDecoder
from .bond_predictor import BondPredictor, _symmetrize_edge_predictions, _symmetrize_edge_predictions_batched
from .augmentations import CropWhiteInference, AdaptiveLongestMaxSize
from .utils import extract_atom_indices_from_tokens, normalize_nodes


__all__ = ['ComoModel']


# Forward reference for _result_to_smiles (imported lazily to avoid circular deps)
def _result_to_smiles(result: Dict, mode: str = 'postprocess') -> Optional[str]:
    """Lazy import wrapper — see ``como.inference`` for the full implementation."""
    from .inference import _result_to_smiles as _impl
    return _impl(result, mode=mode)


class ComoModel(nn.Module):
    """Canonical COMO model for molecule-image recognition.

    Architecture:
      ImageEncoder  (Swin / ResNet)  →  2-D positional encoding
      SequenceDecoder (Transformer)  →  chartok_coords token logits + hidden states
      BondPredictor   (MLP)          →  pairwise bond-type logits

    Provides the full inference pipeline: ``predict()``, ``predict_batch()``,
    ``generate()``, and ``load_model()``.
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
            AdaptiveLongestMaxSize(max_size=self.image_size[0]),
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
        """Forward pass — teacher-forced sequence decoding + bond prediction.

        Args:
            images: ``[B, 3, H, W]`` RGB image tensor.
            tgt_tokens: ``[B, T]`` target token indices.
            tgt_key_padding_mask: ``[B, T]`` padding mask (optional).
            atom_indices: ``[B, N]`` pre-computed atom positions (optional;
                if None, extracted automatically from *tgt_tokens*).
            atom_mask: ``[B, N]`` atom validity mask (optional).
            max_atoms: Maximum atoms per sample (optional).

        Returns:
            (token_logits ``[B, T, V]``,
             edge_logits  ``[B, 7, N, N]``,
             hidden_states ``[B, T, d_model]``)
        """
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
        """Load trained model weights from a checkpoint file.

        Args:
            path: Path to the ``.pth`` or ``.pt`` checkpoint.
            device: Target device (default: current parameter device).
        """
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        
        # Lazy import to avoid hard dependency on training package
        try:
            from training.ddp_utils import is_main_process
            if is_main_process():
                print(f'Model loaded from: {path}')
        except ImportError:
            print(f'Model loaded from: {path}')

    # ======================== Internal Inference Methods ========================

    def predict_step(self, img_features: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits given image features and the sequence decoded so far."""
        _, logits = self.sequence_decoder(
            img_features=img_features,
            tgt_tokens=current_tokens,
            tgt_key_padding_mask=None
        )
        return logits[:, -1, :]

    def _apply_constraints(self, logits: torch.Tensor, last_token: int) -> torch.Tensor:
        """Apply structural constraints for chartok_coords format (single sequence).

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
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits.masked_fill_(mask, float('-inf'))

        elif is_last_y_bin:
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')

        elif last_token == vocab.sos_idx:
            logits[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.eos_idx, vocab.unk_idx]] = float('-inf')

        else:
            logits[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = float('-inf')
            logits[[vocab.pad_idx, vocab.sos_idx, vocab.unk_idx]] = float('-inf')

        return logits

    def _apply_constraints_batch(self, logits: torch.Tensor,
                                 last_tokens: torch.Tensor,
                                 finished: torch.Tensor) -> torch.Tensor:
        """Vectorised chartok_coords constraints for a whole batch.

        Operates on the full ``[B, V]`` logits tensor at once using boolean
        index masking — no CPU↔GPU sync per sample.

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

        is_x = (last_tokens >= vocab.x_bin_start_idx) & (last_tokens <= vocab.x_bin_end_idx)
        is_y = (last_tokens >= vocab.y_bin_start_idx) & (last_tokens <= vocab.y_bin_end_idx)
        is_sos = (last_tokens == vocab.sos_idx)
        is_char = ~is_x & ~is_y & ~is_sos & ~finished

        if is_x.any():
            x_mask = torch.ones(V, dtype=torch.bool, device=logits.device)
            x_mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = False
            logits[is_x] = logits[is_x].masked_fill(x_mask.unsqueeze(0), NEG_INF)

        if is_y.any():
            y_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            y_mask[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            y_mask[vocab.pad_idx] = True
            y_mask[vocab.sos_idx] = True
            y_mask[vocab.unk_idx] = True
            logits[is_y] = logits[is_y].masked_fill(y_mask.unsqueeze(0), NEG_INF)

        if is_sos.any():
            sos_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            sos_mask[vocab.x_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            sos_mask[vocab.pad_idx] = True
            sos_mask[vocab.sos_idx] = True
            sos_mask[vocab.eos_idx] = True
            sos_mask[vocab.unk_idx] = True
            logits[is_sos] = logits[is_sos].masked_fill(sos_mask.unsqueeze(0), NEG_INF)

        if is_char.any():
            char_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            char_mask[vocab.y_bin_start_idx:vocab.y_bin_end_idx + 1] = True
            char_mask[vocab.pad_idx] = True
            char_mask[vocab.sos_idx] = True
            char_mask[vocab.unk_idx] = True
            logits[is_char] = logits[is_char].masked_fill(char_mask.unsqueeze(0), NEG_INF)

        return logits

    def _greedy_decode(self, feat: torch.Tensor, max_len: int, device: torch.device) -> List[int]:
        """Greedy decoding for a single image."""
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
        """Beam search decoding for a single image."""
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
            List of ``n_top`` token-id lists, sorted best-first by cumulative
            log-probability.
        """
        vocab = self.vocab
        beam_size = n_top

        memory = feat.expand(beam_size, -1, -1, -1).flatten(2).permute(0, 2, 1)  # [beam_size, S, d_model]

        tokens = torch.full((beam_size,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)
        scores = torch.zeros(beam_size, device=device)
        finished = torch.zeros(beam_size, dtype=torch.bool, device=device)
        cache = None

        completed: List[Tuple[List[int], float]] = []

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )

            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            log_probs = F.log_softmax(logits, dim=-1)
            V = log_probs.size(-1)

            candidate_scores = scores.unsqueeze(1) + log_probs
            if finished.any():
                candidate_scores[finished] = float('-inf')
                candidate_scores[finished, vocab.pad_idx] = scores[finished]

            flat_scores = candidate_scores.view(-1)
            topk_scores, topk_indices = torch.topk(flat_scores, beam_size)

            beam_indices = topk_indices // V
            token_indices = topk_indices % V

            new_eos = (token_indices == vocab.eos_idx) & ~finished[beam_indices]
            if new_eos.any():
                for k in new_eos.nonzero(as_tuple=False).squeeze(-1).tolist():
                    src_beam = beam_indices[k].item()
                    full_seq = seqs[src_beam].tolist() + [vocab.eos_idx]
                    completed.append((full_seq, topk_scores[k].item()))

            seqs = torch.cat([seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)
            scores = topk_scores
            finished = finished[beam_indices] | (token_indices == vocab.eos_idx)

            new_cache = []
            for layer_cache in cache:
                new_cache.append({
                    'self_k': layer_cache['self_k'][:, :, :, :].index_select(0, beam_indices) if layer_cache['self_k'] is not None else None,
                    'self_v': layer_cache['self_v'][:, :, :, :].index_select(0, beam_indices) if layer_cache['self_v'] is not None else None,
                    'cross_k': layer_cache['cross_k'],
                    'cross_v': layer_cache['cross_v'],
                })
            cache = new_cache

            tokens = token_indices

            if len(completed) >= beam_size:
                best_incomplete = scores[~finished].max().item() if (~finished).any() else float('-inf')
                worst_completed = min(c[1] for c in completed)
                if best_incomplete <= worst_completed:
                    break

        pad, eos = vocab.pad_idx, vocab.eos_idx
        for b in range(beam_size):
            if not finished[b]:
                seq = seqs[b].tolist()
                while seq and seq[-1] == pad:
                    seq.pop()
                completed.append((seq, scores[b].item()))

        completed.sort(key=lambda x: x[1], reverse=True)
        result = []
        for seq, _score in completed[:n_top]:
            if eos in seq:
                seq = seq[:seq.index(eos) + 1]
            while seq and seq[-1] == pad:
                seq.pop()
            result.append(seq)

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

        Uses :meth:`SequenceDecoder.forward_step_cached` for O(T) per step
        instead of re-encoding the full sequence.
        """
        B = feats.size(0)
        vocab = self.vocab

        memory = feats.flatten(2).permute(0, 2, 1)

        tokens = torch.full((B,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )

            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            tokens = torch.argmax(logits, dim=-1)

            tokens = torch.where(finished,
                                 torch.full_like(tokens, vocab.pad_idx),
                                 tokens)

            finished = finished | (tokens == vocab.eos_idx)
            seqs = torch.cat([seqs, tokens.unsqueeze(1)], dim=1)

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

        Uses temperature-scaled multinomial sampling instead of argmax.
        Intended for the RL sampling phase (called under ``torch.no_grad()``).
        """
        B = feats.size(0)
        vocab = self.vocab

        memory = feats.flatten(2).permute(0, 2, 1)

        tokens = torch.full((B,), vocab.sos_idx, dtype=torch.long, device=device)
        seqs = tokens.unsqueeze(1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cache = None

        for step in range(max_len):
            if finished.all():
                break

            logits, cache = self.sequence_decoder.forward_step_cached(
                memory, tokens, step, cache
            )

            last_tokens = seqs[:, -1]
            logits = self._apply_constraints_batch(logits, last_tokens, finished)

            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs, 1).squeeze(-1)

            tokens = torch.where(
                finished,
                torch.full_like(tokens, vocab.pad_idx),
                tokens,
            )
            finished = finished | (tokens == vocab.eos_idx)
            seqs = torch.cat([seqs, tokens.unsqueeze(1)], dim=1)

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
        """Batched bond prediction + SMILES reconstruction for decoded sequences."""
        B = len(all_seqs)
        vocab = self.vocab
        
        max_len = max(len(s) for s in all_seqs)
        padded = torch.full((B, max_len), vocab.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(all_seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        
        padding_mask = (padded == vocab.pad_idx)
        
        atom_indices, atom_counts = extract_atom_indices_from_tokens(padded, vocab)
        max_atoms = int(atom_counts.max().item())
        
        edge_logits_all = None
        if max_atoms > 0:
            atom_mask = torch.arange(atom_indices.size(1), device=device) < atom_counts.unsqueeze(1)
            hidden_states, _ = self.sequence_decoder(img_features, padded, tgt_key_padding_mask=padding_mask)
            edge_logits_all = self.bond_predictor(hidden_states, atom_indices, atom_mask)
        
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

    # ======================== Public Inference API ========================

    @torch.no_grad()
    def generate(self, images: torch.Tensor, beam_size: int = 1, max_len: int = 500, 
                 device: Optional[torch.device] = None) -> List[Dict]:
        """Auto-regressively decode token sequences and predict bond matrices.

        When ``beam_size=1`` and ``B > 1``, uses batched greedy decoding for
        much higher GPU utilization (~10-30x faster than per-image decoding).

        Args:
            images: ``[B, 3, H, W]`` preprocessed image tensor.
            beam_size: Beam width (1 = greedy).
            max_len: Maximum decoding length.
            device: Compute device (default: image device).
        Returns:
            List of dicts with keys: token_ids, decode_smiles, symbols,
            coords, bond_mat, success.
        """
        self.eval()
        
        if device is None:
            device = images.device
        
        B = images.size(0)
        
        img_features = self.image_encoder(images)
        img_features = self.pos_enc_2d(img_features)
        
        # Fast path: batched greedy decode
        if beam_size == 1 and B > 1:
            all_seqs = self._greedy_decode_batch(img_features, max_len, device)
            return self._postprocess_sequences_batch(all_seqs, img_features, device)
        
        # Fallback: per-image decode (beam search or single image)
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
        """Preprocess raw image input (path / numpy / PIL)."""
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

    def predict(self, image_source, device: Optional[torch.device] = None, beam_size: int = 3,
                max_len: int = 500, return_preprocessed: bool = False,
                smiles_mode: Optional[str] = None) -> Dict:
        """End-to-end prediction for a single image.

        Args:
            image_source: File path, numpy array, PIL Image, or pre-processed tensor.
            device: Compute device (default: model's current device).
            beam_size: Beam width for decoding (1 = greedy).
            max_len: Maximum decoding length.
            return_preprocessed: If True, include the preprocessed image tensor.
            smiles_mode: If set, add ``'pred_smiles'`` key:
                ``'decoder'``, ``'graph'``, ``'postprocess'``, or ``None``.
        Returns:
            Prediction dict (see :meth:`generate` for keys).
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
            image_sources: List of file paths, numpy arrays, PIL Images, or tensors.
            device: Compute device.
            beam_size: Beam width (1 = greedy).
            max_len: Maximum decoding length.
            smiles_mode: If set, add ``'pred_smiles'`` key to each result.
        Returns:
            List of prediction dicts.
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
