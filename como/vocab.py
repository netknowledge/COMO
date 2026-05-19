"""
COMO Vocabulary — chartok_coords tokenizer
==========================================

Canonical vocabulary for the COMO molecule-image recognition model.

Token layout::

    [special] + [SMILES chars] + [X_BIN_0 … X_BIN_{n-1}] + [Y_BIN_0 … Y_BIN_{n-1}]

Coordinates are quantized into *n_bins* bins per axis.
"""

import string
from typing import Dict, List, Optional, Tuple

from SmilesPE.pretokenizer import atomwise_tokenizer


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
