#!/usr/bin/env python3
"""Normalise R-group representations in a USPTO molecule CSV.

Reads the MOL files referenced by the ``mol_path`` column to determine the
correct visual label for each wildcard atom (``*``) and rewrites SMILES
accordingly:

  - ``[N*]`` → ``[RN]``  (e.g. ``[1*]`` → ``[R1]``)
  - bare ``*`` → ``[<label>]`` where *<label>* comes from the **element
    symbol** in the MOL file's atom block (e.g. ``[X]``, ``[A]``, ``[M]``).
    Different bare ``*`` in the same molecule can receive different labels
    (per-atom mapping via RDKit canonical ordering).
    Falls back to ``[R]`` when the MOL file is unavailable, RDKit cannot
    parse it, or the label is not a valid R-group identifier.

Usage
-----
    python normalise_rgroup.py \\
        --csv  data/uspto_mol/train_680k.csv \\
        --data_dir data \\
        --output data/uspto_mol/train_680k_normalised.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------
def _is_valid_label(label: str) -> bool:
    """True if *label* is a usable R-group label for bracket encoding.

    Accepts alphanumeric strings of length 1-6 whose first character is
    uppercase.  This covers R, A, X, Q, L, M, R1a, R4O, Ar, Hal, etc.
    """
    return (
        1 <= len(label) <= 6
        and label[0].isupper()
        and label.isalnum()
    )


def _clean_element_symbol(sym: str) -> str:
    """Derive a clean R-group label from a MOL-file element symbol.

    Strips trailing non-alphanumeric characters (e.g. ``R5.`` → ``R5``).
    Maps ``*`` and ``R#`` to ``R``.
    """
    if sym in ('*', 'R#'):
        return 'R'
    cleaned = sym.rstrip('.,;:!+-*#')
    return cleaned if cleaned else 'R'


# ---------------------------------------------------------------------------
# MOL file parsing helpers
# ---------------------------------------------------------------------------
def parse_mol_element_symbols(mol_path: str) -> List[str]:
    """Read element symbols from the V2000 MOL file atom block.

    Returns a list of element symbols, one per atom, in MOL-file order
    (0-indexed).
    """
    try:
        with open(mol_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return []
    if len(lines) < 5:
        return []
    parts = lines[3].strip().split()
    try:
        n_atoms = int(parts[0])
    except (ValueError, IndexError):
        return []
    symbols = []
    for i in range(4, min(4 + n_atoms, len(lines))):
        p = lines[i].split()
        if len(p) >= 4:
            symbols.append(p[3])
    return symbols


# ---------------------------------------------------------------------------
# Per-atom mapping via RDKit
# ---------------------------------------------------------------------------
def _get_ordered_bare_labels(
    mol_path: str,
) -> Optional[List[str]]:
    """Determine the label for each bare ``*`` in canonical SMILES order.

    1. Parse the MOL file with RDKit to obtain the molecular graph.
    2. Identify non-numbered wildcard atoms (``atomic_num == 0``,
       ``isotope == 0``).
    3. Look up each atom's element symbol from the MOL-file text.
    4. Generate canonical SMILES and use ``_smilesAtomOutputOrder``
       to establish the correspondence without perturbing the molecule.

    Returns ``None`` on failure, or a list of label strings (one per bare
    ``*``, in the order they appear in canonical SMILES).
    """
    if not _HAS_RDKIT:
        return None

    elem_symbols = parse_mol_element_symbols(mol_path)
    if not elem_symbols:
        return None

    try:
        mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    except Exception:
        return None
    if mol is None:
        return None

    # Collect bare-wildcard atom indices and their labels.
    # A bare ``*`` in canonical SMILES has atomic_num=0, isotope=0,
    # formal_charge=0, no explicit Hs, and no atom map.  Any of these
    # properties would force bracket notation (e.g. ``[*-]``), which
    # ``_count_bare_stars`` does not count.
    bare_info: Dict[int, str] = {}  # rdkit_idx → label
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 0
                and atom.GetIsotope() == 0
                and atom.GetFormalCharge() == 0
                and atom.GetNumExplicitHs() == 0
                and atom.GetAtomMapNum() == 0):
            idx = atom.GetIdx()
            sym = elem_symbols[idx] if idx < len(elem_symbols) else '*'
            label = _clean_element_symbol(sym)
            if not _is_valid_label(label):
                label = 'R'
            bare_info[idx] = label

    if not bare_info:
        return []

    # Use _smilesAtomOutputOrder to determine canonical ordering.
    # Unlike the previous isotope-tagging approach, this does NOT modify
    # the molecule, so the canonical atom traversal order is preserved.
    try:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        Chem.MolToSmiles(mol)
        order_str = mol.GetProp('_smilesAtomOutputOrder')
        atom_order = [
            int(x)
            for x in order_str.strip('[]() ').split(',')
            if x.strip()
        ]
    except Exception:
        return None

    ordered: List[str] = [
        bare_info[mol_idx]
        for mol_idx in atom_order
        if mol_idx in bare_info
    ]
    return ordered


# ---------------------------------------------------------------------------
# Bare-star counting / replacement helpers
# ---------------------------------------------------------------------------
def _count_bare_stars(smiles: str) -> int:
    """Count bare ``*`` (outside ``[…]`` brackets) in *smiles*."""
    count = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            j = smiles.find(']', i + 1)
            i = (j + 1) if j != -1 else (i + 1)
        elif smiles[i] == '*':
            count += 1
            i += 1
        else:
            i += 1
    return count


def _replace_bare_stars(smiles: str, labels: List[str]) -> str:
    """Replace each bare ``*`` in *smiles* with ``[label]`` from *labels*."""
    parts: List[str] = []
    idx = 0
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            j = smiles.find(']', i + 1)
            if j == -1:
                j = len(smiles) - 1
            parts.append(smiles[i:j + 1])
            i = j + 1
        elif smiles[i] == '*':
            lab = labels[idx] if idx < len(labels) else 'R'
            parts.append(f'[{lab}]')
            idx += 1
            i += 1
        else:
            parts.append(smiles[i])
            i += 1
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------
def normalise_rgroup(smiles: str, mol_path: str = None) -> str:
    """Normalise wildcard atoms in *smiles* using the MOL file at *mol_path*.

    Returns the normalised SMILES string.
    """
    if not isinstance(smiles, str):
        return smiles

    # [N*] → [RN]  (numbered wildcard → named R-group)
    smiles = re.sub(r'\[(\d+)\*\]', r'[R\1]', smiles)
    if '*' not in smiles:
        return smiles

    bare_count = _count_bare_stars(smiles)
    if bare_count == 0:
        return smiles

    # Try per-atom mapping via RDKit + MOL element symbols
    if mol_path and os.path.isfile(mol_path):
        ordered = _get_ordered_bare_labels(mol_path)
        if ordered is not None and len(ordered) == bare_count:
            return _replace_bare_stars(smiles, ordered)

    # Fallback: replace all bare * with [R]
    return _replace_bare_stars(smiles, ['R'] * bare_count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Normalise R-group notation in a USPTO molecule CSV.')
    parser.add_argument('--csv', required=True,
                        help='Input CSV file (with file_path, mol_path, SMILES, …)')
    parser.add_argument('--data_dir', default='.',
                        help='Root directory for resolving relative mol_path entries')
    parser.add_argument('--output', required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    print(f'Reading {args.csv} …')
    df = pd.read_csv(args.csv)
    print(f'  {len(df)} rows, columns: {list(df.columns)}')

    has_mol_path = 'mol_path' in df.columns

    n_changed = 0
    new_smiles = []
    for idx in tqdm(range(len(df)), desc='Normalising'):
        row = df.iloc[idx]
        smiles = row['SMILES']
        mol_path = None
        if has_mol_path and isinstance(row['mol_path'], str):
            mol_path = os.path.join(args.data_dir, row['mol_path'])
        result = normalise_rgroup(smiles, mol_path)
        new_smiles.append(result)
        if result != smiles:
            n_changed += 1

    df['SMILES'] = new_smiles
    print(f'\nNormalised {n_changed}/{len(df)} rows '
          f'({100 * n_changed / len(df):.1f}%)')

    df.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
