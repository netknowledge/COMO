"""
Convert MolParser E-SMILES to wild SMILES-like strings for RL fine-tuning.

E-SMILES format:
    backbone_SMILES<sep><a>atom_idx:label</a>...<r>atom_idx:rgroup</r>...

Atom indexing:
    Indices inside <a> tags are 0-based and count atoms in the order they
    appear in the backbone SMILES string.  Only normal hydrogen atoms (no
    isotope) are skipped.  Isotope hydrogens like [2H] (deuterium) are
    counted as regular atoms.

Design rationale:
    • Keep substituents / abbreviations / R-group labels as displayed labels
        instead of collapsing to bare '*'.
    • Do NOT canonicalize here. Canonicalization/legality checks can be handled
        later by reward-specific logic.
    • Output may be illegal for RDKit but can be rendered by Indigo-based
        wild-SMILES drawing.

Conversion rules:
    DROP entries whose tag section contains <r>, </r>, <c>, </c>, or ?
        — these encode structural relationships not representable in SMILES.
    ACCEPT all <a> labels (maximally inclusive):
        1. Pure SMILES (no <a> tags and no '*'): use backbone directly.
        2. <dum> / empty label / no mapping: convert '*' to [R].
        3. Any other label on a '*' atom: keep verbatim as [LABEL].
        4. Chirality and charges on '*' atoms are preserved
           ([*@@] → [Pd@@], [*-] → [R-], [*+] → [label+]).
    Label normalisation: subscript brackets (R[1]→R1, CH[2]→CH2),
    trailing primes (R'→R), and stray '*' chars (R[1*]→R1) are stripped.
    Final output is NOT canonicalized and NOT sanitized.

Outputs:
    sft_real.csv         — filtered (image_id, file_path, SMILES)
    sft_real_esmiles.csv — unfiltered (image_id, file_path, ESMILES)

Usage:
    python scripts/data/convert_esmiles.py      (from the repository root)
"""
import os
import sys
import re
from pathlib import Path

# Add parent directory to path for chemistry module
def _project_root():
    root = Path(__file__).resolve().parent
    while not (root / 'project_naming.py').exists():
        if root.parent == root:
            raise FileNotFoundError('COMO repository root not found from convert_esmiles.py')
        root = root.parent
    return root

_root = _project_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from project_naming import ensure_package_root_on_syspath
ensure_package_root_on_syspath(_root)

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from como.chemistry import _replace_functional_group, _expand_functional_group
from como.constants import ABBREVIATIONS
 
_SUBSCRIPT_RE = re.compile(r'\[(\w+)\]')   # R[1] → R1, CH[2] → CH2


def _normalize_label(label: str) -> str:
    """Normalize an E-SMILES label.

    • Strip stray '*' characters (e.g. R[1*] → R[1]).
    • Collapse subscript brackets (R[1] → R1, CH[2] → CH2).
    • Strip trailing primes (R' → R).
    """
    label = label.replace('*', '')
    label = _SUBSCRIPT_RE.sub(r'\1', label)
    label = label.rstrip("'")
    return label


def _replace_star_labels(backbone: str, a_tags=()):
    """Replace '*' atoms in backbone with [LABEL], keeping order stable.

    For each <a> tag, the labelled '*' atom gets the specified label.
    Any remaining '*' atoms without an <a> mapping default to [R].
    Formal charges on '*' atoms are preserved (e.g. [*-] → [R-]).

    E-SMILES atom indices are 0-based and skip only normal hydrogen atoms
    (H without isotope label).  Isotope hydrogens like [2H] (deuterium)
    are counted in the index.

    The replacement is done by assigning temporary isotopes to target '*'
    atoms, exporting non-canonical SMILES, then using regex to replace
    [isotope*EXTRAS] → [LABELEXTRAS], preserving chirality (@/@@), H-count,
    and charge on the output label.
    """
    mol = Chem.MolFromSmiles(backbone, sanitize=False)
    if mol is None:
        return None, set()

    mol_rw = Chem.RWMol(mol)

    # E-SMILES indices are 0-based and skip only normal hydrogen atoms
    # (AtomicNum==1, no isotope).  Isotope H like [2H] is NOT skipped.
    # Build mapping: E-SMILES index -> RDKit atom index.
    esmiles_to_rdkit = {}
    esmiles_idx = 0
    for rdkit_idx in range(mol_rw.GetNumAtoms()):
        atom = mol_rw.GetAtomWithIdx(rdkit_idx)
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0:
            continue
        esmiles_to_rdkit[esmiles_idx] = rdkit_idx
        esmiles_idx += 1

    # RDKit atom idx -> normalized label (from <a> tags)
    star_labels = {}
    for idx_str, raw_label in a_tags:
        esmiles_idx = int(idx_str)
        rdkit_idx = esmiles_to_rdkit.get(esmiles_idx)
        if rdkit_idx is None:
            return None, set()
        atom = mol_rw.GetAtomWithIdx(rdkit_idx)
        if atom.GetSymbol() != '*':
            continue
        label = _normalize_label(raw_label)
        if label in ('<dum>', 'dum', ''):
            label = 'R'
        star_labels[rdkit_idx] = label

    # Default any unmapped '*' atoms to [R]
    for rdkit_idx in range(mol_rw.GetNumAtoms()):
        atom = mol_rw.GetAtomWithIdx(rdkit_idx)
        if atom.GetSymbol() == '*' and rdkit_idx not in star_labels:
            star_labels[rdkit_idx] = 'R'

    if not star_labels:
        return backbone, set()

    labels_used = set(star_labels.values())

    next_iso = 1001
    iso_to_label = {}
    for idx, label in star_labels.items():
        atom = mol_rw.GetAtomWithIdx(idx)
        atom.SetIsotope(next_iso)
        iso_to_label[next_iso] = label
        next_iso += 1

    out = Chem.MolToSmiles(mol_rw, canonical=False)
    # Replace [isotope*EXTRAS] → [labelEXTRAS], where EXTRAS can include
    # chirality (@/@@), H-count, and/or charge — all preserved on the label.
    for iso in sorted(iso_to_label, reverse=True):
        label = iso_to_label[iso]
        out = re.sub(
            r'\[' + str(iso) + r'\*([^\]]*)\]',
            lambda m, lb=label: f'[{lb}{m.group(1)}]',
            out,
        )
    return out, labels_used


def esmiles_to_smiles(esmiles: str):
    """Convert a single E-SMILES string to wild SMILES-like output.

    E-SMILES format:  backbone<sep><a>idx:label</a>...
    Atom indices inside <a> tags are 0-based and skip only normal hydrogen
    atoms (no isotope).  Isotope hydrogens like [2H] are counted.

    Any '*' atoms in the backbone that lack an <a> mapping are replaced
    with [R] (generic attachment point).

    Returns (smiles | None, category) where category is one of:
    'pure', 'wild', 'drop_tag', 'fail'
    """
    if '<sep>' not in esmiles:
        return None, 'drop_tag'

    backbone, tags_str = esmiles.split('<sep>', 1)
    tags_str = tags_str.strip()

    # Drop structural annotations not representable in SMILES
    if any(tok in tags_str for tok in ('<r>', '</r>', '<c>', '</c>', '?')):
        return None, 'drop_tag'

    # Parse <a> tags (may be empty)
    a_tags = re.findall(r'<a>(\d+):(.*?)</a>', tags_str)

    # If no tags and no wildcards, it's a pure SMILES
    if not a_tags and '*' not in backbone:
        return backbone, 'pure'

    # Replace labelled and unlabelled '*' atoms
    out, labels_used = _replace_star_labels(backbone, a_tags)

    if out is not None:
        # Protect non-abbreviation labels (R-groups, unknown labels) from
        # _replace_functional_group which would strip them to bare '*'.
        # Use rare element Lv (livermorium) with high isotopes as placeholders.
        _PLACEHOLDER_EL = 'Lv'
        _protect_map = {}   # isotope -> original label
        _protect_iso = 900
        for label in sorted(labels_used, key=len, reverse=True):
            if label in ABBREVIATIONS:
                continue  # let the expansion pipeline handle these
            _protect_map[_protect_iso] = label
            out = re.sub(
                r'\[' + re.escape(label) + r'([^\]]*?)\]',
                lambda m, i=_protect_iso: f'[{i}{_PLACEHOLDER_EL}{m.group(1)}]',
                out,
            )
            _protect_iso += 1

        # Expand functional groups (abbreviations only)
        out, mappings = _replace_functional_group(out)
        mol = Chem.MolFromSmiles(out, sanitize=False)
        out, mol = _expand_functional_group(mol, mappings)

        # Restore protected labels
        for iso, label in _protect_map.items():
            out = re.sub(
                r'\[' + str(iso) + re.escape(_PLACEHOLDER_EL) + r'([^\]]*?)\]',
                lambda m, lb=label: f'[{lb}{m.group(1)}]',
                out,
            )
    else:
        return None, 'fail'
    return out, 'wild' if a_tags else 'pure'


def main():
    _root = Path(__file__).resolve().parent
    while not (_root / 'project_naming.py').exists():
        if _root.parent == _root:
            raise FileNotFoundError('COMO repository root not found')
        _root = _root.parent
    data_dir = os.path.join(str(_root), 'data', 'molparser_sft_real')
    output_csv = os.path.join(data_dir, 'sft_real.csv')
    raw_csv = os.path.join(data_dir, 'sft_real_esmiles.csv')

    print("Loading MolParser-7M sft_real from HuggingFace cache...")
    from datasets import load_dataset
    ds = load_dataset("UniParser/MolParser-7M", name="sft_real", split="train")
    print(f"Total entries: {len(ds)}")

    converted = []
    all_entries = []
    counts = {'pure': 0, 'wild': 0, 'drop_tag': 0, 'fail': 0}

    for i, item in enumerate(tqdm(ds, desc="Converting E-SMILES")):
        esmiles = item['SMILES']
        smiles, category = esmiles_to_smiles(esmiles)
        counts[category] += 1

        all_entries.append({
            'image_id': str(i),
            'file_path': f'molparser_sft_real/images/{i}.png',
            'ESMILES': esmiles,
        })

        if smiles is not None:
            converted.append({
                'image_id': str(i),
                'file_path': f'molparser_sft_real/images/{i}.png',
                'SMILES': smiles,
            })

    df_out = pd.DataFrame(converted)
    df_out.to_csv(output_csv, index=False)

    df_raw = pd.DataFrame(all_entries)
    df_raw.to_csv(raw_csv, index=False)

    print(f"\n=== Conversion Summary ===")
    print(f"  Pure SMILES:              {counts['pure']}")
    print(f"  Wild labels kept:         {counts['wild']}")
    print(f"  Dropped (<r>/<c>/?):      {counts['drop_tag']}")
    print(f"  Failed (parse/validate):  {counts['fail']}")
    print(f"  Total output:             {len(df_out)}")
    print(f"  Saved to: {output_csv}")
    print(f"  Raw E-SMILES saved to: {raw_csv}")


if __name__ == '__main__':
    main()

    # Suppress spurious ResourceTracker.__del__ error from multiprocess
    # (known bug in multiprocess used by HuggingFace datasets).
    try:
        from multiprocess.resource_tracker import _resource_tracker
        _resource_tracker._stop = lambda *args, **kwargs: None
    except Exception:
        pass
