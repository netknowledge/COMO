"""
COMO Inference Helpers
======================

SMILES post-processing modes and evaluation utilities.

Three SMILES output modes:
  • ``decoder``     – raw SMILES from the decoded token sequence
  • ``graph``       – SMILES reconstructed from predicted atoms + bonds
  • ``postprocess`` – MolScribe-style: replace functional groups, restore stereo
"""

from typing import Dict, List, Optional
import os
import logging

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs, RDLogger

# Suppress RDKit C++ warnings during evaluation (SMILES parse errors, etc.)
RDLogger.DisableLog('rdApp.*')

from .utils import compute_tanimoto_similarity, remove_atom_mapping
from .chemistry import (
    _convert_graph_to_smiles,
    canonicalize_smiles,
    canonicalize_tautomer,
    _replace_functional_group,
    _expand_functional_group,
    _verify_chirality,
)


# ======================== SMILES Mode Constants ========================

SMILES_MODE_DECODER = 'decoder'
SMILES_MODE_GRAPH   = 'graph'
SMILES_MODE_POSTPROCESS = 'postprocess'


# ======================== SMILES Conversion Functions ========================

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
        result: prediction dict with keys 'decode_smiles', 'symbols',
            'coords', 'bond_mat', 'success'.
        mode: one of ``'decoder'``, ``'graph'``, ``'postprocess'``.
    Returns:
        Canonical SMILES string, or None if conversion failed.
    """
    if mode == SMILES_MODE_DECODER:
        return _result_to_smiles_decoder(result)
    elif mode == SMILES_MODE_GRAPH:
        return _result_to_smiles_graph(result)
    elif mode == SMILES_MODE_POSTPROCESS:
        return _result_to_smiles_postprocess(result)
    else:
        raise ValueError(f"Unknown smiles_mode: {mode!r}. "
                         f"Choose from '{SMILES_MODE_DECODER}', "
                         f"'{SMILES_MODE_GRAPH}', '{SMILES_MODE_POSTPROCESS}'.")


# ======================== Benchmark Helpers ========================

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
    Returns:
        Dict with keys: exact_match_acc, avg_tanimoto, total, valid,
        failed_predictions, and optionally tautomer_match_acc, records_df.
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
        taut_match = match
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
