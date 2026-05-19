import traceback
import multiprocessing
import numpy as np
import itertools

from .constants import RGROUP_SYMBOLS, ABBREVIATIONS, VALENCES, FORMULA_REGEX
from rdkit import Chem, RDLogger

# Suppress RDKit C++ warnings (SMILES parse errors, hydrogen removal, etc.)
RDLogger.DisableLog('rdApp.*')

from SmilesPE.pretokenizer import atomwise_tokenizer

BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 
              2: Chem.rdchem.BondType.DOUBLE, 
              3: Chem.rdchem.BondType.TRIPLE}


def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # skip unreasonable number of atoms
        if num > 100000:
            i += 1; continue
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            if next_num > 100000:
                i += 1; continue
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)


def get_smiles_from_symbol(symbol, bonds):
    """
    Convert symbol (abbrev. or condensed formula) to smiles
    If condensed formula, determine parsing direction and num. bonds on each side using coordinates
    """
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    if len(symbol) > 20:
        return None

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    if success:
        return smiles
    return None


def _replace_functional_group(smiles):
    """Replace R-groups and unknown tokens with isotope-labeled wildcards.
    Returns (smiles, mappings) where mappings maps isotope -> original symbol.
    Ported from MolScribe.
    """
    smiles = smiles.replace('<unk>', 'C')
    for r in RGROUP_SYMBOLS:
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*]
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if token[1:-1] in ABBREVIATIONS or Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def _expand_functional_group(mol, mappings):
    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()
        for i, atom in enumerate(mol_w.GetAtoms()):  # reset radical electrons
            atom.SetNumRadicalElectrons(0)

        atoms_to_remove = []
        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() == '*':
                symbol = Chem.GetAtomAlias(atom)
                isotope = atom.GetIsotope()
                if isotope > 0 and isotope in mappings:
                    symbol = mappings[isotope]
                if not (isinstance(symbol, str) and len(symbol) > 0):
                    continue
                # rgroups do not need to be expanded
                if symbol in RGROUP_SYMBOLS:
                    continue

                bonds = atom.GetBonds()
                sub_smiles = get_smiles_from_symbol(symbol, bonds)

                # create mol object for abbreviation/condensed formula from its SMILES
                mol_r = convert_smiles_to_mol(sub_smiles)

                if mol_r is None:
                    # atom.SetAtomicNum(6)
                    atom.SetIsotope(0)
                    continue

                # remove bonds connected to abbreviation/condensed formula
                adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
                for adjacent_idx in adjacent_indices:
                    mol_w.RemoveBond(i, adjacent_idx)

                adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
                for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                    adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

                # get indices of atoms of main body that connect to substituent
                bonding_atoms_w = adjacent_indices
                # assume indices are concated after combine mol_w and mol_r
                bonding_atoms_r = [mol_w.GetNumAtoms()]
                for atm in mol_r.GetAtoms():
                    if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                        bonding_atoms_r.append(mol_w.GetNumAtoms() + atm.GetIdx())

                # combine main body and substituent into a single molecule object
                combo = Chem.CombineMols(mol_w, mol_r)

                # connect substituent to main body with bonds
                mol_w = Chem.RWMol(combo)
                # if len(bonding_atoms_r) == 1:  # substituent uses one atom to bond to main body
                for atm in bonding_atoms_w:
                    bond_order = mol_w.GetAtomWithIdx(atm).GetNumRadicalElectrons()
                    mol_w.AddBond(atm, bonding_atoms_r[0], order=BOND_TYPES[bond_order])

                # reset radical electrons
                for atm in bonding_atoms_w:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                for atm in bonding_atoms_r:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                atoms_to_remove.append(i)

        # Remove atom in the end, otherwise the id will change
        # Reverse the order and remove atoms with larger id first
        atoms_to_remove.sort(reverse=True)
        for i in atoms_to_remove:
            mol_w.RemoveAtom(i)
        smiles = Chem.MolToSmiles(mol_w)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol


def _verify_chirality(mol, coords, edges, debug=False):
    try:
        n = mol.GetNumAtoms()
        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        Chem.SanitizeMol(mol_tmp)

        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  # List[Tuple[int, any]] -> List[int]

        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)

        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _convert_graph_to_smiles(coords, symbols, edges, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    pred_smiles = '<invalid>'

    try:
        mol = _verify_chirality(mol, coords, edges, debug)
        # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
        # TODO: make sure molblock has the abbreviation information
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, {})
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, mol, success


def convert_graph_to_smiles(coords, symbols, edges, num_workers=16):
    args_zip = zip(coords, symbols, edges)
    if num_workers <= 1:
        results = itertools.starmap(_convert_graph_to_smiles, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles, args_zip, chunksize=128)

    smiles_list, molblock_list, mol_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, mol_list, r_success


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif symbol in RGROUP_SYMBOLS:
                    tokens[j] = '*'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def canonicalize_tautomer(smiles: str) -> tuple:
    """Canonicalize SMILES with tautomer standardization.

    Uses RDKit's TautomerEnumerator to get the canonical tautomer,
    which ensures that different tautomeric forms (e.g., lactam/lactim)
    are normalized to the same canonical representation.

    Args:
        smiles: Input SMILES string.

    Returns:
        Tuple of (canonical_smiles, success_bool).
    """
    if not smiles or not isinstance(smiles, str):
        return '', False
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, False
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)
        canon_smiles = Chem.MolToSmiles(canon_mol)
        return canon_smiles, True
    except Exception:
        return smiles, False