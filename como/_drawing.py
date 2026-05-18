from ._indigo import Indigo
from ._indigo.renderer import IndigoRenderer

import random
import os
import glob
from typing import Dict, Any
import numpy as np
import string
import re
import cv2
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

from func_timeout import func_set_timeout

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from ._constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, ABBREVIATIONS, COLORS
from ._chemistry import _verify_chirality

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_WAVY_BOND_PROB = 0.2
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


# ----------------------Functions for molecule augmentation--------------------------
def add_explicit_hydrogen(mol):
    """
    Randomly make implicit hydrogens explicit in the molecule
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(mol, smiles):
    """
    Randomly replace an implicit H atom with a R-group as a pseudo atom
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    # Optional: Add lowercase to simulate two-letter elements or organic abbreviations
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    # Optional: Add uppercase (less common in standard formulas but adds noise)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    # Recursive group generation (e.g., (COOH)2)
    if random.random() < 0.1:
        # Recursion is fine, but we trust gen_rand_condensed to handle the content
        symb = f'({gen_rand_condensed()})'
    return symb


def get_rand_num():
    """
    Returns a number string or empty string.
    """
    if random.random() < 0.9:
        # 80% chance of empty string (implicit 1) inside this block
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        # 10% chance of double digits
        return '1' + str(random.randint(0, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        symb = get_rand_symb()
        num = get_rand_num()
        
        tokens.append(symb)
        tokens.append(num)
        
        # Check if what we have so far is just a "single atom"
        # It is a single atom if:
        # 1. We are on the first iteration (i==0)
        # 2. No number was generated (num == '')
        # 3. It's not a complex group (no parenthesis in symb)
        is_single_atom = (i == 0) and (num == '') and ('(' not in symb)
        
        if is_single_atom:
            # If it's just a single atom (e.g., "C"), we FORCE the loop 
            # to continue to the next iteration to add another part.
            # This ensures we get at least "CH" or "CBr" instead of just "C".
            continue

        # Standard stop condition:
        # If we are past the first valid chunk, we have an 80% chance to stop.
        if i >= 1 and random.random() < 0.8:
            break
        # If we are at i==0 but we had a number (e.g. "C2"), we can also stop.
        elif i == 0 and not is_single_atom and random.random() < 0.8:
            break

    return ''.join(tokens)


def add_rand_condensed(mol):
    """
    Randomly replace a implicit H atom with random condensed formula as pseudo atoms
    """

    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
            
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol

def add_functional_group(indigo, mol, debug=False):
    ''' Randomly replace functional groups as pseudo atoms with their abbreviations '''

    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol

def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol

def add_wavy_bond(mol, debug=False):
    ''' Randomly replace a wedge bond with wavy bond '''
    atoms = []
    for atom in mol.iterateStereocenters():
        atoms.append(atom)
    if len(atoms) > 0 and (random.random() < INDIGO_WAVY_BOND_PROB or debug):
        atom = random.choice(atoms)
        atom.changeStereocenterType(4)
    return mol

# --------------------------------------------------------------------------------------


# ----------------------Functions for graph and SMILES generation-----------------------
def get_atom_smiles(atom) -> str:
    """
    Generates a descriptive SMILES-like string for an Indigo Atom.
    """
    # To get ALL hydrogens (explicit and implicit)
    try:
        h_count = atom.countImplicitHydrogens()
    except:
        h_count = 0

    symbol = atom.symbol()

    # Check if we can use a simple representation (e.g., "C" instead of "[CH4]")
    is_in_organic_subset = symbol in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', '*']
    
    # An atom needs brackets if it's not in the organic subset, or if it has special properties.
    if not is_in_organic_subset or atom.isotope() or atom.charge() != 0 or h_count > 0:
        # Build the full descriptive string
        isotope_str = str(atom.isotope()) if atom.isotope() else ""
        h_str = f"H{h_count}" if h_count > 1 else ("H" if h_count == 1 else "")
        charge = atom.charge()
        charge_str = f"+{charge}" if charge > 0 else (str(charge) if charge < 0 else "")
        if charge == 1: charge_str = "+"
        if charge == -1: charge_str = "-"
        
        return f"[{isotope_str}{symbol}{h_str}{charge_str}]"
    
    return symbol


def get_graph(mol, image, n_bins=None):
    '''
    Get the graph representation of the molecule.
    '''

    coords, symbols = [], []
    index_map = {}

    # Atom
    atoms = [atom for atom in mol.iterateAtoms()]
    h, w, _ = image.shape
    for i, atom in enumerate(atoms):
        # Coordinates
        x, y = atom.coords()

        if n_bins is not None: # Binning
            x_bin = int((x / w) * n_bins)
            y_bin = int((y / h) * n_bins)
            x_bin = max(0, min(n_bins - 1, x_bin))
            y_bin = max(0, min(n_bins - 1, y_bin))
            coords.append([x_bin, y_bin])
        else: # No binning
            coords.append([x, y])
        # Symbols
        symbols.append(get_atom_smiles(atom))
        index_map[atom.index()] = i

    # Edges
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph


def _graph_to_molblock(graph: Dict[str, Any]) -> str:
    """Convert a molecule graph dict to an RDKit MolBlock string.

    Rebuilds the molecule from symbols / edges / coords via RDKit, verifies
    chirality, and returns the V2000 MolBlock.  Shared by
    ``_augment_graph_and_render`` and reusable by other graph pipelines.
    """
    from rdkit import Chem
    from chemistry import _verify_chirality
    from constants import RGROUP_SYMBOLS, ABBREVIATIONS

    mol = Chem.RWMol()
    symbols = graph['symbols']
    edges = graph['edges']
    coords = graph['coords']
    n = len(symbols)
    ids = []

    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom('*')
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom('*')
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except Exception:
                atom = Chem.Atom('*')
                Chem.SetAtomAlias(atom, symbol)
        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)
        ids.append(mol.AddAtom(atom))

    for i in range(n):
        for j in range(i + 1, n):
            order = int(edges[i][j])
            if order == 0:
                continue
            if order in (1, 5, 6):
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                if order == 5:
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(
                        Chem.BondDir.BEGINWEDGE)
                elif order == 6:
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(
                        Chem.BondDir.BEGINDASH)
            elif order == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif order == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif order == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)

    mol = _verify_chirality(mol, coords, edges, False)
    return Chem.MolToMolBlock(mol)


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        m = re.search(r'\$(.*)\$', part_b)
        part_b = m.group(1) if m else ''
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            # special cases with extension
            smiles = smiles.split(' ')[0]
        return mol, smiles
# --------------------------------------------------------------------------------------


# ---------------------- Functions for Augmentation ------------------------------------
_CJK_RANGES = [
    (0x3000, 0x9FFF),   # CJK Unified, Hiragana, Katakana, punctuation, etc.
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),   # Full-width forms
]

def _has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# CJK font pool — covers common styles found in patents and papers:
#   Sans (黑体/ゴシック): NotoSansCJK {sc,jp} × {Regular,Bold,Light}
#   Serif (宋体/明朝体):  NotoSerifCJK {sc,jp} × {Regular,Bold}
#   Fallback:             DroidSansFallbackFull
_CJK_FONT_SEARCH_DIRS = [
    os.path.expanduser('~/.local/share/fonts'),
    '/usr/share/fonts/truetype/noto',
    '/usr/share/fonts/opentype/noto',
    '/usr/share/fonts/truetype/droid',
]
_CJK_FONT_CANDIDATES = [
    # Sans 黑体 / ゴシック体 — patent headings and labels
    'NotoSansCJKsc-Regular.otf',
    'NotoSansCJKsc-Bold.otf',
    'NotoSansCJKsc-Light.otf',
    'NotoSansCJKjp-Regular.otf',
    'NotoSansCJKjp-Bold.otf',
    # Serif 宋体 / 明朝体 — patent body text
    'NotoSerifCJKsc-Regular.otf',
    'NotoSerifCJKsc-Bold.otf',
    'NotoSerifCJKjp-Regular.otf',
    'NotoSerifCJKjp-Bold.otf',
    # TTC bundles (if installed via system package)
    'NotoSansCJK-Regular.ttc',
    'NotoSansCJK-Medium.ttc',
    # Fallback
    'DroidSansFallbackFull.ttf',
]
_CJK_FONT_GLOB_CANDIDATES = [
    'NotoSansCJK-*.ttc',
    'NotoSerifCJK-*.ttc',
    'DroidSansFallback*.ttf',
]


def _discover_cjk_font_paths() -> list:
    paths = []
    for font_dir in _CJK_FONT_SEARCH_DIRS:
        if not os.path.isdir(font_dir):
            continue
        for name in _CJK_FONT_CANDIDATES:
            path = os.path.join(font_dir, name)
            if os.path.isfile(path) and path not in paths:
                paths.append(path)
        for pattern in _CJK_FONT_GLOB_CANDIDATES:
            for path in sorted(glob.glob(os.path.join(font_dir, pattern))):
                if os.path.isfile(path) and path not in paths:
                    paths.append(path)
    return paths


# Resolved at import time: list of absolute paths that actually exist
_CJK_FONT_PATHS: list = _discover_cjk_font_paths()
if not _CJK_FONT_PATHS:
    print('[drawing_engine] WARNING: no CJK font files found, will fall back to default font')

# Cache: keyed by (path, size) to avoid re-loading the same font object
_CJK_FONT_CACHE: dict = {}
_CJK_FONT_FAILED_PATHS: set = set()


def _get_cjk_font(size: int, randomize: bool = True):
    """Return a CJK-capable PIL font of the given *size*.

    When *randomize* is True (default during training augmentation),
    a font is chosen uniformly at random from the installed pool so that
    the model sees a variety of type-faces.  When False, the first
    available font is returned deterministically.
    """
    if not _CJK_FONT_PATHS:
        return ImageFont.load_default()

    candidates = [path for path in _CJK_FONT_PATHS if path not in _CJK_FONT_FAILED_PATHS]
    if not candidates:
        return ImageFont.load_default()

    if randomize:
        random.shuffle(candidates)

    for path in candidates:
        cache_key = (path, size)
        if cache_key in _CJK_FONT_CACHE:
            return _CJK_FONT_CACHE[cache_key]
        try:
            _CJK_FONT_CACHE[cache_key] = ImageFont.truetype(path, size)
            return _CJK_FONT_CACHE[cache_key]
        except OSError:
            _CJK_FONT_FAILED_PATHS.add(path)

    return ImageFont.load_default()


def _overlay_comment(img: np.ndarray, text: str, font_size: int = 20,
                     alignment: float = 0.5, position: str = 'bottom',
                     offset: int = 10) -> np.ndarray:
    """Draw *text* onto a numpy BGR image using Pillow (supports CJK)."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = _get_cjk_font(font_size)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(offset + th + 6)
    w = int(max(pil_img.width, tw + 10))
    if position == 'top':
        new_img = Image.new('RGB', (w, pil_img.height + pad), (255, 255, 255))
        new_img.paste(pil_img, ((w - pil_img.width) // 2, pad))
        ty = int(offset + 2)
    else:
        new_img = Image.new('RGB', (w, pil_img.height + pad), (255, 255, 255))
        new_img.paste(pil_img, ((w - pil_img.width) // 2, 0))
        ty = int(pil_img.height + offset)
    tx = int((w - tw) * alignment)
    draw = ImageDraw.Draw(new_img)
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)


def _gen_rand_comment():
    """Generate a random comment string mimicking real patent figure captions."""
    category = random.choices(
        ['numbered', 'bracketed', 'compound_name', 'property', 'cjk_label'],
        weights=[3, 3, 2, 1, 3],
        # weights=[0, 0, 0, 0, 1],
    )[0]

    if category == 'numbered':
        templates = [
            lambda: f"{random.randint(1, 20)}{random.choice(string.ascii_letters)}",
            lambda: f"Fig. {random.randint(1, 30)}",
            lambda: f"Fig.{random.randint(1, 30)}",
            lambda: f"Compound {random.randint(1, 50)}",
            lambda: f"Structure {random.randint(1, 20)}",
            lambda: random.choice(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII',
                                   'VIII', 'IX', 'X', 'XI', 'XII']),
            lambda: f"Scheme {random.randint(1, 10)}",
            lambda: f"Formula {random.randint(1, 20)}",
            lambda: f"Ex. {random.randint(1, 50)}",
            lambda: f"Example {random.randint(1, 50)}",
            lambda: f"Table {random.randint(1, 10)}",
            lambda: f"No. {random.randint(1, 100)}",
        ]
        return random.choice(templates)()

    if category == 'bracketed':
        core = random.choice([
            str(random.randint(1, 50)),
            random.choice(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']),
            f"{random.randint(1, 20)}{random.choice('abcdef')}",
            f"{random.choice(['I', 'II', 'III'])}-{random.choice('abcdef')}",
        ])
        l, r = random.choice([
            ('(', ')'), ('[', ']'), ('（', '）'), ('【', '】'), ('<', '>'),
        ])
        prefix = random.choice(['', '', '', 'Compound ', 'Formula ', 'Cpd. ',
                                '化合物', '式', '実施例', '实施例'])
        return f"{prefix}{l}{core}{r}"

    if category == 'compound_name':
        en_names = [
            'Aspirin', 'Ibuprofen', 'Paracetamol', 'Metformin', 'Atorvastatin',
            'Omeprazole', 'Amlodipine', 'Losartan', 'Cetirizine', 'Diclofenac',
            'Naproxen', 'Captopril', 'Lisinopril', 'Simvastatin', 'Clopidogrel',
            'Erlotinib', 'Sorafenib', 'Gefitinib', 'Dasatinib', 'Nilotinib',
        ]
        jp_names = [
            'アスピリン', 'イブプロフェン', 'メトホルミン', 'オメプラゾール',
            'アムロジピン', 'ロサルタン', 'セチリジン', 'ジクロフェナク',
            'カプトプリル', 'シンバスタチン', 'エルロチニブ', 'ゲフィチニブ',
        ]
        cn_names = [
            '阿司匹林', '布洛芬', '对乙酰氨基酚', '二甲双胍', '阿托伐他汀',
            '奥美拉唑', '氨氯地平', '氯沙坦', '西替利嗪', '双氯芬酸',
            '卡托普利', '辛伐他汀', '厄洛替尼', '吉非替尼', '达沙替尼',
        ]
        return random.choice(en_names + jp_names + cn_names)

    if category == 'property':
        en_props = [
            f"m.p. {random.randint(50, 300)} C",
            f"MW: {random.randint(100, 800)}",
            f"b.p. {random.randint(30, 250)} C",
            f"yield: {random.randint(10, 99)}%",
            f"purity: {random.randint(90, 99)}.{random.randint(0, 9)}%",
            f"IC50 = {random.randint(1, 500)} nM",
            f"Ki = {random.randint(1, 100)} nM",
        ]
        jp_props = [
            f"融点: {random.randint(50, 300)} C",
            f"分子量: {random.randint(100, 800)}",
            f"沸点: {random.randint(30, 250)} C",
            f"収率: {random.randint(10, 99)}%",
        ]
        cn_props = [
            f"熔点: {random.randint(50, 300)} C",
            f"分子量: {random.randint(100, 800)}",
            f"沸点: {random.randint(30, 250)} C",
            f"收率: {random.randint(10, 99)}%",
        ]
        return random.choice(en_props + jp_props + cn_props)

    # cjk_label
    jp_labels = [
        f"化合物{random.randint(1, 50)}",
        f"式({random.choice(['I', 'II', 'III', 'IV', 'V'])})",
        f"構造式{random.randint(1, 20)}",
        f"実施例{random.randint(1, 30)}",
        f"参考例{random.randint(1, 20)}",
        '一般式', '中間体',
    ]
    cn_labels = [
        f"化合物{random.randint(1, 50)}",
        f"式({random.choice(['I', 'II', 'III', 'IV', 'V'])})",
        f"结构式{random.randint(1, 20)}",
        f"实施例{random.randint(1, 30)}",
        f"参考例{random.randint(1, 20)}",
        '通式', '中间体',
    ]
    return random.choice(jp_labels + cn_labels)


def _generate_style_config(mol, default_drawing_style=False):
    """
    Generate drawing style configuration dict for Indigo rendering.
    Returns a dict with all render options that can be applied to any Indigo session.
    """
    indigo_option_config: dict = {
        'render-output-format': 'png',
        'render-background-color': '1,1,1',
        'render-stereo-style': 'none',
        'render-label-mode': 'hetero',
        'render-font-family': 'Arial',
        'render-valences-visible': False,
    }

    mol_config: dict = {
        'mol.dearomatize': False,
    }

    if not default_drawing_style:
        thickness = random.uniform(0.5, 2)
        indigo_option_config['render-relative-thickness'] = thickness
        indigo_option_config['render-bond-line-width'] = random.uniform(1, 4 - thickness)
        
        if random.random() < 0.5:
            indigo_option_config['render-font-family'] = random.choice(['Arial', 'Times', 'Courier', 'Helvetica'])
        
        indigo_option_config['render-label-mode'] = random.choice(['hetero', 'terminal-hetero'])
        indigo_option_config['render-implicit-hydrogens-visible'] = random.choice([True, False])
        
        if random.random() < 0.1:
            indigo_option_config['render-stereo-style'] = 'old'
        if random.random() < 0.2:
            indigo_option_config['render-atom-ids-visible'] = True
        
        if random.random() < INDIGO_COMMENT_PROB:
            comment_text = _gen_rand_comment()
            comment_font_size = random.randint(40, 60)
            comment_alignment = random.choice([0, 0.5, 1])
            comment_position = random.choice(['top', 'bottom'])
            comment_offset = random.randint(2, 30)
            if _has_cjk(comment_text):
                mol_config['comment_overlay'] = {
                    'text': comment_text,
                    'font_size': comment_font_size,
                    'alignment': comment_alignment,
                    'position': comment_position,
                    'offset': comment_offset,
                }
            else:
                indigo_option_config['render-comment'] = comment_text
                indigo_option_config['render-comment-font-size'] = comment_font_size
                indigo_option_config['render-comment-alignment'] = comment_alignment
                indigo_option_config['render-comment-position'] = comment_position
                indigo_option_config['render-comment-offset'] = comment_offset

        if random.random() < INDIGO_DEARMOTIZE_PROB: # The way aromatic rings are represented
            mol_config['mol.dearomatize'] = True
        else:
            mol_config['mol.dearomatize'] = False

    return mol, indigo_option_config, mol_config


def _apply_style_config(indigo, indigo_option_config, mol, mol_config):
    """Apply style configuration dict to an Indigo session."""
    for key, value in indigo_option_config.items():
        indigo.setOption(key, value)
    if mol_config['mol.dearomatize']:
        mol.dearomatize()
    else:
        mol.aromatize()
# --------------------------------------------------------------------------------------


# -------------------------------- Drawing Functions -----------------------------------
def _blank_image() -> np.ndarray:
    """Return a small blank white image as fallback."""
    return np.full((10, 10, 3), 255, dtype=np.uint8)


@func_set_timeout(5)  # Set timeout of 5 seconds
def generate_image_from_smiles(smiles, n_bins=None,
                               mol_augment=True, include_condensed=True, 
                               default_drawing_style=False, 
                               debug=False):
    '''
    Generate molecule image and graph from SMILES string.
    Args:
        smiles: SMILES string
        n_bins: int or None, number of bins for coordinate binning
        mol_augment: bool, whether to perform molecule augmentation
        include_condensed: bool, whether to include condensed formula in augmentation. Only used if mol_augment is True.
        default_drawing_style: bool, whether to use default drawing style
        debug: If True, raise exceptions instead of returning fallback
    Returns:
        img: np.ndarray (H, W, 3) Color image
        smiles: SMILES string (possibly modified by augmentation)
        graph: dict with 'symbols', 'edges', 'coords'
        success: bool, whether the generation was successful
        style_config: dict, drawing style configuration used
        mol_config: dict, molecule configuration used
    '''
    indigo = None
    renderer = None
    style_config, mol_config = None, None

    try:
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)

        mol = indigo.loadMolecule(smiles)
        # if mol.countAtoms() > 100:  # Skip huge molecules
        #     raise ValueError("Molecule too large")

        # Drawing Style Augmentation
        mol, style_config, mol_config = _generate_style_config(mol, default_drawing_style)
        _apply_style_config(indigo, style_config, mol, mol_config)

        # Molecule Augmentation
        if mol_augment:
            smiles = mol.canonicalSmiles()
            mol = add_explicit_hydrogen(mol)
            mol = add_rgroup(mol, smiles)
            if include_condensed:
                mol = add_rand_condensed(mol)
            mol = add_functional_group(indigo, mol, debug)
            mol = add_wavy_bond(mol, debug=debug)
            mol = add_color(indigo, mol)
            mol, smiles = generate_output_smiles(indigo, mol)
        
        mol.layout()
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1) # BGR format
        if style_config and 'comment_overlay' in mol_config:
            co = mol_config['comment_overlay']
            img = _overlay_comment(img, **co)
        graph = get_graph(mol, img, n_bins=n_bins)
        success = True
    except Exception:
        if debug:
            raise
        img = _blank_image()
        graph = {}
        success = False

    return img, smiles, graph, success, style_config, mol_config


def _relabel_wild_atoms(indigo, mol):
    """Relabel wildcard atoms (symbol 'A') to R-group pseudo-atoms.

    Indigo loads ``[N*]`` as atom with symbol='A' and isotope=N.
    This helper converts them to proper pseudo-atom labels:
        isotope > 0  →  RN   (e.g. [1*] → R1)
        isotope == 0 →  R    (bare *)

    Returns a new Indigo molecule with correct pseudo-atom labels.
    """
    # 1. Collect isotopes of 'A' atoms in iteration order
    a_isotopes = []
    for atom in mol.iterateAtoms():
        if atom.symbol() == 'A':
            a_isotopes.append(atom.isotope())

    if not a_isotopes:
        return mol  # No wildcards to relabel

    # 2. Get extended SMILES and parse the label block
    smi = mol.smiles()
    parts = smi.split('|')
    if len(parts) < 2:
        return mol  # No extended block

    label_block = parts[1].strip().strip('$')
    labels = label_block.split(';')

    # 3. Replace 'A' labels with R-group labels
    a_idx = 0
    for i, label in enumerate(labels):
        if label == 'A':
            isotope = a_isotopes[a_idx]
            labels[i] = f'R{isotope}' if isotope > 0 else 'R'
            a_idx += 1

    # 4. Reconstruct extended SMILES and reload
    new_smi = parts[0] + '|$' + ';'.join(labels) + '$|'
    return indigo.loadMolecule(new_smi)


@func_set_timeout(5)  # Set timeout of 5 seconds
def generate_image_from_wild_smiles(smiles, n_bins=None,
                                    default_drawing_style=False,
                                    debug=False):
    '''
    Generate molecule image and graph from wild SMILES (containing * wildcards).
    Bare * is rendered/labeled as R; [N*] (isotope-labeled) is rendered as RN.
    No molecule-level augmentation (no add_rgroup, add_condensed, etc.).

    Args:
        smiles: Wild SMILES string (e.g. "[1*]N([2*])C1CCC(...)CC1")
        n_bins: int or None, number of bins for coordinate binning
        default_drawing_style: bool, whether to use default drawing style
        debug: If True, raise exceptions instead of returning fallback
    Returns:
        Same 6-tuple as generate_image_from_smiles:
        (img, output_smiles, graph, success, style_config, mol_config)
    '''
    indigo = None
    renderer = None
    style_config, mol_config = None, None

    try:
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)

        mol = indigo.loadMolecule(smiles)

        # Relabel wildcard atoms (* → R, [N*] → RN) as proper pseudo-atoms
        mol = _relabel_wild_atoms(indigo, mol)

        # Drawing Style Augmentation (visual diversity only)
        mol, style_config, mol_config = _generate_style_config(mol, default_drawing_style)
        _apply_style_config(indigo, style_config, mol, mol_config)

        # Color augmentation (doesn't change molecular structure)
        mol = add_color(indigo, mol)

        # Generate output SMILES with R-group labels ([R], [R1], ...)
        mol, output_smiles = generate_output_smiles(indigo, mol)

        mol.layout()
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # BGR format
        if style_config and 'comment_overlay' in mol_config:
            co = mol_config['comment_overlay']
            img = _overlay_comment(img, **co)
        graph = get_graph(mol, img, n_bins=n_bins)
        success = True
    except Exception:
        if debug:
            raise
        img = _blank_image()
        graph = {}
        output_smiles = smiles
        success = False

    return img, output_smiles, graph, success, style_config, mol_config


@func_set_timeout(5)  # Set timeout of 5 seconds
def generate_image_from_graph(graph,
                              style_config=None, mol_config=None, default_drawing_style=True,
                              debug=False):
    """
    Build molecule via RDKit from graph (to preserve solid/dashed wedges) and render using Indigo.
    This function is used for the cycle consistency check during training.
    
    Args:
        graph: Graph dict with 'symbols', 'edges', 'coords'
        style_config: dict, drawing style config for Indigo
        mol_config: dict, molecule config for Indigo
        debug: If True, raise exceptions instead of returning fallback
    
    Returns:
        img_pred: np.ndarray (H, W, 3) Color image
        style_config: dict, drawing style configuration used
        mol_config: dict, molecule configuration used
        success: bool, whether the generation was successful
    """
    try:
        mol = Chem.RWMol()
        symbols = graph['symbols']
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

        edges = graph['edges']
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

        coords = graph['coords']
        mol = _verify_chirality(mol, coords, edges, debug) # Atom coodinates are set here
        mol_block = Chem.MolToMolBlock(mol)

        indigo = Indigo()
        renderer = IndigoRenderer(indigo)

        mol = indigo.loadMolecule(mol_block)

        if style_config is None or mol_config is None: # Generate style config if not provided
            mol, style_config, mol_config = _generate_style_config(mol, default_drawing_style)
        _apply_style_config(indigo, style_config, mol, mol_config)

        mol.layout()
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # 1 flag for color
        if mol_config and 'comment_overlay' in mol_config:
            co = mol_config['comment_overlay']
            img = _overlay_comment(img, **co)
        success = True
    except Exception:
        if debug:
            raise
        img = _blank_image()
        success = False
    return img, style_config, mol_config, success
# --------------------------------------------------------------------------------------