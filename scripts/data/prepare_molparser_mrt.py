"""
Prepare the MolParser-SFT real-image data for MRT training (~83K MRT samples).

Two modes:

1. Download (recommended): fetch the prepared archive from the Hugging Face
   model repo Keylab/COMO (folder `training_data/`) and extract it:

       data/molparser_sft_real/sft_real.csv          (image_id, file_path, SMILES)
       data/molparser_sft_real/sft_real_esmiles.csv  (raw E-SMILES, provenance)
       data/molparser_sft_real/images/<image_id>.png

2. Regenerate from source (`--regenerate`): rebuild the exact dataset from the
   upstream HuggingFace dataset `UniParser/MolParser-7M` (config `sft_real`)
   by converting its E-SMILES labels with the same deterministic converter
   (``convert_esmiles.py``) that produced the released file.

License note: MolParser-7M is provided for non-commercial use only (see
UniParser/MolParser-7M on Hugging Face). The MRT target labels are E-SMILES
with R-group / decoration markers.

Usage:
    python scripts/prepare_molparser_mrt.py
    python scripts/prepare_molparser_mrt.py --data_dir /path/to/data
    python scripts/prepare_molparser_mrt.py --regenerate
"""
import argparse
import sys
from pathlib import Path

HF_BASE = "https://huggingface.co/Keylab/COMO/resolve/main/training_data"
ARCHIVE_NAME = "molparser_sft_real.tar.gz"


def download_prepared(data_dir: Path) -> None:
    out_dir = data_dir / 'molparser_sft_real'
    csv = out_dir / 'sft_real.csv'
    if csv.exists():
        print(f'Already prepared: {csv}')
        return
    import urllib.request
    tar_path = data_dir / ARCHIVE_NAME
    if not tar_path.exists():
        url = f'{HF_BASE}/{ARCHIVE_NAME}'
        print(f'Downloading {url}')
        tmp = tar_path.with_suffix('.tar.gz.part')
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(tar_path)
    print(f'Extracting {tar_path.name} into {data_dir} ...')
    import subprocess
    subprocess.run(['tar', 'xzf', str(tar_path), '-C', str(data_dir)], check=True)
    assert csv.exists(), f'CSV missing after extraction: {csv}'
    print(f'Done: {csv}')


def regenerate(data_dir: Path) -> None:
    import pandas as pd
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_esmiles import esmiles_to_smiles

    out_dir = data_dir / 'molparser_sft_real'
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MolParser-7M sft_real from HuggingFace ...")
    from datasets import load_dataset  # requires `pip install datasets`
    ds = load_dataset("UniParser/MolParser-7M", name="sft_real", split="train")
    print(f'Total entries: {len(ds)}')

    converted, all_entries = [], []
    counts = {'pure': 0, 'wild': 0, 'drop_tag': 0, 'fail': 0}
    for i, item in enumerate(tqdm(ds, desc='Converting E-SMILES')):
        img = item['image']
        img_path = f'molparser_sft_real/images/{i}.png'
        img.convert('RGB').save(img_dir / f'{i}.png')

        esmiles = item['SMILES']
        smiles, category = esmiles_to_smiles(esmiles)
        counts[category] += 1
        all_entries.append({'image_id': str(i), 'file_path': img_path,
                            'ESMILES': esmiles})
        if smiles is not None:
            converted.append({'image_id': str(i), 'file_path': img_path,
                              'SMILES': smiles})

    pd.DataFrame(converted).to_csv(out_dir / 'sft_real.csv', index=False)
    pd.DataFrame(all_entries).to_csv(out_dir / 'sft_real_esmiles.csv', index=False)
    print('\n=== Conversion Summary ===')
    for k, v in counts.items():
        print(f'  {k:24s} {v}')
    print(f'  {"total kept":24s} {len(converted)}')
    print(f'Wrote {out_dir / "sft_real.csv"}')


def main():
    parser = argparse.ArgumentParser(description='Prepare MolParser-SFT MRT data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <project>/data)')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate from UniParser/MolParser-7M instead of '
                             'downloading the prepared archive')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
while not (project_dir / 'project_naming.py').exists():
    if project_dir.parent == project_dir:
        raise FileNotFoundError(
            'COMO repository root not found (project_naming.py missing) from '
            f'{__file__}')
    project_dir = project_dir.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_dir / 'data'

    print('=' * 60)
    print('MolParser-SFT real-image data (MRT training, ~83K)')
    print('=' * 60)
    print(f'Data dir: {data_dir}')

    if args.regenerate:
        regenerate(data_dir)
    else:
        download_prepared(data_dir)

    print('Training scripts read it as data_dir/molparser_sft_real/.')
    print('License: non-commercial use only (MolParser-7M, UniParser on HF).')


if __name__ == '__main__':
    main()