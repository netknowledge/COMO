"""
Prepare benchmark datasets for training-time validation.

Downloads the pre-packaged benchmark archives (`benchmarks/<name>.tar.gz`)
from the Hugging Face model repo Keylab/COMO and extracts them into
`data/benchmark/real/`, matching the layout the training scripts expect:

    data/benchmark/real/<name>/            (image files)
    data/benchmark/real/<name>.csv         (image_id, file_path, SMILES)

Each tar contains `<name>_corrected.csv`; when no plain `<name>.csv` exists
the corrected CSV is copied to `<name>.csv` so the training scripts
(--val_benchmarks) find it directly.

Benchmarks: CLEF JPO UOB USPTO USPTO-10K Staker ACS WildMol-10K Indigo ChemDraw.

Usage:
    python scripts/prepare_benchmarks.py                     # USPTO + JPO (training defaults)
    python scripts/prepare_benchmarks.py --all               # every benchmark
    python scripts/prepare_benchmarks.py CLEF UOB            # explicit selection
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BASE_URL = "https://huggingface.co/Keylab/COMO/resolve/main/benchmarks"

# HF tar name -> (benchmark key used by --val_benchmarks, benchmark kind)
CANONICAL = {
    'CLEF':       ('CLEF',        'real'),
    'JPO':        ('JPO',         'real'),
    'UOB':        ('UOB',         'real'),
    'USPTO':      ('USPTO',       'real'),
    'USPTO-10K':  ('USPTO-10K',   'real'),
    'WildMol-10K':('WildMol-10K', 'real'),
    'acs':        ('acs',         'real'),
    'staker':     ('staker',      'real'),
    'indigo':     ('indigo',      'synthetic'),
    'chemdraw':   ('chemdraw',    'synthetic'),
}

TRAIN_DEFAULTS = ['USPTO', 'JPO']


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f'  archive already exists: {dest}', )
        return
    print(f'  {url}')
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + '.part')
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)


def extract(tar_path: Path, kind_dir: Path, name: str, key: str) -> None:
    img_dir = kind_dir / key
    if not img_dir.exists():
        print(f'  extracting {tar_path.name} ...')
        subprocess.run(['tar', 'xzf', str(tar_path), '-C', str(kind_dir)], check=True)
    else:
        print(f'  images already extracted: {img_dir}/')

    csv_out = kind_dir / f'{key}.csv'
    if not csv_out.exists():
        # Prefer the conventional `<name>_corrected.csv` shipped in the tar;
        # only then fall back to any nearby csv whose stem starts with <name>.
        exact = kind_dir / f'{name}_corrected.csv'
        candidates = [exact] if exact.exists() else \
            [p for p in sorted(kind_dir.glob(f'{name}*.csv'))]
        if candidates:
            csv_in = candidates[0]
            if csv_in.parent != kind_dir:
                shutil.copy(csv_in, csv_out)
            else:
                shutil.move(str(csv_in), csv_out)
            print(f'  csv: {csv_in.name} -> {csv_out.name}')
        else:
            print(f'  WARNING: no csv found for {name} in the archive')
    else:
        print(f'  csv already present: {csv_out.name}')


def main():
    parser = argparse.ArgumentParser(description='Download benchmark archives for validation')
    parser.add_argument('names', nargs='*', default=TRAIN_DEFAULTS,
                        help='Benchmark names from the HF repo (default: USPTO JPO)')
    parser.add_argument('--all', action='store_true', help='Download every benchmark (~3.5 GB)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <project>/data)')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_dir / 'data'
    out_dir = data_dir / 'benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(CANONICAL) if args.all else args.names
    for name in names:
        entry = CANONICAL.get(name)
        if entry is None:
            print(f'Unknown benchmark: {name} (available: {", ".join(CANONICAL)})', file=sys.stderr)
            sys.exit(2)
        key, kind = entry
        kind_dir = out_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        print(f'[{name}] -> data/benchmark/{kind}/{key}')
        tar_path = kind_dir / f'{name}.tar.gz'
        download(f'{BASE_URL}/{name}.tar.gz', tar_path)
        extract(tar_path, kind_dir, name, key)

    print()
    print('Training scripts reference benchmarks as:')
    print('  --val_benchmarks USPTO JPO   (dir data/benchmark/real/<name>/ + <name>.csv)')


if __name__ == '__main__':
    main()