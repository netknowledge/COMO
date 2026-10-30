"""
Download and prepare USPTO Mol data for Stage 2 training.

Downloads uspto_mol.zip from HuggingFace and extracts it.  The ZIP contains
patent molecule images and a CSV with file_path, SMILES, node_coords, and
edges columns.  These are used directly by ``USPTOMolDataset`` during
training.

Usage:
    python scripts/prepare_uspto_mol.py
    python scripts/prepare_uspto_mol.py --data_dir /path/to/data
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd


def download_uspto_mol(data_dir: str) -> str:
    """Download uspto_mol.zip from HuggingFace if not already present."""
    zip_path = os.path.join(data_dir, "uspto_mol.zip")
    if os.path.exists(zip_path):
        print(f"  ZIP already exists: {zip_path}")
        return zip_path

    url = "https://huggingface.co/yujieq/MolScribe/resolve/main/uspto_mol.zip"
    print(f"  Downloading from: {url}")
    print(f"  Saving to: {zip_path}")

    import urllib.request
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Download complete: {os.path.getsize(zip_path) / 1e9:.2f} GB")
    return zip_path


def extract_zip(zip_path: str, extract_dir: str) -> str:
    """Extract the zip file to data directory."""
    uspto_dir = os.path.join(extract_dir, "uspto_mol")
    if os.path.isdir(uspto_dir):
        print(f"  Already extracted: {uspto_dir}/")
        return uspto_dir

    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"  Extracted to: {extract_dir}")
    return uspto_dir


def inspect_data(uspto_dir: str, data_dir: str):
    """Find the training CSV and print a summary."""
    csv_candidates = [
        os.path.join(uspto_dir, "train_680k.csv"),
        os.path.join(uspto_dir, "train.csv"),
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        csvs = [f for f in os.listdir(uspto_dir) if f.endswith('.csv')]
        if csvs:
            csv_path = os.path.join(uspto_dir, csvs[0])
        else:
            print("WARNING: No CSV file found in USPTO directory!", file=sys.stderr)
            print(f"  Contents of {uspto_dir}:")
            for item in sorted(os.listdir(uspto_dir)):
                print(f"    {item}")
            return

    print(f"  Training CSV: {csv_path}")
    df = pd.read_csv(csv_path, nrows=5)
    print(f"  Columns: {list(df.columns)}")
    total = len(pd.read_csv(csv_path, usecols=[0]))
    print(f"  Total rows: {total}")

    # Verify a sample image exists
    if 'file_path' in df.columns:
        sample_path = df['file_path'].iloc[0]
        # Try both absolute and relative to data_dir
        if os.path.isfile(sample_path):
            print(f"  Sample image OK: {sample_path}")
        elif os.path.isfile(os.path.join(data_dir, sample_path)):
            print(f"  Sample image OK: {os.path.join(data_dir, sample_path)}")
        else:
            print(f"  WARNING: Sample image not found: {sample_path}")
            print(f"    (tried relative to {data_dir})")

    # Show expected columns
    expected = {'file_path', 'SMILES', 'node_coords'}
    present = set(df.columns) & expected
    missing = expected - present
    if missing:
        print(f"  WARNING: Missing expected columns: {missing}")
    if 'edges' in df.columns:
        print("  Edge data: present")
    else:
        print("  Edge data: NOT present (bond predictor will train without GT edges)")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare USPTO Mol data")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <project>/data)')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    while not (project_dir / 'project_naming.py').exists():
        if project_dir.parent == project_dir:
            raise FileNotFoundError('COMO repository root not found')
        project_dir = project_dir.parent
    data_dir = args.data_dir or str(project_dir / "data")

    print("=" * 60)
    print("USPTO Mol Data Preparation")
    print("=" * 60)
    print(f"Project dir: {project_dir}")
    print(f"Data dir:    {data_dir}")
    print()

    # Step 1: Download
    print("[1/3] Downloading USPTO Mol data...")
    zip_path = download_uspto_mol(data_dir)
    print()

    # Step 2: Extract (keeps images + CSV intact)
    print("[2/3] Extracting ZIP...")
    uspto_dir = extract_zip(zip_path, data_dir)
    print()

    # Step 3: Inspect
    print("[3/3] Inspecting data...")
    inspect_data(uspto_dir, data_dir)
    print()

    print("=" * 60)
    print(f"Done! USPTO data ready at: {uspto_dir}/")
    print("Images and CSV are used directly by USPTOMolDataset.")
    print("=" * 60)


if __name__ == '__main__':
    main()
