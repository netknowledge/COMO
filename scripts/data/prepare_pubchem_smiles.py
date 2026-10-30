"""
Prepare the PubChem SMILES subset for MLE training.

Downloads the pinned 1M-SMILES subset used by the released COMO recipes
(`pubchem_1M_seed2026.txt`, exported from the full PubChem pool with
seed 2026, matching train_stage1.py / train_joint_mle_mrt.py sampling) from
the Hugging Face model repo Keylab/COMO into `data/pubchem_smiles/`.

The sha256 of the file is verified against the published `.sha256` pin.

Optionally (`--full_pool`) also downloads the complete 122M-SMILES
PubChem pool pickle (`pubchem_smile_list.pkl`, ~7.3 GB) used to regenerate
subsets with export_pubchem_subset.py.

Usage:
    python scripts/prepare_pubchem_smiles.py
    python scripts/prepare_pubchem_smiles.py --data_dir /path/to/data
    python scripts/prepare_pubchem_smiles.py --full_pool
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

BASE_URL = "https://huggingface.co/Keylab/COMO/resolve/main/training_data/pubchem_smiles"
FILES_1M = ["pubchem_1M_seed2026.txt", "pubchem_1M_seed2026.sha256"]
FILE_POOL = "pubchem_smile_list.pkl"
# Pin baked in for extra safety; the downloaded .sha256 file is authoritative.
EXPECTED_SHA256_1M = "9705003a0e2ca3de6568162256659ab7a88556683872ae661d6617440d5c89e8"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists: {dest}")
        return
    print(f"  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    import urllib.request
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)


def verify_sha256(txt_path: Path, sha_path: Path) -> bool:
    if not sha_path.exists():
        print(f"  WARNING: no {sha_path.name}; skipping checksum")
        return True
    expected = sha_path.read_text().split()[0].strip()
    digest = hashlib.sha256(txt_path.read_bytes()).hexdigest()
    ok = (digest == expected) and (digest == EXPECTED_SHA256_1M)
    if digest != EXPECTED_SHA256_1M:
        print(f"  WARNING: digest differs from the baked-in pin "
              f"({digest[:16]}... vs {EXPECTED_SHA256_1M[:16]}...)")
    print(f"  sha256 {'OK' if ok else 'MISMATCH'} ({digest[:16]}... vs {expected[:16]}...)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download PubChem SMILES for COMO MLE training")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <project>/data)')
    parser.add_argument('--full_pool', action='store_true',
                        help='Also download the full 122M-SMILES pool (~7.3 GB, optional)')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_dir / "data"
    out_dir = data_dir / "pubchem_smiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PubChem SMILES preparation (COMO MLE synthetic data)")
    print("=" * 60)
    print(f"Data dir: {data_dir}")

    print("[1/2] Pinned 1M-SMILES subset (train_stage1 / train_joint_mle_mrt --smiles_list)")
    for name in FILES_1M:
        download(f"{BASE_URL}/{name}", out_dir / name)
    ok = verify_sha256(out_dir / FILES_1M[0], out_dir / FILES_1M[1])
    with open(out_dir / FILES_1M[0]) as f:
        n = sum(1 for line in f if line.strip())
    print(f"  SMILES count: {n:,}")

    if args.full_pool:
        print(f"[2/2] Full pool {FILE_POOL} (~7.3 GB)")
        download(f"{BASE_URL}/{FILE_POOL}", out_dir / FILE_POOL)
    else:
        print("[2/2] Full pool skipped (--full_pool to download, ~7.3 GB)")

    print()
    if not ok:
        print("Checksum mismatch — re-download or report the issue.", file=sys.stderr)
        sys.exit(1)
    print("=" * 60)
    print("Done. Training usage:")
    print("  torchrun --nproc_per_node=4 scripts/train_joint_mle_mrt.py \\")
    print("      --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt ...")
    print("=" * 60)


if __name__ == '__main__':
    main()