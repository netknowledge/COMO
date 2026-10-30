"""
Download the MolDiscriminator checkpoint used as the frozen visual-reward
judge for COMO's visual MRT arm (paper §II.1, §II.2 Eq. 3).

The released COMO_joint/visual checkpoints on Hugging Face were trained with
`--visual_reward_encoder <this checkpoint>`; see doc/training.md §7.

Architecture: SiameseNetwork (EfficientNet-V2-S backbone, 128-d
L2-normalized projection head), symmetric InfoNCE (tau=0.07), CReM hard
negatives — re-trainable with `scripts/train/train_MolDiscriminator.py`.

Usage:
    python scripts/data/prepare_mol_discriminator.py
    python scripts/data/prepare_mol_discriminator.py --data_dir /path/to/data
"""
import argparse
import sys
from pathlib import Path

HF_BASE = "https://huggingface.co/Keylab/COMO/resolve/main/training_data"
CKPT_NAME = "20260321_final.pth"
EXPECTED_SIZE = 85_187_306  # bytes; sanity check


def main():
    parser = argparse.ArgumentParser(
        description='Download the MolDiscriminator visual-reward encoder checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <project>/data)')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    while not (project_dir / 'project_naming.py').exists():
        if project_dir.parent == project_dir:
            raise FileNotFoundError('COMO repository root not found')
        project_dir = project_dir.parent
    out_dir = Path(args.data_dir) if args.data_dir else project_dir / 'data'
    out_dir = out_dir / 'mol_discriminator'
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / CKPT_NAME
    if not dest.exists():
        import urllib.request
        url = f'{HF_BASE}/mol_discriminator/{CKPT_NAME}'
        print(f'Downloading {url}')
        tmp = dest.with_suffix('.pth.part')
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(dest)

    size = dest.stat().st_size
    print(f'Checkpoint: {dest} ({size / 1e6:.1f} MB)')
    if size != EXPECTED_SIZE:
        print(f'WARNING: unexpected size (expected {EXPECTED_SIZE} bytes) — '
              're-download or report an issue.', file=sys.stderr)
        sys.exit(1)

    print('Usage:')
    print('  torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \\')
    print('      --reward_mode visual \\')
    print(f'      --visual_reward_encoder data/mol_discriminator/{CKPT_NAME} ...')


if __name__ == '__main__':
    main()