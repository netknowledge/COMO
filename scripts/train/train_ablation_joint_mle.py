"""
Joint Training Ablation: Interleaved MLE (No MRT)
==================================================

Controlled ablation for the joint MLE+MRT pipeline.  Replaces MRT
steps with interleaved MLE steps on MolParser data, keeping
everything else identical: same cadence (_mrt_every), loss weight
(mrt_weight=0.1), warmup delay (mrt_warmup_epochs=5), and batch
size (mrt_batch_size=16).

The ONLY difference from joint MRT training is the loss function:
  - Joint MRT: sampling-based Minimum Risk Training loss
  - This ablation: teacher-forced MLE (cross-entropy + bond) loss

This isolates the effect of the MRT training mechanism from the
additional MolParser training data.

Methodology:
  - Same MLE backbone as joint training: 1M PubChem (rendered) + 680K
    USPTO (real images).
  - MolParser SMILES (~83K) are trained via a SEPARATE interleaved
    DataLoader (not merged into the main ConcatDataset), matching
    the MRT interleaving schedule exactly.
  - All hyperparameters (LR, epochs, batch size, architecture) match
    the joint training script (train_joint_mle_mrt.py).

Comparison:
  Joint (train_joint_mle_mrt.py):
    MLE on {PubChem+USPTO} + interleaved MRT on {MolParser real images}
  Ablation (this script):
    MLE on {PubChem+USPTO+MolParser rendered}  (no MRT)

Usage:
    # Default: joint MLE data mode
    torchrun --nproc_per_node=4 scripts/train_ablation_joint_mle.py

    # Synthetic-only MLE (no USPTO)
    torchrun --nproc_per_node=4 scripts/train_joint_ablation_mle.py --data_mode synthetic

    # Resume
    torchrun --nproc_per_node=4 scripts/train_joint_ablation_mle.py --resume
"""
import sys
import os
from pathlib import Path
import argparse
import pickle

import pandas as pd

# Locate the repository root (walks up; works for the nested COMO_dev layout
# and the flat public-twin layout alike) and put it first on sys.path.
project_dir = Path(__file__).resolve().parent
while not (project_dir / 'project_naming.py').exists():
    if project_dir.parent == project_dir:
        raise FileNotFoundError(
            'COMO repository root not found (project_naming.py missing) from '
            f'{__file__}')
    project_dir = project_dir.parent
sys.path.insert(0, str(project_dir))

from project_naming import (
    ensure_package_root_on_syspath,
    resolve_artifact_dir,
    resolve_checkpoint_path,
)

ensure_package_root_on_syspath(project_dir)

from training.train_joint import train_joint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint training ablation: MLE-only (no MRT)')

    # ===== Data mode =====
    parser.add_argument('--data_mode', type=str, default='joint',
                        choices=['joint', 'synthetic'],
                        help='MLE data mode (default: joint)')
    parser.add_argument('--synthetic_num', type=int, default=1_000_000,
                        help='PubChem SMILES to sample (default: 1M)')
    parser.add_argument('--smiles_list', type=str, default=None,
                        help='Plain-text PubChem SMILES subset (one per line), '
                             'e.g. data/pubchem_smiles/pubchem_1M_seed2026.txt; '
                             'overrides the pkl+resample path for exact-set pinning')

    # ===== General training =====
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Specific checkpoint to resume from')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Pretrained weights for fine-tuning (fresh optimizer)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['USPTO', 'JPO'],
                        help='Validation benchmarks (first is primary)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Total epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='MLE batch size per GPU (default: 64)')
    parser.add_argument("--encoder_lr", type=float, default=4e-4,
                        help="Encoder learning rate (default: 4e-4)")
    parser.add_argument("--decoder_lr", type=float, default=4e-4,
                        help="Decoder learning rate (default: 4e-4)")
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    log_dir = resolve_artifact_dir(
        project_dir,
        'ablation_joint_mle',
        category='checkpoints',
        for_write=True,
    )
    pubchem_smiles_pkl = data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"

    FAST_TEST = args.fast_test

    # ===== Load PubChem SMILES pool =====
    # Text list (e.g. the pinned pubchem_1M_seed2026.txt released on HF)
    # takes precedence; the full PubChem pickle pool is the fallback.
    if args.smiles_list:
        with open(args.smiles_list) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        if not pubchem_smiles_pkl.exists():
            print(f"ERROR: PubChem SMILES not found: {pubchem_smiles_pkl}")
            print("       (obtain data/pubchem_smiles/pubchem_smile_list.pkl — see TRAINING.md — "
                  "or pass --smiles_list <path/to/smiles.txt>)")
            sys.exit(1)
        with open(pubchem_smiles_pkl, 'rb') as f:
            smiles_list = pickle.load(f)
    smiles_num = min(args.synthetic_num, len(smiles_list))
    print(f"PubChem SMILES pool: {len(smiles_list)}, using {smiles_num}")

    # ===== Load MolParser SMILES (replaces MRT with MLE) =====
    molparser_csv = data_dir / "molparser_sft_real" / "sft_real.csv"
    if not molparser_csv.exists():
        print(f"ERROR: MolParser CSV not found: {molparser_csv}")
        sys.exit(1)
    molparser_smiles = pd.read_csv(molparser_csv)['SMILES'].tolist()
    print(f"MolParser SMILES (extra MLE): {len(molparser_smiles)}")

    # ===== MLE data kwargs =====
    data_kwargs = dict(
        synthetic_smiles_list=smiles_list,
        synthetic_smiles_num=4096 if FAST_TEST else smiles_num,
        mol_augment=True,
        extra_smiles_list=molparser_smiles[:256] if FAST_TEST else molparser_smiles,
    )

    # Add USPTO for joint mode
    if args.data_mode == 'joint':
        uspto_dir = data_dir / "uspto_mol"
        csv_candidates = [
            uspto_dir / "train_680k_normalised.csv",
            uspto_dir / "train_680k.csv",
            uspto_dir / "train.csv",
        ]
        train_csv = None
        for c in csv_candidates:
            if c.exists():
                train_csv = str(c)
                break
        if train_csv is None and uspto_dir.exists():
            csvs = sorted(uspto_dir.glob("*.csv"))
            if csvs:
                train_csv = str(csvs[0])
        if train_csv is None:
            print(f"ERROR: No USPTO training CSV found in {uspto_dir}")
            sys.exit(1)
        print(f"USPTO training CSV: {train_csv}")
        data_kwargs['uspto_csv_path'] = train_csv
        data_kwargs['uspto_data_dir'] = str(data_dir)

    # ===== Validation benchmarks =====
    real_dir = data_dir / "benchmark" / "real"
    val_benchmarks = []
    for bm_name in args.val_benchmarks:
        bm_dir = real_dir / bm_name
        bm_csv = real_dir / f"{bm_name}.csv"
        if not bm_csv.exists():
            print(f"WARNING: benchmark CSV not found: {bm_csv}, skipping")
            continue
        val_benchmarks.append({'name': bm_name, 'dir': str(bm_dir),
                               'csv': str(bm_csv)})

    # ===== Resume =====
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(
            resolve_checkpoint_path(project_dir, 'ablation_joint_mle', 'checkpoint_resume')
        )

    print(f"Data mode: {args.data_mode}")
    print(f"Save path: {log_dir}")

    train_joint(
        **data_kwargs,

        # No MRT — interleaved MLE ablation
        # extra_smiles_list (in data_kwargs) creates a separate DataLoader
        # with the same cadence, weight, warmup, and batch size as MRT.
        mrt_csv_paths=None,
        mrt_image_dirs=None,

        # Interleaving params (actively used for iMLE cadence/weight/warmup)
        mrt_weight=0.1,
        mrt_warmup_epochs=5,
        mrt_every_n_steps=0,
        mrt_batch_size=16,

        # validation
        benchmarks=val_benchmarks,
        val_max_samples=50 if FAST_TEST else None,

        # architecture (identical to train_joint_mle_mrt.py)
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # training (identical to train_joint_mle_mrt.py)
        save_path=str(log_dir),
        num_epochs=3 if FAST_TEST else args.num_epochs,
        batch_size=4 if FAST_TEST else args.batch_size,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        num_workers=2 if FAST_TEST else 8,
        seed=2026,
        use_amp=True,

        # molecular
        max_atoms=100,

        # resume / finetune
        resume_from=resume_path,
        finetune_from=args.finetune_from,
    )
