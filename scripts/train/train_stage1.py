"""
[Deprecated 2-stage pipeline]
Stage 1: Supervised MLE Pre-training for COMO
========================================

Trains COMO from scratch. Four data modes, matching the original
MolScribe training strategy:

  --data_mode joint      (default)  1M PubChem synthetic + 680K USPTO patent images
  --data_mode synthetic             1M PubChem synthetic only
  --data_mode uspto                 680K USPTO patent images only (not recommended)
  --data_mode molparser             83K MolParser SFT real data (continual MLE baseline)

The "joint" mode combines dynamically rendered PubChem SMILES with real USPTO
patent images in a single ConcatDataset, following the original MolScribe
AuxTrainDataset approach and avoiding catastrophic forgetting.

The "molparser" mode supports optional MLE regularization (--mle_weight) using
PubChem synthetic data, matching the same mechanism in Stage 2 MRT finetuning
for apple-to-apple comparison.

Prerequisites (for joint / USPTO modes):
    USPTO data prepared: python scripts/prepare_uspto_mol.py

Usage:
    # Joint training (recommended, matches original MolScribe)
    torchrun --nproc_per_node=4 scripts/train_stage1.py

    # Synthetic only
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode synthetic

    # USPTO only (not recommended)
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode uspto

    # MolParser continual MLE (baseline for Stage 2 comparison)
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode molparser --finetune_from ...

    # MolParser continual MLE with PubChem regularization (ablation for 2-stage pipeline)
    torchrun --nproc_per_node=4 scripts/train_stage1.py --data_mode molparser --mle_weight 0.3

    # Resume if interrupted
    torchrun --nproc_per_node=4 scripts/train_stage1.py --resume

    # Custom synthetic count
    torchrun --nproc_per_node=4 scripts/train_stage1.py --synthetic_num 500000
"""
import sys
import os
from pathlib import Path
import pickle
import argparse

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

from training.train_mle import train_mle

# Model artifact keys per data mode
MODEL_ARTIFACT_KEYS = {
    'joint':     'stage1_joint',
    'synthetic': 'stage1_synthetic',
    'uspto':     'stage1_uspto',
    'molparser': 'ablation_continual_mle',  # Continual MLE baseline for Stage 2 comparison
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 1: Supervised pre-training on synthetic and/or USPTO data')
    parser.add_argument('--data_mode', type=str, default='joint',
                        choices=['joint', 'synthetic', 'uspto', 'molparser'],
                        help='Data mode: joint (1M+680K, default), synthetic (1M), uspto (680K), or molparser (83K)')
    parser.add_argument('--synthetic_num', type=int, default=int(1e6),
                        help='Number of PubChem SMILES to use (default: 1M)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Path to a pretrained checkpoint to fine-tune from (loads weights only, fresh optimizer)')
    parser.add_argument('--encoder_weights', type=str, default=None,
                        help='Path to a MIM-pretrained image_encoder.* checkpoint '
                             '(Swin-B init replacement; everything else identical)')
    parser.add_argument('--smiles_list', type=str, default=None,
                        help='Plain-text PubChem SMILES subset (one per line), '
                             'e.g. data/pubchem_smiles/pubchem_1M_seed2026.txt; '
                             'overrides the pkl+resample path for exact-set pinning')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['USPTO', 'JPO'],
                        help='Validation benchmark names (default: USPTO JPO). '
                             'First one is primary for early stopping.')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU (default: 64)')
    parser.add_argument("--encoder_lr", type=float, default=4e-4,
                        help="Learning rate for the encoder (default: 4e-4)")
    parser.add_argument("--decoder_lr", type=float, default=4e-4,
                        help="Learning rate for the decoder (default: 4e-4)")
    # MLE regularization (matches Stage 2 mechanism for apple-to-apple comparison)
    parser.add_argument('--mle_weight', type=float, default=0.0,
                        help='MLE regularization weight (0 = disabled). '
                             'Recommended: 0.3 to prevent distribution drift.')
    parser.add_argument('--mle_smiles_path', type=str, default=None,
                        help='Path to PubChem SMILES .pkl for MLE regularization '
                             '(default: data/pubchem_smiles/pubchem_smile_list.pkl)')
    parser.add_argument('--mle_smiles_num', type=int, default=50000,
                        help='Number of SMILES to sample for MLE (default: 50000)')
    parser.add_argument('--mle_batch_size', type=int, default=8,
                        help='MLE batch size per GPU (default: 8)')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    artifact_key = MODEL_ARTIFACT_KEYS[args.data_mode]
    log_dir = resolve_artifact_dir(
        project_dir,
        artifact_key,
        category='checkpoints',
        for_write=True,
    )
    pubchem_smiles_pkl = data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"

    # ===== Load primary training SMILES (for joint / synthetic / molparser modes) =====
    smiles_list = None
    smiles_num = 0
    if args.smiles_list:
        with open(args.smiles_list) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        smiles_num = len(smiles_list)
        print(f'Pinned SMILES list: {args.smiles_list} ({smiles_num:,} SMILES)')
    elif args.data_mode in ('joint', 'synthetic'):
        if not pubchem_smiles_pkl.exists():
            print(f"ERROR: PubChem SMILES not found: {pubchem_smiles_pkl}")
            sys.exit(1)
        with open(pubchem_smiles_pkl, 'rb') as f:
            smiles_list = pickle.load(f)
        smiles_num = min(args.synthetic_num, len(smiles_list))
        print(f"PubChem SMILES pool: {len(smiles_list)}, using {smiles_num}")
    elif args.data_mode == 'molparser':
        smiles_path = data_dir / "molparser_sft_real" / "sft_real.csv"
        if not smiles_path.exists():
            print(f"ERROR: MolParser SFT SMILES not found: {smiles_path}")
            sys.exit(1)
        import pandas as pd
        smiles_list = pd.read_csv(smiles_path)['SMILES'].tolist()
        smiles_num = len(smiles_list)
        print(f"MolParser SFT SMILES count: {smiles_num}")

    # ===== Locate USPTO CSV (for joint / uspto modes) =====
    train_csv = None
    if args.data_mode in ('joint', 'uspto'):
        uspto_dir = data_dir / "uspto_mol"
        csv_candidates = [
            uspto_dir / "train_680k_normalised.csv",
            uspto_dir / "train_680k.csv",
            uspto_dir / "train.csv",
        ]
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
            print("Run first: python scripts/prepare_uspto_mol.py")
            sys.exit(1)
        print(f"USPTO training CSV: {train_csv}")

    # ===== Determine resume checkpoint path =====
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(
            resolve_checkpoint_path(project_dir, artifact_key, 'checkpoint_resume')
        )

    # ===== Fast Test Mode =====
    FAST_TEST = args.fast_test

    # ===== Build train() kwargs based on data mode =====
    data_kwargs = {}
    if args.data_mode == 'synthetic':
        # Pure synthetic: dynamic rendering from SMILES
        data_kwargs = dict(
            synthetic_smiles_list=smiles_list,
            synthetic_smiles_num=4096 if FAST_TEST else smiles_num,
            mol_augment=True,
        )
    elif args.data_mode == 'molparser':
        # Continual MLE on MolParser SFT real data (baseline for Stage 2 comparison).
        # Optionally add PubChem MLE regularization via --mle_weight to match Stage 2.
        data_kwargs = dict(
            synthetic_smiles_list=smiles_list,
            synthetic_smiles_num=4096 if FAST_TEST else smiles_num,
            wild_smiles=True,
            mol_augment=False,  # No mol augment for real data
        )
    elif args.data_mode == 'uspto':
        # Pure USPTO: file-based patent images (not recommended)
        data_kwargs = dict(
            uspto_csv_path=train_csv,
            uspto_data_dir=str(data_dir),
            mol_augment=True,
        )
    else:
        # Joint: USPTO images + synthetic via ConcatDataset
        data_kwargs = dict(
            synthetic_smiles_list=smiles_list,
            synthetic_smiles_num=4096 if FAST_TEST else smiles_num,
            uspto_csv_path=train_csv,
            uspto_data_dir=str(data_dir),
            mol_augment=True,

        )

    # ===== Build validation benchmark list =====
    real_dir = data_dir / "benchmark" / "real"
    val_benchmarks = []
    for bm_name in args.val_benchmarks:
        bm_dir = real_dir / bm_name
        bm_csv = real_dir / f"{bm_name}.csv"
        if not bm_csv.exists():
            print(f"WARNING: benchmark CSV not found: {bm_csv}, skipping")
            continue
        val_benchmarks.append({'name': bm_name, 'dir': str(bm_dir), 'csv': str(bm_csv)})

    print(f"Data mode: {args.data_mode}")
    print(f"Save path: {log_dir}")

    # ===== Resolve MLE regularization path =====
    mle_smiles_path = args.mle_smiles_path or str(pubchem_smiles_pkl)

    train_mle(
        **data_kwargs,

        # training
        save_path=str(log_dir),
        num_epochs=2 if FAST_TEST else args.num_epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        num_workers=4 if FAST_TEST else 8,
        seed=2026,
        use_amp=True,

        # molecular
        max_atoms=100,

        # validation (real-world benchmarks)
        benchmarks=val_benchmarks,
        val_max_samples=50 if FAST_TEST else None,

        # vision transformer
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,
        encoder_weights=args.encoder_weights,

        # decoder transformer
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # MLE regularization (matches Stage 2 mechanism)
        mle_weight=args.mle_weight,
        mle_smiles_path=mle_smiles_path,
        mle_smiles_num=args.mle_smiles_num,
        mle_batch_size=2 if FAST_TEST else args.mle_batch_size,

        # resume / finetune
        resume_from=resume_path,
        finetune_from=args.finetune_from,
    )
