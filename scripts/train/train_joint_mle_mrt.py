"""
Interleaved MLE + MRT Joint Training for COMO
============================================

Trains COMO from scratch with interleaved supervised MLE on
synthetic/USPTO data and Minimum Risk Training (MRT) on real-world
molecule images.  Unlike the two-stage pipeline (Stage 1 MLE → Stage 2
MRT), this mode performs both objectives in a single run.

Training phases:
  1. **Warmup** (epochs 1..mrt_warmup_epochs): MLE-only, to build
     meaningful encoder features before MRT sampling.
  2. **Interleaved** (remaining epochs): each MLE step is paired with
     one MRT step.  Per-step loss:  L = L_mle + mrt_weight · L_mrt

MLE data modes (same as Stage 1):
  --data_mode joint      (default) 1M PubChem synthetic + 680K USPTO
  --data_mode synthetic             1M PubChem synthetic only

MRT data:
  Real molecule images with GT SMILES from MolParser sft_real (~83K).

Prerequisites:
    1. PubChem SMILES pool: data/pubchem_smiles/pubchem_smile_list.pkl
    2. USPTO data (for joint mode): python scripts/prepare_uspto_mol.py
    3. MolParser sft_real data: data/molparser_sft_real/

Usage:
    # Default: joint MLE + visual MRT, 5-epoch warmup
    torchrun --nproc_per_node=4 scripts/train_joint_mle_mrt.py

    # Tanimoto reward, quick test
    python scripts/train_joint_mle_mrt.py --reward_mode tanimoto --fast_test

    # Custom warmup + MRT weight
    torchrun --nproc_per_node=4 scripts/train_joint_mle_mrt.py \\
        --mrt_warmup_epochs 10 --mrt_weight 0.5

    # Resume
    torchrun --nproc_per_node=4 scripts/train_joint_mle_mrt.py --resume
"""
import sys
import os
from pathlib import Path
import argparse
import pickle

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
        description='Interleaved MLE + MRT joint training')

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

    # ===== MRT =====
    parser.add_argument('--mrt_warmup_epochs', type=int, default=5,
                        help='MLE-only warmup epochs before MRT (default: 5)')
    parser.add_argument('--mrt_weight', type=float, default=0.1,
                        help='MRT loss weight (default: 0.1). Keep low because '
                             'the MLE learning rate (4e-4) is ~40x higher '
                             'than the Stage 2 MRT rate (1e-5).')
    parser.add_argument('--loader_timeout', type=float, default=60.0,
                        help='DataLoader worker ping timeout in seconds for the '
                             'MLE + MRT loaders (default 60 = the released '
                             'training-scripts value). A slow molecule-rendering '
                             'batch past this limit raises a loader error; the '
                             'run then kills a worker and rebuilds the whole '
                             'pipeline. Raise it (e.g. 600) on busy hosts so '
                             'slow batches are delivered instead of skipped.')
    parser.add_argument('--mrt_every_n_steps', type=int, default=0,
                        help='Add an MRT backward into every Nth optimizer '
                             'step (the MRT never steps alone - an MRT slot '
                             'is one optimizer step whose backward contains '
                             'both the MLE and the MRT losses). '
                             '0 = auto (~one MRT loader pass per epoch; '
                             'accounts for mle_accum / mrt_accum automatically).')
    parser.add_argument('--mrt_batch_size', type=int, default=16,
                        help='MRT batch size per GPU (default: 16)')
    parser.add_argument('--mrt_accum', type=int, default=1,
                        help='Sequential MRT sub-batches per MRT step; the '
                             'update sees mrt_batch_size x world_size x N '
                             'images with 1/N-weighted backward each - the '
                             'same per-update mean as one batch of that size, '
                             'at the peak memory of a single sub-batch.')
    parser.add_argument('--mrt_alpha', type=float, default=1.0,
                        help='MRT alpha (sharpening exponent)')
    parser.add_argument('--reward_mode', type=str, default='visual',
                        choices=['visual', 'tanimoto', 'edit_distance'],
                        help='Reward similarity mode (default: visual)')
    parser.add_argument('--visual_reward_renderer', type=str, default='drawing_engine',
                        choices=['rdkit', 'drawing_engine'],
                        help='Renderer for visual reward images')
    parser.add_argument('--mrt_sampling_method', type=str, default='multinomial',
                        choices=['multinomial', 'beam_search'],
                        help='MRT candidate sampling method')
    parser.add_argument('--mrt_n_samples', type=int, default=32,
                        help='MRT candidates per image / beam size (default: 32)')
    parser.add_argument('--adaptive_n', action='store_true',
                        help='Enable adaptive-N MRT (beam_search only)')
    parser.add_argument('--adaptive_n_gamma', type=float, default=1.5,
                        help='Adaptive-N scaling factor γ: N(x)=clip(γ·ESS, N_min, N_max). '
                             'Recommended sweep: 1.0 / 1.5 / 2.0')
    parser.add_argument('--adaptive_n_min', type=int, default=8,
                        help='Minimum N per image for adaptive-N (default: 8)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Optional experiment name appended to save path. '
                             'When adaptive_n is enabled and exp_name is not '
                             'given, defaults to "adaptive_n".')
    parser.add_argument('--visual_reward_encoder', type=str, default=None,
                        help='Path to external visual reward encoder (.pth)')
    parser.add_argument('--visual_reward_embedding_dim', type=int, default=128,
                        help='Embedding dim of external encoder (default: 128)')

    # ===== General training =====
    parser.add_argument('--fast_test', action='store_true',
                        help='3-epoch smoke (ep1 MLE, ep2-3 MLE+MRT) + validation + '
                             'checkpointing; real batch recipe, data capped by '
                             'FAST_TEST_MLE_LIMIT/FAST_TEST_MRT_LIMIT (env)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Specific checkpoint to resume from')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Pretrained weights for fine-tuning (fresh optimizer)')
    parser.add_argument('--encoder_weights', type=str, default=None,
                        help='Image-encoder-only init: flat image_encoder.* '
                             'ckpt (e.g. MIM-pretrained); overrides the '
                             'ImageNet weights, proj stays fresh')
    parser.add_argument('--artifact-key', type=str, default='joint_mle_mrt',
                        help='project_naming key for the output dir '
                             '(use joint_mle_mrt_simmim / _mixmim for the '
                             'MIM-init arms so COMO_joint stays untouched)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['USPTO', 'JPO'],
                        help='Validation benchmarks (first is primary)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Total epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='MLE batch size per GPU (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='MLE DataLoader workers (default: 8)')
    parser.add_argument('--mle_accum', type=int, default=1,
                        help='MLE microbatches per optimizer step; effective '
                             'MLE batch = batch_size x world_size x mle_accum. '
                             'Keep the product constant across hardware to '
                             'keep the optimizer trajectory identical.')
    parser.add_argument("--encoder_lr", type=float, default=4e-4,
                        help="Encoder learning rate (default: 4e-4)")
    parser.add_argument("--decoder_lr", type=float, default=4e-4,
                        help="Decoder learning rate (default: 4e-4)")
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    # ===== Determine save / log directory =====
    # exp_name allows explicit naming; adaptive_n gets an automatic suffix
    # so it never overwrites the fixed-N baseline checkpoints.
    _exp_suffix = args.exp_name or ('adaptive_n' if args.adaptive_n else None)
    log_dir = resolve_artifact_dir(
        project_dir,
        args.artifact_key,
        category='checkpoints',
        reward_mode=args.reward_mode,
        visual_reward_encoder=args.visual_reward_encoder,
        experiment_name=_exp_suffix,
        for_write=True,
    )
    pubchem_smiles_pkl = data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"

    FAST_TEST = args.fast_test
    if FAST_TEST:  # data-limit + flush-cadence checks in train_joint read env
        os.environ['FAST_TEST'] = '1'

    # ===== Load MLE SMILES pool =====
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

    # ===== MLE data kwargs =====
    data_kwargs = dict(
        synthetic_smiles_list=smiles_list,
        synthetic_smiles_num=4096 if FAST_TEST else smiles_num,
        mol_augment=True,
    )

    # Add USPTO for joint mode
    train_csv = None
    if args.data_mode == 'joint':
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
            sys.exit(1)
        print(f"USPTO training CSV: {train_csv}")
        data_kwargs['uspto_csv_path'] = train_csv
        data_kwargs['uspto_data_dir'] = str(data_dir)

    # ===== MRT data (MolParser sft_real) =====
    mrt_csv_paths = [str(data_dir / "molparser_sft_real" / "sft_real.csv")]
    mrt_image_dirs = [str(data_dir / "molparser_sft_real" / "images")]

    for csv_p in mrt_csv_paths:
        if not os.path.exists(csv_p):
            print(f"ERROR: MRT CSV not found: {csv_p}")
            sys.exit(1)

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
        # resume within THE RUN'S OWN artifact dir (the key below decides
        # which arm's checkpoint_resume.pth is continued; the default key
        # 'joint_mle_mrt' = the deployed reference dir)
        resume_path = str(
            resolve_checkpoint_path(
                project_dir,
                args.artifact_key,
                'checkpoint_resume',
                reward_mode=args.reward_mode,
                visual_reward_encoder=args.visual_reward_encoder,
                experiment_name=_exp_suffix,
            )
        )

    print(f"Data mode: {args.data_mode}")
    print(f"Save path: {log_dir}")

    train_joint(
        **data_kwargs,

        # MRT data
        mrt_csv_paths=mrt_csv_paths,
        mrt_image_dirs=mrt_image_dirs,

        # MRT hyperparams
        mrt_weight=args.mrt_weight,
        mrt_warmup_epochs=1 if FAST_TEST else args.mrt_warmup_epochs,
        mrt_every_n_steps=1 if FAST_TEST else args.mrt_every_n_steps,
        # FAST_TEST no longer shrinks the batch recipe (smoke = real shapes,
        # doubling as the perf benchmark); the data limits live in train_joint
        # (FAST_TEST_MLE_LIMIT / FAST_TEST_MRT_LIMIT envs)
        mrt_batch_size=args.mrt_batch_size,
        mrt_accum=args.mrt_accum,
        # trainer's kwargs are rl_* (mirrors train_mrt's mrt_* params)
        rl_max_len=500,
        rl_temperature=0.5,
        rl_n_samples=args.mrt_n_samples,
        rl_sampling_method=args.mrt_sampling_method,
        adaptive_n=args.adaptive_n,
        adaptive_n_gamma=args.adaptive_n_gamma,
        adaptive_n_min=2 if FAST_TEST else args.adaptive_n_min,
        reward_validity_weight=0.1,
        reward_similarity_weight=0.5,
        reward_exact_match_weight=0.4,
        reward_mode=args.reward_mode,
        visual_reward_renderer=args.visual_reward_renderer,
        mrt_alpha=args.mrt_alpha,
        visual_reward_encoder_path=args.visual_reward_encoder,
        visual_reward_embedding_dim=args.visual_reward_embedding_dim,

        # validation
        benchmarks=val_benchmarks,
        val_max_samples=50 if FAST_TEST else None,

        # architecture
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        pretrained=True,
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # training
        save_path=str(log_dir),
        # unrenderable-SMILES list: keep it with the run's log + checkpoints
        render_badlist_path=str(log_dir / "render_badlist.txt"),
        num_epochs=3 if FAST_TEST else args.num_epochs,
        batch_size=args.batch_size,
        mle_accum=args.mle_accum,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        early_stopping_patience=30,
        num_workers=args.num_workers,
        seed=2026,
        use_amp=True,
        loader_timeout=args.loader_timeout,

        # molecular
        max_atoms=100,

        # resume / finetune
        resume_from=resume_path,
        finetune_from=args.finetune_from,
        encoder_weights=args.encoder_weights,
    )
