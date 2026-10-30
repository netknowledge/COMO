"""
[Deprecated 2-stage pipeline]
Stage 2: Real-Image MRT Finetuning on 83K MolParser Data
========================================================

MRT finetuning on real-world molecule images with optional MLE regularization.
MRT with visual reward:
    pred SMILES → render → encode → cosine_sim(features_orig, features_rendered)
Optional MLE loss on synthetic PubChem data prevents distribution drift.

Continues from a Stage 1 checkpoint.
By default this uses the 1M+652K joint pretrain, but historical COMO
branches based on 1M synthetic only or 1M+436K USPTO can be selected via
--stage2_artifact. Historical directories containing "82K" were in fact
trained on ~83K MolParser examples.

Trains on MolParser sft_real data (~83K), validates on USPTO and JPO (held out).

Prerequisites:
    1. Default branch: checkpoints/COMO_stage1_joint/final.pth exists
    2. Synthetic historical branch: checkpoints/COMO_stage1_synthetic/best.pth exists
    3. 1M+436K historical branch: pass --pretrained_checkpoint explicitly
    4. MolParser sft_real data available: data/molparser_sft_real/

Usage:
    torchrun --nproc_per_node=4 scripts/train_stage2_83K_MolParser.py
    torchrun --nproc_per_node=4 scripts/train_stage2_83K_MolParser.py --resume
    python scripts/train_stage2_83K_MolParser.py --fast_test
    python scripts/train_stage2_83K_MolParser.py --mle_weight 0.3  # with MLE regularization
    torchrun --nproc_per_node=4 scripts/train_stage2_83K_MolParser.py \
        --stage2_artifact stage2_mrt_from_stage1_synthetic
    torchrun --nproc_per_node=4 scripts/train_stage2_83K_MolParser.py \
        --stage2_artifact stage2_mrt_from_stage1_1M436K \
        --pretrained_checkpoint /path/to/pretrain_best.pth
"""
import sys
import os
from pathlib import Path
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

from training.train_mrt import train_mrt


STAGE2_ARTIFACT_DEFAULTS = {
    'stage2_mrt': {
        'description': 'Default Stage 2 branch from the 1M+652K joint pretrain',
        'pretrained_artifact_key': 'stage1_joint',
        'pretrained_checkpoint_name': 'final',
    },
    'stage2_mrt_from_stage1_synthetic': {
        'description': 'Historical Stage 2 branch from the 1M synthetic pretrain',
        'pretrained_artifact_key': 'stage1_synthetic',
        'pretrained_checkpoint_name': 'best',
    },
    'stage2_mrt_from_stage1_1M436K': {
        'description': 'Historical Stage 2 branch from the 1M+436K USPTO pretrain',
        'pretrained_artifact_key': None,
        'pretrained_checkpoint_name': None,
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-image MRT finetuning')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint_resume.pth')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a specific checkpoint file to resume from')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained model weights for fine-tuning '
                             '(default: checkpoints/COMO_stage1_joint/final.pth)')
    parser.add_argument('--stage2_artifact', type=str, default='stage2_mrt',
                        choices=list(STAGE2_ARTIFACT_DEFAULTS.keys()),
                        help='Artifact family for save/resume paths. '
                            'Use the non-default values to continue the '
                            'historical MolScribe_re branches safely.')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke-test with tiny data')
    parser.add_argument('--mrt_alpha', type=float, default=1.0,
                        help='MRT alpha (sharpening exponent)')
    parser.add_argument('--reward_mode', type=str, default='visual',
                        choices=['visual', 'tanimoto', 'edit_distance'],
                        help='Reward similarity mode')
    parser.add_argument('--visual_reward_renderer', type=str, default='drawing_engine',
                        choices=['rdkit', 'drawing_engine'],
                        help='Renderer backend for visual reward images')
    parser.add_argument('--mrt_sampling_method', type=str, default='multinomial',
                        choices=['multinomial', 'beam_search'],
                        help='MRT candidate sampling method (default: multinomial)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids (default: "0")')
    parser.add_argument('--val_benchmarks', type=str, nargs='+',
                        default=['JPO', 'USPTO'],
                        help='Validation benchmark names (default: JPO, USPTO). '
                             'First one is primary for early stopping.')
    parser.add_argument('--visual_reward_encoder', type=str, default=None,
                        help='Path to external image encoder checkpoint (e.g. '
                             'MolDiscriminator SiameseNetwork .pth) for visual reward')
    parser.add_argument('--visual_reward_embedding_dim', type=int, default=128,
                        help='Embedding dim of the external visual reward encoder (default: 128)')
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
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    data_dir = project_dir / "data"
    stage2_cfg = STAGE2_ARTIFACT_DEFAULTS[args.stage2_artifact]
    log_dir = resolve_artifact_dir(
        project_dir,
        args.stage2_artifact,
        category='checkpoints',
        reward_mode=args.reward_mode,
        visual_reward_encoder=args.visual_reward_encoder,
        for_write=True,
    )
    real_dir = data_dir / "benchmark" / "real"

    # ===== Determine checkpoint paths =====
    if args.pretrained_checkpoint:
        pretrained_path = args.pretrained_checkpoint
    else:
        pretrained_artifact_key = stage2_cfg['pretrained_artifact_key']
        pretrained_checkpoint_name = stage2_cfg['pretrained_checkpoint_name']
        if pretrained_artifact_key is None or pretrained_checkpoint_name is None:
            parser.error(
                '--pretrained_checkpoint is required when '
                '--stage2_artifact stage2_mrt_from_stage1_1M436K because the '
                'historical 1M+436K Stage 1 checkpoint is not produced by an '
                'active training entrypoint in this repository.'
            )
        pretrained_path = str(
            resolve_checkpoint_path(
                project_dir,
                pretrained_artifact_key,
                pretrained_checkpoint_name,
            )
        )

    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = str(
            resolve_checkpoint_path(
                project_dir,
                args.stage2_artifact,
                'checkpoint_resume',
                reward_mode=args.reward_mode,
                visual_reward_encoder=args.visual_reward_encoder,
            )
        )

    # ===== Build validation benchmark list =====
    val_benchmarks = []
    corrected_benchmarks = {'JPO', 'CLEF', 'UOB'}
    for bm_name in args.val_benchmarks:
        bm_dir = real_dir / bm_name
        csv_name = f"{bm_name}_corrected.csv" if bm_name in corrected_benchmarks else f"{bm_name}.csv"
        bm_csv = real_dir / csv_name
        if not bm_csv.exists():
            print(f"WARNING: benchmark CSV not found: {bm_csv}, skipping")
            continue
        val_benchmarks.append({'name': bm_name, 'dir': str(bm_dir), 'csv': str(bm_csv)})

    # Training data: MolParser sft_real (~83K)
    train_csv_paths = [
        str(data_dir / "molparser_sft_real" / "sft_real.csv"),
    ]
    train_image_dirs = [
        str(data_dir / "molparser_sft_real" / "images"),
    ]

    # ===== Fast Test Mode =====
    FAST_TEST = args.fast_test

    print(f"Stage 2 artifact: {args.stage2_artifact} ({stage2_cfg['description']})")
    print(f"Pretrained checkpoint: {pretrained_path}")

    train_mrt(
        # data
        train_csv_paths=train_csv_paths,
        train_image_dirs=train_image_dirs,

        # validation benchmarks
        benchmarks=val_benchmarks,
        val_max_samples=50 if FAST_TEST else None,

        # model
        pretrained_path=pretrained_path,
        save_path=str(log_dir),

        # architecture (must match pretrained)
        image_size=(384, 384),
        n_bins=64,
        backbone='swin_b',
        num_decoder_layers=6,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,

        # training
        num_epochs=2 if FAST_TEST else 6,
        batch_size=4 if FAST_TEST else 16,
        encoder_lr=1e-5,
        decoder_lr=1e-5,
        weight_decay=1e-6,
        warmup_ratio=0.02,
        seed=2026,
        early_stopping_patience=10,
        num_workers=2 if FAST_TEST else 4,
        use_amp=True,

        # MRT
        mrt_max_len=500,
        mrt_temperature=0.5,
        mrt_n_samples=32,
        mrt_sampling_method=args.mrt_sampling_method,

        # Reward weights:
        # R = w_v·𝟙[valid] + w_sim·cosine_sim(render(pred), orig) + w_e·𝟙[exact]
        reward_validity_weight=0.1,
        reward_similarity_weight=0.5,
        reward_exact_match_weight=0.4,
        reward_mode=args.reward_mode,
        visual_reward_renderer=args.visual_reward_renderer,

        mrt_alpha=args.mrt_alpha,

        # External visual reward encoder
        visual_reward_encoder_path=args.visual_reward_encoder,
        visual_reward_embedding_dim=args.visual_reward_embedding_dim,

        # MLE regularization
        mle_weight=args.mle_weight,
        mle_smiles_path=args.mle_smiles_path or str(data_dir / "pubchem_smiles" / "pubchem_smile_list.pkl"),
        mle_smiles_num=args.mle_smiles_num,
        mle_batch_size=2 if FAST_TEST else args.mle_batch_size,

        # resume
        resume_from=resume_path,
    )
