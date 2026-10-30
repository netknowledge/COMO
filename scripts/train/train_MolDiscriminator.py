import sys
import os
import argparse
import pickle
from pathlib import Path


def _parse_optional_patience(value: str):
    value_norm = value.strip().lower()
    if value_norm in {'none', 'null'}:
        return None
    try:
        patience = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "early_stopping_patience must be an integer or 'none'"
        ) from e
    if patience < 1:
        raise argparse.ArgumentTypeError(
            "early_stopping_patience must be >= 1, or 'none'"
        )
    return patience


def parse_args():
    parser = argparse.ArgumentParser(description='Train molecule discriminator')
    parser.add_argument('--num_smiles', type=int, default=5e5,
                        help='Number of SMILES to sample (default: 500K)')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size per GPU (default: 48)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--val_split', type=float, default=0.01,
                        help='Validation split fraction (default: 0.01)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='DataLoader workers per process (default: 10)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed (default: 2026)')
    parser.add_argument('--early_stopping_patience',
                        type=_parse_optional_patience, default=2,
                        help="Early stopping patience in epochs, or 'none' to disable (default: 2)")
    parser.add_argument('--no_ddp', action='store_true',
                        help='Disable DDP even with multiple GPUs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature (default: 0.07)')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Image size (square, default: 384)')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='LR warmup ratio (default: 0.03)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (automatic mixed precision)')
    parser.add_argument('--crem_db_path', type=str, default='',
                        help='Path to CReM fragment database '
                             '(default: crem_db/chembl33_f5.db)')
    parser.add_argument('--crem_radius', type=int, default=3,
                        help='CReM context radius for hard negatives '
                             '(3=conservative/hard, 1=aggressive/easy, default: 3)')
    parser.add_argument('--no_hard_negatives', action='store_true',
                        help='Disable CReM hard negatives (use in-batch only)')
    parser.add_argument('--use_random_renderer', action='store_true',
                        help='Randomly pick Indigo or RDKit per sample pair')
    parser.add_argument('--fast_test', action='store_true',
                        help='Quick smoke test with 100 SMILES, 2 epochs, '
                             'batch_size=8, 1 worker, no DDP')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume training from this epoch (0 = start fresh). '
                             'Loads from epoch_N.pth or training_state.pth')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids to use, e.g. "0,1" or "2". '
                             'Sets CUDA_VISIBLE_DEVICES. (default: use all GPUs)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Restrict visible GPUs before any CUDA initialization
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    project_dir = Path(__file__).resolve().parent
    while not (project_dir / 'project_naming.py').exists():
        if project_dir.parent == project_dir:
            raise FileNotFoundError('COMO repository root not found')
        project_dir = project_dir.parent
    sys.path.insert(0, str(project_dir))

    from project_naming import resolve_artifact_dir, ensure_package_root_on_syspath
    ensure_package_root_on_syspath(project_dir)
    from training.mol_discriminator import MoleculeDiscriminator

    log_dir = resolve_artifact_dir(
        project_dir,
        'mol_discriminator',
        category='checkpoints',
        for_write=True,
    )
    data_dir = project_dir / "data"
    file_path = data_dir / "pubchem_smiles/pubchem_smile_list.pkl"
    with open(file_path, 'rb') as f:
        smiles_list = pickle.load(f)

    # Override settings for fast smoke test
    if args.fast_test:
        print('=== FAST TEST MODE ===')
        args.num_smiles = 100
        args.num_epochs = 2
        args.batch_size = 8
        args.num_workers = 1
        args.no_ddp = False
        args.val_split = 0.2

    # Initialize discriminator
    discriminator = MoleculeDiscriminator(
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        image_size=(args.image_size, args.image_size),
    )

    # Train with DDP
    losses = discriminator.train(
        smiles_list=smiles_list,
        num_smiles=args.num_smiles,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        crem_db_path=args.crem_db_path,
        crem_radius=args.crem_radius,
        use_hard_negatives=not args.no_hard_negatives,
        use_random_renderer=args.use_random_renderer,
        save_path=str(log_dir),
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        use_ddp=not args.no_ddp,
        warmup_ratio=args.warmup_ratio,
        use_amp=not args.no_amp,
        resume_epoch=args.resume_epoch,
    )