#!/usr/bin/env python
"""Run COMO on the bundled sample images and compare with the ground-truth
SMILES in examples.csv (exact match + tautomer-normalized match).

Usage:
    python predict_batch.py --checkpoint <path/to/final.pth> \
        [--images sample_images] [--csv examples.csv] [--device cuda:0]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ensure_como_importable():
    """Work both with a pip install and from a repository clone."""
    try:
        import como  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
        import como  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description='COMO batch prediction on the bundled samples')
    ap.add_argument('--checkpoint', required=True, help='path to a COMO .pth checkpoint')
    ap.add_argument('--images', default=str(HERE / 'sample_images'), help='directory of input images')
    ap.add_argument('--csv', default=str(HERE / 'examples.csv'), help='CSV with (name, SMILES) ground truth')
    ap.add_argument('--device', default=None, help='"cuda:0", "cpu", ... (default: auto)')
    ap.add_argument('--internal_batch_size', type=int, default=8, help='sub-batch size for decoding')
    args = ap.parse_args()

    import torch
    _ensure_como_importable()
    import como

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    names = [r['name'].strip() for r in rows]
    gt = [r['SMILES'].strip() for r in rows]
    paths = [os.path.join(args.images, f'{n}.png') for n in names]
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f'missing image: {p}')

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, {len(paths)} images')
    model = como.load_model(args.checkpoint, device=device)

    preds = como.predict_batch(model, paths, internal_batch_size=args.internal_batch_size,
                               return_confidence=True)

    exact = taut = 0
    n = len(paths)
    print(f'{"name":14s} {"exact":5s} {"taut":4s} {"conf":5s} prediction')
    for name, g, (pred, conf) in zip(names, gt, preds):
        ref_can, ok = como.canonicalize_smiles(g)
        pred_can, ok = como.canonicalize_smiles(pred) if pred else ('', False)
        e = bool(ok and pred_can == ref_can)
        ref_tau, ok_t = como.canonicalize_tautomer(g)
        pred_tau, _ = como.canonicalize_tautomer(pred) if pred else ('', False)
        t = bool(ok_t and pred_tau == ref_tau)
        exact += e
        taut += t
        print(f'{name:14s} {"✓" if e else "✗":5s} {"✓" if t else "✗":4s} {conf:5.2f} {pred}')

    print(f'\nexact match:            {exact}/{n} ({exact / n:.0%})')
    print(f'tautomer-normalized:    {taut}/{n} ({taut / n:.0%})')


if __name__ == '__main__':
    main()