#!/usr/bin/env python
"""Predict the SMILES string for a single molecular image with COMO.

Usage:
    python predict_single.py --checkpoint <path/to/final.pth> \
        [--image sample_images/Metformin.png] [--device cuda:0] [--beam_size 1]

Download a checkpoint from https://huggingface.co/Keylab/COMO, e.g.:
    hf download Keylab/COMO models/edit_distance/final.pth --local-dir checkpoints
"""
import argparse
import os
import sys


def _ensure_como_importable():
    """Work both with a pip install and from a repository clone."""
    try:
        import como  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import como  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description='COMO single-image SMILES prediction')
    ap.add_argument('--checkpoint', required=True, help='path to a COMO .pth checkpoint')
    ap.add_argument('--image', default=os.path.join(os.path.dirname(__file__), 'sample_images', 'Metformin.png'),
                    help='input image (default: bundled Metformin sample)')
    ap.add_argument('--device', default=None, help='"cuda:0", "cpu", ... (default: auto)')
    ap.add_argument('--beam_size', type=int, default=1, help='beam width (1 = greedy)')
    ap.add_argument('--confidence', action='store_true', help='also report a confidence score')
    args = ap.parse_args()

    import torch
    _ensure_como_importable()
    import como

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = como.load_model(args.checkpoint, device=device)

    if args.confidence:
        smiles, conf = como.predict(model, args.image, beam_size=args.beam_size,
                                    return_confidence=True)
        print(f'image:      {args.image}')
        print(f'SMILES:     {smiles}')
        print(f'confidence: {conf:.3f}')
    else:
        smiles = como.predict(model, args.image, beam_size=args.beam_size)
        print(f'image:  {args.image}')
        print(f'SMILES: {smiles}')


if __name__ == '__main__':
    main()