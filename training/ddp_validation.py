"""
COMO DDP Validation
===================

DDP-aware validation on benchmarks — runs across all DDP ranks and
aggregates metrics via all_reduce.

Moved here from ``como/evaluation.py`` so that the pip-publishable
``como`` package has no training dependencies.  Training code that needs
to run periodic validation during the training loop should import from
this module:

    from training.ddp_validation import validate

Public API:
  - ``validate`` — evaluate on a benchmark using all DDP ranks
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from como.inference import (
    SMILES_MODE_DECODER,
    SMILES_MODE_GRAPH,
    SMILES_MODE_POSTPROCESS,
    _result_to_smiles,
    _load_benchmark_gt,
)
from como.utils import compute_tanimoto_similarity
from training.ddp_utils import is_main_process, get_rank, get_world_size

__all__ = ['validate']


# ======================== Validation Dataset ========================

class _ValImageDataset(Dataset):
    """Lightweight dataset for batched validation image loading.

    Returns (image_tensor, index) pairs.  Images that fail to load are
    replaced with zero tensors so that the entire batch is not lost.
    """

    def __init__(self, items: List[Dict], transforms, image_size: Tuple[int, int] = (384, 384)):
        self.items = items
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.items[idx]['img_path']).convert('RGB')
            tensor = self.transforms(image=np.array(img))['image']
        except Exception:
            tensor = torch.zeros(3, *self.image_size)
        return tensor, idx


# ======================== DDP Validation ========================

def validate(
    model: torch.nn.Module,
    benchmark_dir: str,
    benchmark_csv_path: str,
    device: torch.device,
    epoch: int,
    writer: Optional['SummaryWriter'],  # type: ignore[name-defined]  # noqa: F821
    global_step: int,
    beam_size: int = 1,
    max_samples: Optional[int] = None,
    val_batch_size: int = 128,
    benchmark_name: str = '',
) -> Dict[str, float]:
    """Evaluate on benchmark using ALL DDP ranks; metrics aggregated via all_reduce.

    Uses mini-batched inference with parallel image loading for high GPU
    utilisation (~80-95 % vs ~20-25 % with single-image processing).

    Reports accuracy for all three SMILES modes:
      - decoder:     SMILES directly from sequence decoding
      - graph:       SMILES reconstructed from predicted atoms + bonds
      - postprocess: decoder SMILES + chirality correction via coords/edges
    """
    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    rank, world_size = get_rank(), get_world_size()

    gt_data = _load_benchmark_gt(benchmark_dir, benchmark_csv_path, max_samples)
    gt_data = [d for d in gt_data if d['gt_ok']]

    tag = benchmark_name or os.path.splitext(os.path.basename(benchmark_csv_path))[0]
    if is_main_process():
        print(f'\nEvaluating on {tag} ({len(gt_data)} valid, {world_size} GPUs)')
    if not gt_data:
        return {'exact_match_acc': 0.0, 'avg_tanimoto': 0.0, 'valid_samples': 0, 'failed_predictions': 0}

    my_data = gt_data[rank::world_size]

    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]
    local_stats = {m: [0, 0, 0.0] for m in modes}
    local_count = 0

    val_dataset = _ValImageDataset(
        my_data, actual_model.inference_transforms, actual_model.image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        n_batches = len(val_loader)
        it = (tqdm(val_loader, total=n_batches, desc=f'Epoch {epoch} [Val-{tag} R{rank}]')
              if is_main_process() else val_loader)

        for img_batch, indices in it:
            img_batch = img_batch.to(device, non_blocking=True)
            B_cur = img_batch.size(0)
            local_count += B_cur

            try:
                batch_results = actual_model.generate(
                    images=img_batch, beam_size=beam_size, device=device)
            except Exception:
                batch_results = [None] * B_cur

            for local_b in range(B_cur):
                idx = indices[local_b].item()
                gt_smi = my_data[idx]['gt_smiles']
                result = batch_results[local_b]

                for mode in modes:
                    try:
                        # GT is canonicalized with ignore_cistrans=True (see
                        # _load_benchmark_gt); predictions must match
                        pred_smi = _result_to_smiles(result, mode=mode,
                                                      ignore_cistrans=True) if result else None
                    except Exception:
                        pred_smi = None
                    if pred_smi is None:
                        local_stats[mode][1] += 1
                        continue
                    if pred_smi == gt_smi:
                        local_stats[mode][0] += 1
                    local_stats[mode][2] += compute_tanimoto_similarity(gt_smi, pred_smi)

    flat = [float(local_count)]
    for mode in modes:
        flat.extend([float(local_stats[mode][0]), float(local_stats[mode][1]), local_stats[mode][2]])
    stats = torch.tensor(flat, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_count = int(stats[0])
    results_out = {}

    if is_main_process():
        print(f'Epoch {epoch} [Val-{tag}] ({total_count} samples):')

    for i, mode in enumerate(modes):
        exact = int(stats[1 + i * 3])
        failed = int(stats[2 + i * 3])
        tan_sum = stats[3 + i * 3].item()
        acc = exact / total_count if total_count else 0.0
        avg_tan = tan_sum / total_count if total_count else 0.0

        results_out[f'{mode}/exact_match_acc'] = acc
        results_out[f'{mode}/avg_tanimoto'] = avg_tan
        results_out[f'{mode}/failed'] = failed

        if is_main_process():
            print(f'  [{mode:11s}] Exact={acc:.4f} ({exact}/{total_count})  '
                  f'Tanimoto={avg_tan:.4f}  Failed={failed}')
            if writer:
                writer.add_scalar(f'Val_{tag}/{mode}_exact_match_acc', acc, global_step)
                writer.add_scalar(f'Val_{tag}/{mode}_avg_tanimoto', avg_tan, global_step)
                writer.add_scalar(f'Val_{tag}/{mode}_failed', failed, global_step)

    results_out['exact_match_acc'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/exact_match_acc', 0.0)
    results_out['avg_tanimoto'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/avg_tanimoto', 0.0)
    results_out['valid_samples'] = total_count
    results_out['failed_predictions'] = results_out.get(f'{SMILES_MODE_POSTPROCESS}/failed', 0)

    return results_out
