"""
COMO Evaluation
===============

DDP-aware validation on benchmarks and multi-GPU inference dispatch.

These functions bridge the public ``como`` package and the private
``training/`` package: they import DDP utilities lazily so that the
public package can be used standalone without training dependencies.

Public API:
  - ``validate``            — DDP validation on a benchmark
  - ``evaluate_benchmarks`` — full benchmark suite evaluation
  - ``predict_multigpu``    — distribute inference across GPUs
"""

import math
import os
from typing import Dict, List, Optional, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from .vocab import ComoVocab
from .inference import (
    SMILES_MODE_DECODER,
    SMILES_MODE_GRAPH,
    SMILES_MODE_POSTPROCESS,
    _result_to_smiles,
    _load_benchmark_gt,
    _compute_benchmark_metrics,
)
from .utils import compute_tanimoto_similarity

__all__ = ['validate', 'evaluate_benchmarks', 'predict_multigpu']


# Lazy imports to avoid hard dependency on training/ package
def _get_ddp_funcs():
    """Lazily import DDP utilities from the training package."""
# Lazy imports to avoid hard dependency on training/ package
def _get_ddp_funcs():
    """Lazily import DDP utilities from the training package."""
    from training.ddp_utils import is_main_process, get_rank, get_world_size
    return is_main_process, get_rank, get_world_size


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
    is_main_process, get_rank, get_world_size = _get_ddp_funcs()

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
                        pred_smi = _result_to_smiles(result, mode=mode) if result else None
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


# ======================== Multi-GPU Inference ========================

def _inference_worker(rank: int, world_size: int, model: 'ComoModel',  # type: ignore[name-defined]  # noqa: F821
                      data_paths: List[str], return_dict: dict,
                      beam_size: int = 1, mini_batch_size: int = 128):
    """Worker for mp.spawn — runs inference on one GPU slice."""
    import cv2
    cv2.setNumThreads(0)
    try:
        device = torch.device(f'cuda:{rank}')
        chunk = math.ceil(len(data_paths) / world_size)
        my_paths = data_paths[rank * chunk:(rank + 1) * chunk]
        if not my_paths:
            return
        model.to(device); model.eval()
        results = []
        it = range(0, len(my_paths), mini_batch_size)
        if rank == 0:
            it = tqdm(it, desc=f"GPU-{rank}", total=math.ceil(len(my_paths) / mini_batch_size))
        for i in it:
            results.extend(model.predict_batch(my_paths[i:i + mini_batch_size],
                                               device=device, beam_size=beam_size))
        return_dict[rank] = results
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Rank {rank} failed: {e}")


def predict_multigpu(model: 'ComoModel', image_paths: List[str],  # type: ignore[name-defined]  # noqa: F821
                     beam_size: int = 1, num_gpus: int | None = None) -> List[Dict]:
    """Distribute inference across GPUs via mp.spawn.

    Args:
        model: A loaded ``ComoModel`` instance.
        image_paths: List of image file paths.
        beam_size: Beam width (1 = greedy).
        num_gpus: Number of GPUs to use. If None (default), uses all available
                  GPUs via ``torch.cuda.device_count()``.  Explicitly pass an
                  integer to limit parallelism (e.g. ``num_gpus=3``).
    Returns:
        Ordered list of prediction dicts.
    """
    import torch.multiprocessing as mp
    world_size = num_gpus if num_gpus is not None else torch.cuda.device_count()
    if world_size <= 1:
        return model.predict_batch(image_paths, beam_size=beam_size)

    print(f"Distributing {len(image_paths)} images across {world_size} GPUs...")
    model.cpu(); model.share_memory()
    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.spawn(_inference_worker,
                 args=(world_size, model, image_paths, return_dict, beam_size),
                 nprocs=world_size, join=True)
    except Exception as e:
        print(f"mp.spawn failed: {e}"); return []

    results, chunk = [], math.ceil(len(image_paths) / world_size)
    for rank in range(world_size):
        if rank in return_dict:
            results.extend(return_dict[rank])
        else:
            results.extend([{'success': False}] * min(chunk, len(image_paths) - len(results)))
    return results


def evaluate_benchmarks(
    model: 'ComoModel',  # type: ignore[name-defined]  # noqa: F821
    benchmarks: List[Dict],
    beam_size: int = 1,
    postproc_workers: int = 32,
    tautomer_standardize: bool = True,
    num_gpus: int | None = None,
) -> Dict[str, Dict]:
    """Evaluate model on multiple benchmarks using mp.spawn.

    Reports all three SMILES modes (decoder / graph / postprocess) per benchmark.

    Args:
        model: Loaded ComoModel.
        benchmarks: List of dicts, each with keys 'name', 'benchmark_dir', 'csv_path'.
        beam_size: Beam width for decoding (1 = greedy).
        postproc_workers: Thread-pool size for parallel SMILES post-processing.
        tautomer_standardize: If True, also compute tautomer-normalized exact match.
        num_gpus: Number of GPUs for multi-GPU inference.  None = all GPUs.
    Returns:
        Dict mapping benchmark name → {<mode>/exact_match_acc, <mode>/avg_tanimoto, …}
    """
    all_results = {}
    modes = [SMILES_MODE_DECODER, SMILES_MODE_GRAPH, SMILES_MODE_POSTPROCESS]

    for b in benchmarks:
        name = b['name']
        print(f"\n{'='*50}\nBenchmark: {name}\n{'='*50}")

        gt_data = _load_benchmark_gt(b['benchmark_dir'], b['csv_path'])
        image_paths = [d['img_path'] for d in gt_data]
        print(f"  Images: {len(image_paths)}")

        raw_results = predict_multigpu(model, image_paths, beam_size=beam_size, num_gpus=num_gpus)
        min_len = min(len(raw_results), len(gt_data))
        raw_results, gt_data = raw_results[:min_len], gt_data[:min_len]

        benchmark_stats = {}
        for mode in modes:
            print(f"  Post-processing [{mode}] ...")
            converter = partial(_result_to_smiles, mode=mode)
            chunksize = max(1, len(raw_results) // (postproc_workers * 4))
            with ProcessPoolExecutor(max_workers=postproc_workers) as ex:
                pred_smiles = list(tqdm(ex.map(converter, raw_results, chunksize=chunksize),
                                        total=len(raw_results), desc=f"  {mode}"))

            stats = _compute_benchmark_metrics(gt_data, pred_smiles,
                                               with_records=(mode == SMILES_MODE_POSTPROCESS),
                                               tautomer_standardize=tautomer_standardize)
            taut_info = f"  Tautomer: {stats.get('tautomer_match_acc', 0):.2f}%" if tautomer_standardize else ""
            print(f"  [{name}/{mode:11s}] Exact Match: {stats['exact_match_acc']:.2f}%{taut_info} "
                  f"Tanimoto: {stats['avg_tanimoto']:.4f}  Failed: {stats['failed_predictions']}")
            for k, v in stats.items():
                benchmark_stats[f'{mode}/{k}'] = v

        benchmark_stats['total'] = len(gt_data)
        all_results[name] = benchmark_stats

    # Summary table
    taut_header = f"{'Tautomer':>10}" if tautomer_standardize else ""
    print(f"\n{'='*90 if tautomer_standardize else '='*80}")
    print(f"{'Benchmark':<12} {'Mode':<13} {'Exact Match':>12} {taut_header}{'Tanimoto':>10} {'Failed':>8}")
    print(f"{'-'*90 if tautomer_standardize else '-'*80}")
    for name, d in all_results.items():
        for mode in modes:
            acc = d.get(f'{mode}/exact_match_acc', 0)
            taut = d.get(f'{mode}/tautomer_match_acc', 0)
            tan = d.get(f'{mode}/avg_tanimoto', 0)
            fail = d.get(f'{mode}/failed_predictions', 0)
            taut_col = f"{taut:>10.2f}%" if tautomer_standardize else ""
            print(f"{name:<12} {mode:<13} {acc:>11.2f}% {taut_col}{tan:>10.4f} {fail:>8}")
    print(f"{'='*90 if tautomer_standardize else '='*80}")
    return all_results
