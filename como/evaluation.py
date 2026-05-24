"""
COMO Evaluation
===============

Benchmark evaluation and multi-GPU inference dispatch.

This module contains only inference-time evaluation
logic, with no training dependencies.

Public API:
  - ``evaluate_benchmarks`` — full benchmark suite evaluation
  - ``predict_multigpu``    — distribute inference across GPUs
"""

import math
from typing import Dict, List, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm

from .inference import (
    SMILES_MODE_DECODER,
    SMILES_MODE_GRAPH,
    SMILES_MODE_POSTPROCESS,
    _result_to_smiles,
    _load_benchmark_gt,
    _compute_benchmark_metrics,
)

__all__ = ['evaluate_benchmarks', 'predict_multigpu']


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


def _load_hf_benchmark(
    hf_dataset: str,
    hf_config: str,
    split: str = "test",
) -> Tuple[List[Dict], List]:
    """Load benchmark ground truth and images from a HuggingFace dataset.

    Args:
        hf_dataset: HuggingFace dataset repo id (e.g. ``"Keylab/OCSR-Benchmarks"``).
        hf_config:  Config / subset name (e.g. ``"USPTO"``).
        split:      Dataset split to load (default ``"test"``).

    Returns:
        (gt_data, images) where *gt_data* is a list of dicts with keys
        ``image_id``, ``gt_smiles``, ``gt_ok``, and *images* is a list of
        PIL Images with the same ordering.
    """
    try:
        from datasets import load_dataset as _hf_load
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install it with: pip install datasets"
        ) from exc

    from .chemistry import canonicalize_smiles
    from .utils import remove_atom_mapping

    print(f"  Loading '{hf_config}' from {hf_dataset} (split={split}) ...")
    ds = _hf_load(hf_dataset, name=hf_config, split=split)

    gt_data: List[Dict] = []
    images: List = []
    for sample in ds:
        try:
            smi = remove_atom_mapping(sample['SMILES'])
            smi, ok = canonicalize_smiles(smi, ignore_cistrans=True)
        except Exception:
            smi, ok = sample['SMILES'], False
        gt_data.append({
            'image_id': str(sample['image_id']),
            'gt_smiles': smi,
            'gt_ok': ok,
        })
        images.append(sample['image'])  # PIL Image

    return gt_data, images


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
        benchmarks: List of dicts, each with keys:

            * File-based: ``'name'``, ``'benchmark_dir'``, ``'csv_path'``
            * HF dataset: ``'name'``, ``'hf_dataset'``, ``'hf_config'``,
              and optionally ``'hf_split'`` (default ``"test"``)

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

        if 'hf_dataset' in b:
            gt_data, images = _load_hf_benchmark(
                b['hf_dataset'],
                b.get('hf_config', name),
                b.get('hf_split', 'test'),
            )
        else:
            gt_data = _load_benchmark_gt(b['benchmark_dir'], b['csv_path'])
            images = [d['img_path'] for d in gt_data]
        print(f"  Images: {len(images)}")

        raw_results = predict_multigpu(model, images, beam_size=beam_size, num_gpus=num_gpus)
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
