"""
COMO Training — Joint MLE + MRT
================================

Interleaved MLE (supervised) and MRT (minimum risk) training from scratch.

The first ``mrt_warmup_epochs`` epochs are MLE-only; after warmup, each MLE
step is immediately followed by one MRT (or interleaved MLE) step.

Per-step loss (after warmup)::

    L = L_mle + mrt_weight · L_mrt

Launch with torchrun::

    torchrun --nproc_per_node=<N> scripts/train_joint_mle_mrt.py

Private module — not included in the pip-publishable ``como`` distribution.
"""

import copy
import gc
import os
import random
import sys
import warnings
from typing import Dict, List, Optional, Tuple
from functools import partial
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from como.vocab import ComoVocab
from como.model import ComoModel
from como.utils import truncate_sequences

from .datasets import MoleculeDataset, USPTOMolDataset, collate_fn
from .datasets.real_image_dataset import RealImageDataset, _real_collate_fn

from training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process
from training.console_mirror import install_console_mirror
from training.losses import compute_losses, get_bond_class_weights
from training.reward import compute_mrt_loss, load_visual_reward_encoder
from training.perf_timers import perf_phase, perf_flush
from . import utils as training_utils


def train_joint(
    *,
    # ===== MLE data (synthetic SMILES + optional USPTO) =====
    synthetic_smiles_list: Optional[List[str]] = None,
    synthetic_smiles_num: Optional[int] = None,
    uspto_csv_path: Optional[str] = None,
    uspto_data_dir: Optional[str] = None,
    wild_smiles: bool = False,
    mol_augment: bool = True,
    render_badlist_path: Optional[str] = None,   # see MoleculeDataset
    max_atoms: int = 100,
    # ===== Extra MLE SMILES (e.g. MolParser for MLE-only ablation) =====
    extra_smiles_list: Optional[List[str]] = None,
    # ===== MRT data (real images with GT SMILES) =====
    mrt_csv_paths: Optional[List[str]] = None,
    mrt_image_dirs: Optional[List[str]] = None,
    # ===== MRT hyperparameters =====
    mrt_weight: float = 1.0,
    mrt_warmup_epochs: int = 5,
    mrt_every_n_steps: int = 0,
    rl_max_len: int = 500,
    rl_temperature: float = 0.5,
    rl_n_samples: int = 32,
    rl_sampling_method: str = 'multinomial',
    reward_validity_weight: float = 0.1,
    reward_similarity_weight: float = 0.5,
    reward_exact_match_weight: float = 0.4,
    reward_mode: str = 'visual',
    visual_reward_renderer: str = 'drawing_engine',
    mrt_alpha: float = 1.0,
    adaptive_n: bool = False,
    adaptive_n_gamma: float = 1.5,
    adaptive_n_min: int = 8,
    # ===== External visual reward encoder =====
    visual_reward_encoder_path: Optional[str] = None,
    visual_reward_embedding_dim: int = 128,
    # ===== Validation =====
    benchmarks: Optional[List[Dict[str, str]]] = None,
    val_max_samples: Optional[int] = None,
    # ===== Architecture =====
    image_size: Tuple[int, int] = (384, 384),
    n_bins: int = 64,
    backbone: str = 'swin_b',
    pretrained: bool = True,
    d_model: int = 256,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    # ===== Training =====
    save_path: str = '',
    num_epochs: int = 30,
    batch_size: int = 64,
    mle_accum: int = 1,
    mrt_batch_size: int = 16,
    mrt_accum: int = 1,
    encoder_lr: float = 4e-4,
    decoder_lr: float = 4e-4,
    weight_decay: float = 1e-6,
    warmup_ratio: float = 0.02,
    seed: int = 2026,
    early_stopping_patience: int = 30,
    num_workers: int = 8,
    use_amp: bool = True,
    loader_timeout: float = 60.0,
    # ===== Resume / finetune =====
    resume_from: Optional[str] = None,
    finetune_from: Optional[str] = None,
    encoder_weights: Optional[str] = None,
) -> List[float]:
    """Interleaved MLE + MRT joint training from scratch.

    Combines supervised MLE on synthetic/USPTO data with Minimum Risk
    Training (MRT) on real-world molecule images in a single run.

    Gradient flow:
      - **MLE**: DDP-wrapped model → all-reduce across ranks.
      - **MRT**: unwrapped model → local gradients, synced via manual all-reduce.
      - Both backward passes accumulate before a single ``optimizer.step()``.

    If *mrt_csv_paths* is None/empty and *extra_smiles_list* is also None,
    degrades to MLE-only training.  If *extra_smiles_list* is given without
    MRT, interleaved MLE steps replace MRT steps (controlled ablation).
    """
    # ===== Console mirror (FIRST thing: the launch banner must be captured)
    if save_path:
        install_console_mirror(os.path.join(save_path, 'console.log'))

    # ===== DDP setup =====
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    if world_size > 1:
        setup_ddp(local_rank, world_size)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    has_mrt = bool(mrt_csv_paths and mrt_image_dirs)

    rollout_fast = os.environ.get('COMO_FAST_MRT', '1') != '0'
    if is_main_process():
        print('=== Interleaved MLE + MRT Joint Training ===')
        print(f'Device: {device}, World size: {world_size}')
        effective_batch = batch_size * world_size * mle_accum
        print(f'MLE batch: {batch_size}/GPU, effective: {effective_batch}')
        if rollout_fast:
            print('Fast MRT ON: fp16 rollout KV (+ COMO_SDPA=0 opts out of fused '
                  'non-fp32 attention)')
        if has_mrt:
            print(f'MRT batch: {mrt_batch_size}/GPU x {mrt_accum} sub-batch(es), '
                  f'effective images per MRT update: {mrt_batch_size * world_size * mrt_accum}')
            print(f'MRT warmup epochs: {mrt_warmup_epochs}, MRT weight: {mrt_weight}')
            print(f'rl_temperature={rl_temperature}, rl_n_samples={rl_n_samples}, '
                  f'rl_sampling_method={rl_sampling_method}')
            print(f'reward_mode={reward_mode}, visual_reward_renderer={visual_reward_renderer}')
            print(f'reward weights: validity={reward_validity_weight}, '
                  f'similarity={reward_similarity_weight}, exact_match={reward_exact_match_weight}')
            print(f'mrt_alpha={mrt_alpha}')
        print(f'Optimizer: encoder_lr={encoder_lr}, decoder_lr={decoder_lr}')

    if has_mrt and visual_reward_renderer not in {'rdkit', 'drawing_engine'}:
        raise ValueError(f"Unknown visual_reward_renderer='{visual_reward_renderer}'.")

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed + local_rank)
    random.seed(seed + local_rank)

    if benchmarks is None:
        benchmarks = []
    if is_main_process() and benchmarks:
        print(f'Validation benchmarks ({len(benchmarks)}):')
        for bm in benchmarks:
            print(f'  {bm["name"]}: {bm["csv"]}')

    # Vocab + Model
    vocab = ComoVocab(n_bins=n_bins)
    if is_main_process():
        print(f'Vocab size: {len(vocab)}')

    model = ComoModel(
        vocab=vocab, backbone=backbone, pretrained=pretrained,
        d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, image_size=image_size,
        encoder_weights=encoder_weights,
    ).to(device)

    if finetune_from is not None and os.path.isfile(finetune_from):
        if is_main_process():
            print(f'Fine-tuning from pretrained weights: {finetune_from}')
        state_dict = torch.load(finetune_from, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        actual_model = model.module if hasattr(model, 'module') else model
        state_dict = training_utils._remap_compiled_keys(state_dict, actual_model)
        actual_model.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        if is_main_process():
            print('  Optimizer/scheduler initialized fresh for fine-tuning')

    # Frozen encoder for visual reward: only when COMO's own encoder is the
    # judge (visual mode, no external reward encoder)
    frozen_encoder = None
    frozen_pos_enc = None
    ext_vis_encoder = None
    ext_vis_transforms = None
    ext_vis_pad = None
    _needs_frozen_encoder = (has_mrt and reward_mode == 'visual'
                             and not visual_reward_encoder_path)

    if has_mrt:
        if _needs_frozen_encoder:
            frozen_encoder = copy.deepcopy(model.image_encoder).eval()
            frozen_pos_enc = copy.deepcopy(model.pos_enc_2d).eval()
            frozen_encoder.requires_grad_(False)
            frozen_pos_enc.requires_grad_(False)

        if visual_reward_encoder_path:
            if not os.path.isfile(visual_reward_encoder_path):
                raise FileNotFoundError(f'visual_reward_encoder_path not found: {visual_reward_encoder_path}')
            if is_main_process():
                print(f'Loading external visual reward encoder from {visual_reward_encoder_path} ...')
            ext_vis_encoder, ext_vis_transforms, ext_vis_pad = load_visual_reward_encoder(
                visual_reward_encoder_path, device, image_size=image_size,
                embedding_dim=visual_reward_embedding_dim,
            )
            if is_main_process():
                print(f'  External encoder loaded (embedding_dim={visual_reward_embedding_dim})')

    # torch.compile (default ON; COMO_COMPILE=0 to disable): the Swin image
    # encoder only (static shapes, ~1 min once per shape-set via the inductor
    # FX cache). Full-model dynamic compile never warmed; decoder stays eager.
    if os.environ.get('COMO_COMPILE', '1') != '0':
        if is_main_process():
            print('torch.compile ON (MLE image_encoder path; COMO_COMPILE=0 disables)')
        model.image_encoder = torch.compile(model.image_encoder, dynamic=False)

    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
    actual_model = model.module if hasattr(model, 'module') else model

    # MLE Dataset
    has_synthetic = synthetic_smiles_list is not None
    has_uspto = uspto_csv_path is not None
    if not has_synthetic and not has_uspto:
        raise ValueError('Must provide synthetic_smiles_list and/or uspto_csv_path')

    mle_datasets = []

    if has_uspto:
        if is_main_process():
            print(f'USPTO data: {uspto_csv_path}')
        uspto_dataset = USPTOMolDataset(
            vocab=vocab, csv_path=uspto_csv_path, data_dir=uspto_data_dir or '',
            image_size=image_size, img_augment=True, geo_augment=True,
        )
        mle_datasets.append(('USPTO', uspto_dataset))
        if is_main_process():
            print(f'  USPTO dataset size: {len(uspto_dataset)}')

    if has_synthetic:
        n = len(synthetic_smiles_list)
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        if synthetic_smiles_num is not None:
            indices = indices[:synthetic_smiles_num]
        sampled_smiles = [synthetic_smiles_list[i] for i in indices]
        if is_main_process():
            print(f'  Synthetic data: {len(sampled_smiles)} SMILES')
        synthetic_dataset = MoleculeDataset(
            smiles_list=sampled_smiles, wild_smiles=wild_smiles,
            shuffle_smiles=True, vocab=vocab, image_size=image_size,
            mol_augment=mol_augment, geo_augment=True, img_augment=True,
            render_badlist_path=render_badlist_path,
        )
        mle_datasets.append(('synthetic', synthetic_dataset))

    if len(mle_datasets) > 1:
        mle_train_dataset = ConcatDataset([d for _, d in mle_datasets])
        if is_main_process():
            parts = ' + '.join(f'{name}: {len(d)}' for name, d in mle_datasets)
            print(f'MLE joint dataset: {len(mle_train_dataset)} ({parts})')
    else:
        mle_train_dataset = mle_datasets[0][1]
        if is_main_process():
            print(f'MLE dataset: {len(mle_train_dataset)} ({mle_datasets[0][0]} only)')

    # FAST_TEST data limits (the smoke/perf path): an epoch must END in
    # minutes; limits are env-tunable (FAST_TEST_MLE_LIMIT / _MRT_LIMIT).
    # FAST_TEST does NOT shrink the batch recipe - batch/accum/n_samples stay
    # the user's values so the smoke doubles as the performance benchmark.
    if os.environ.get('FAST_TEST', '') == '1':
        from torch.utils.data import Subset
        _mle_limit = int(os.environ.get('FAST_TEST_MLE_LIMIT', '4096'))
        if len(mle_train_dataset) > _mle_limit:
            _stride = max(1, len(mle_train_dataset) // _mle_limit)
            _ind = np.arange(0, len(mle_train_dataset), _stride)[:_mle_limit]
            _orig_len = len(mle_train_dataset)
            mle_train_dataset = Subset(mle_train_dataset, _ind.tolist())
            if is_main_process():
                print(f'[FAST_TEST] MLE limited: {len(_ind)}/{_orig_len} '
                      f'(stride {_stride})')

    bond_class_weights = get_bond_class_weights()

    # MLE DataLoader
    mle_sampler = DistributedSampler(
        mle_train_dataset, num_replicas=world_size, rank=local_rank,
        shuffle=True, seed=seed,
    ) if world_size > 1 else None

    dl_common = {
        'batch_size': batch_size, 'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'collate_fn': partial(collate_fn, vocab=vocab, max_atoms_limit=max_atoms),
        'shuffle': (mle_sampler is None), 'sampler': mle_sampler, 'drop_last': True,
        'timeout': loader_timeout,
    }
    if num_workers > 0:
        # persistent pool = the MRT loader's deployed behavior; skips the
        # per-epoch forkserver respawn (expensive under CPU contention);
        # prefetch 8 = a deeper burst buffer against co-tenant CPU spikes
        dl_common.update({
            'prefetch_factor': 8, 'persistent_workers': True,
            'multiprocessing_context': 'forkserver',
        })

    def _make_mle_loader():
        return DataLoader(mle_train_dataset, **dl_common)

    mle_loader = _make_mle_loader()

    # MRT Dataset + DataLoader
    mrt_loader = None
    mrt_sampler = None
    mrt_total_images = 0
    if has_mrt:
        all_items = []
        for csv_path, img_dir in zip(mrt_csv_paths, mrt_image_dirs):
            ds = RealImageDataset(csv_path, img_dir, actual_model.inference_transforms, image_size)
            all_items.extend(ds.items)
            if is_main_process():
                print(f'  MRT loaded {len(ds)} images from {csv_path}')

        _mrt_dataset = RealImageDataset.__new__(RealImageDataset)
        _mrt_dataset.items = all_items
        _mrt_dataset.transforms = actual_model.inference_transforms
        _mrt_dataset.image_size = image_size
        _mrt_total = len(_mrt_dataset)
        if os.environ.get('FAST_TEST', '') == '1':
            _mrt_limit = int(os.environ.get('FAST_TEST_MRT_LIMIT', '512'))
            if _mrt_total > _mrt_limit:
                _mrt_dataset.items = _mrt_dataset.items[:_mrt_limit]
                _mrt_total = len(_mrt_dataset.items)
                if is_main_process():
                    print(f'[FAST_TEST] MRT limited to {_mrt_limit} images')

        mrt_dataset = _mrt_dataset
        mrt_total_images = _mrt_total

        if is_main_process():
            print(f'MRT total training images: {mrt_total_images}')

        mrt_sampler = DistributedSampler(
            mrt_dataset, num_replicas=world_size, rank=local_rank,
            shuffle=True, seed=seed,
        ) if world_size > 1 else None

        mrt_num_workers = min(num_workers, 4)
        mrt_dl_kwargs = dict(
            batch_size=mrt_batch_size, shuffle=(mrt_sampler is None),
            sampler=mrt_sampler, num_workers=mrt_num_workers,
            pin_memory=device.type == 'cuda', collate_fn=_real_collate_fn, drop_last=True,
            timeout=loader_timeout,
        )
        if mrt_num_workers > 0:
            mrt_dl_kwargs.update({
                'prefetch_factor': 8, 'persistent_workers': True,
                'multiprocessing_context': 'forkserver',
            })
        mrt_loader = DataLoader(mrt_dataset, **mrt_dl_kwargs)

    # Interleaved MLE DataLoader (controlled ablation)
    has_extra_mle = extra_smiles_list is not None and len(extra_smiles_list) > 0
    extra_mle_loader = None
    extra_mle_sampler = None
    extra_mle_total = 0
    if has_extra_mle:
        if is_main_process():
            print(f'Interleaved MLE ablation: {len(extra_smiles_list)} SMILES '
                  f'(wild_smiles=True, mol_augment=False)')
            print(f'  batch_size={mrt_batch_size}, weight={mrt_weight}, warmup_epochs={mrt_warmup_epochs}')
        extra_mle_dataset = MoleculeDataset(
            smiles_list=extra_smiles_list, wild_smiles=True, shuffle_smiles=True,
            vocab=vocab, image_size=image_size, mol_augment=False,
            geo_augment=True, img_augment=True,
            render_badlist_path=render_badlist_path,
        )
        extra_mle_total = len(extra_mle_dataset)
        extra_mle_sampler = DistributedSampler(
            extra_mle_dataset, num_replicas=world_size, rank=local_rank,
            shuffle=True, seed=seed,
        ) if world_size > 1 else None
        _emle_nw = min(num_workers, 4)
        _emle_dl_kwargs = dict(
            batch_size=mrt_batch_size, shuffle=(extra_mle_sampler is None),
            sampler=extra_mle_sampler, num_workers=_emle_nw,
            pin_memory=device.type == 'cuda',
            collate_fn=partial(collate_fn, vocab=vocab, max_atoms_limit=max_atoms),
            drop_last=True,
        )
        if _emle_nw > 0:
            _emle_dl_kwargs.update({
                'prefetch_factor': 8, 'persistent_workers': True,
                'multiprocessing_context': 'forkserver',
                'timeout': loader_timeout,
            })
        extra_mle_loader = DataLoader(extra_mle_dataset, **_emle_dl_kwargs)

    # Optimizer + Scheduler
    optimizer = training_utils.create_optimizer(model, encoder_lr, decoder_lr, weight_decay)

    len_micro = len(mle_loader)                # microbatch iterations/epoch
    steps_per_epoch = (len_micro + mle_accum - 1) // mle_accum   # optimizer steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = training_utils.create_cosine_scheduler(optimizer, total_steps, warmup_ratio)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    # Resume
    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    resume_log_dir = None

    if resume_from is not None and os.path.isfile(resume_from):
        ckpt_info = training_utils.load_checkpoint(resume_from, model, optimizer, scheduler, scaler, device)
        start_epoch = ckpt_info['start_epoch']
        best_val_acc = ckpt_info['best_val_acc']
        epochs_no_improve = ckpt_info['epochs_no_improve']
        resume_log_dir = ckpt_info['resume_log_dir']

    global_step = (start_epoch - 1) * steps_per_epoch

    # Compute MRT step cadence (0 = auto). Invariant = the deployed
    # single-accumulation behavior: floor of the per-rank loader-length
    # ratio. Both sides are per-rank loader lengths, so world size cancels
    # and the ratio depends only on the dataset and the batch-size ratio
    # (it sits just under 5 for this recipe, so floor -> 4 on every
    # accumulation/world-size variant - the mrt_accum factor restores the
    # units after mle_accum, since steps_per_epoch counts optimizer steps
    # (= microbatches / mle_accum) while each MRT step consumes mrt_accum
    # loader batches).
    _mrt_every = mrt_every_n_steps
    if _mrt_every <= 0 and mrt_loader is not None:
        _mrt_every = max(1, (steps_per_epoch * mrt_accum) // len(mrt_loader))
    elif _mrt_every <= 0 and extra_mle_loader is not None:
        _mrt_every = max(1, steps_per_epoch // len(extra_mle_loader))
    elif _mrt_every <= 0:
        _mrt_every = 1

    # TensorBoard
    hparams = {
        'batch_size': batch_size, 'mle_accum': mle_accum,
        'effective_batch_size': batch_size * world_size * mle_accum,
        'mrt_batch_size': mrt_batch_size, 'mrt_accum': mrt_accum, 'encoder_lr': encoder_lr,
        'decoder_lr': decoder_lr, 'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio, 'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'image_size': str(image_size), 'n_bins': n_bins, 'backbone': backbone,
        'pretrained': pretrained, 'd_model': d_model, 'nhead': nhead,
        'num_decoder_layers': num_decoder_layers, 'dim_feedforward': dim_feedforward,
        'dropout': dropout, 'max_atoms': max_atoms, 'mol_augment': mol_augment,
        'use_amp': use_amp, 'seed': seed,
        'mrt_weight': mrt_weight, 'mrt_warmup_epochs': mrt_warmup_epochs,
        'mrt_every_n_steps': _mrt_every,
        'rl_max_len': rl_max_len, 'rl_temperature': rl_temperature,
        'rl_n_samples': rl_n_samples, 'rl_sampling_method': rl_sampling_method,
        'reward_mode': reward_mode, 'visual_reward_renderer': visual_reward_renderer,
        'reward_validity_weight': reward_validity_weight,
        'reward_similarity_weight': reward_similarity_weight,
        'reward_exact_match_weight': reward_exact_match_weight,
        'mrt_alpha': mrt_alpha, 'mle_train_samples': len(mle_train_dataset),
        'mrt_train_images': mrt_total_images, 'extra_mle_samples': extra_mle_total,
        'has_extra_mle': has_extra_mle,
        'visual_reward_encoder_path': visual_reward_encoder_path or '',
        'visual_reward_embedding_dim': visual_reward_embedding_dim,
        'resume_from': resume_from, 'finetune_from': finetune_from,
        'encoder_weights': encoder_weights,
        'persistent_workers': True,
    }
    writer, log_dir = training_utils.setup_tensorboard(save_path, resume_log_dir, hparams, log_prefix='joint')
    log_dir = training_utils.broadcast_log_dir(log_dir, device, world_size)

    if is_main_process():
        print(f'\nSteps/epoch: {steps_per_epoch} optimizer steps '
              f'({len(mle_loader)} microbatches, mle_accum={mle_accum}), '
              f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')
        if has_mrt and mrt_loader is not None:
            print(f'MRT backward co-fires with every {_mrt_every} optimizer '
                  f'step(s) (~{steps_per_epoch // _mrt_every} MRT-backward '
                  f'steps/epoch x {mrt_accum} sub-batch(es), '
                  f'MRT loader has {len(mrt_loader)} batches)')
        elif has_extra_mle and extra_mle_loader is not None:
            print(f'Interleaved MLE every {_mrt_every} main-MLE steps '
                  f'(~{steps_per_epoch // _mrt_every} iMLE steps/epoch, '
                  f'iMLE loader has {len(extra_mle_loader)} batches)')
        print(f'Starting from epoch {start_epoch}\n')

    # Training Loop
    mrt_iter = None
    if mrt_loader is not None:
        mrt_iter = iter(mrt_loader)
    extra_mle_iter = None
    if extra_mle_loader is not None:
        extra_mle_iter = iter(extra_mle_loader)
    _gt_render_cache: Dict = {}
    _frozen_refreshed = False

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        mrt_active = (epoch > mrt_warmup_epochs) and has_mrt
        extra_mle_active = (epoch > mrt_warmup_epochs) and has_extra_mle and not has_mrt
        interleaved_active = mrt_active or extra_mle_active

        if mrt_active and not _frozen_refreshed and _needs_frozen_encoder:
            if is_main_process():
                print('Freezing warmup-trained encoder as the visual-reward reference...')
            frozen_encoder = copy.deepcopy(actual_model.image_encoder).eval()
            frozen_pos_enc = copy.deepcopy(actual_model.pos_enc_2d).eval()
            frozen_encoder.requires_grad_(False)
            frozen_pos_enc.requires_grad_(False)
            _frozen_refreshed = True
            _gt_render_cache.clear()

        if mle_sampler is not None:
            mle_sampler.set_epoch(epoch)
        if mrt_active and mrt_sampler is not None:
            mrt_sampler.set_epoch(epoch)
        if extra_mle_active and extra_mle_sampler is not None:
            extra_mle_sampler.set_epoch(epoch)

        running_mle_loss = 0.0
        running_mrt_loss = 0.0
        running_mrt_reward = 0.0
        num_batches = 0
        mrt_steps_done = 0
        _last_mrt_loss = 0.0
        _last_mrt_reward = 0.0

        phase_name = ('MLE+MRT' if mrt_active else 'MLE+iMLE' if extra_mle_active else 'MLE')
        len_micro = len(mle_loader)
        train_bar = tqdm(total=len_micro,
                         desc=f'Epoch {epoch} [{phase_name}]',
                         disable=not is_main_process())
        mle_train_iter = iter(mle_loader)
        _epoch_opt_step = 0   # executed optimizer steps this epoch

        for micro in range(len_micro):
            # group bookkeeping BEFORE the fetch, so an invalid leading
            # micro cannot skip zero_grad and bleed grads across groups
            _micro_last = (micro % mle_accum == mle_accum - 1) or \
                (micro == len_micro - 1)
            _group_size = min(mle_accum,
                              len_micro - (micro - micro % mle_accum))
            if micro % mle_accum == 0:
                optimizer.zero_grad(set_to_none=True)
                _group_valid = True
            with perf_phase('mle_data_fetch'):
                try:
                    batch = next(mle_train_iter)
                except StopIteration:
                    break
                except RuntimeError as e:
                    if is_main_process():
                        warnings.warn(f'MLE DataLoader error at epoch {epoch} micro {micro}: {e}')
                    batch = None
                    del mle_train_iter
                    mle_loader = _make_mle_loader()
                    mle_train_iter = iter(mle_loader)

            with perf_phase('mle_prep'):
                batch_valid = batch is not None
                if world_size > 1:
                    valid_tensor = torch.tensor([1 if batch_valid else 0], dtype=torch.int32, device=device)
                    dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                    batch_valid = valid_tensor.item() == 1
                if not batch_valid:
                    # skip the WHOLE group's update (matching the single-batch
                    # loader-error semantics: one timeout -> no optimizer step
                    # for this slot, rather than a half-magnitude step)
                    _group_valid = False
                    train_bar.update(1)
                    continue

                images = batch['images'].to(device, non_blocking=True).contiguous()
                tgt_tokens = batch['tgt_tokens'].to(device, non_blocking=True).contiguous()
                tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True).contiguous()
                atom_indices = batch['atom_indices'].to(device, non_blocking=True).contiguous()
                atom_mask = batch['atom_mask'].to(device, non_blocking=True).contiguous()
                max_atoms_val = batch['max_atoms']
                bond_matrices_list = batch['bond_matrices_list']

                tgt_tokens, tgt_padding_mask, atom_indices, atom_mask = \
                    truncate_sequences(tgt_tokens, tgt_padding_mask, atom_indices, atom_mask, max_seq_len=600)

                tgt_input = tgt_tokens[:, :-1].contiguous()
                tgt_target = tgt_tokens[:, 1:].contiguous()
                tgt_input_mask = tgt_padding_mask[:, :-1].contiguous()

            _sync_ctx = model.no_sync() \
                if (world_size > 1 and mle_accum > 1 and not _micro_last) \
                else nullcontext()
            if scaler is not None:
                with perf_phase('mle_fwd'), _sync_ctx, torch.amp.autocast('cuda'):
                    token_logits, edge_logits, _hidden = model(
                        images=images, tgt_tokens=tgt_input,
                        tgt_key_padding_mask=tgt_input_mask,
                        atom_indices=atom_indices, atom_mask=atom_mask,
                        max_atoms=max_atoms_val,
                    )
                    del _hidden
                    mle_losses = compute_losses(
                        token_logits=token_logits, edge_logits=edge_logits,
                        tgt_tokens=tgt_target, bond_matrices_list=bond_matrices_list,
                        vocab=vocab, bond_class_weights=bond_class_weights,
                    )
                with perf_phase('mle_bwd'):
                    scaler.scale(mle_losses['total_loss']
                                 / _group_size).backward()
            else:
                with perf_phase('mle_fwd'):
                    token_logits, edge_logits, _hidden = model(
                        images=images, tgt_tokens=tgt_input,
                        tgt_key_padding_mask=tgt_input_mask,
                        atom_indices=atom_indices, atom_mask=atom_mask,
                        max_atoms=max_atoms_val,
                    )
                    del _hidden
                    mle_losses = compute_losses(
                        token_logits=token_logits, edge_logits=edge_logits,
                        tgt_tokens=tgt_target, bond_matrices_list=bond_matrices_list,
                        vocab=vocab, bond_class_weights=bond_class_weights,
                    )
                with perf_phase('mle_bwd'):
                    (mle_losses['total_loss'] / _group_size).backward()

            # ===== Interleaved step (MRT or ablation iMLE) =====
            # one interleaved step per optimizer step (last micro only);
            # the trigger uses the GROUP position (= the single-batch loop's
            # step index), so a skipped group still consumes its slot and
            # the per-epoch MRT count stays deterministic
            mrt_loss_item = 0.0
            mrt_reward = 0.0
            mrt_rl_info = {}
            _did_interleaved = False
            if mrt_active and _micro_last and _group_valid \
                    and ((micro // mle_accum) % _mrt_every == 0):
                # Temporal sub-batching (mrt_accum>1): consume K sequential
                # mrt batches per update, each backward-weighted 1/K, so the
                # update's MRT gradient = mean over K x mrt_batch x world
                # images - the same mean over the same number of images that
                # a single large batch would produce, at the peak memory of
                # ONE per-rank batch. mrt_accum=1 keeps the plain
                # single-batch MRT step.
                _sub_loss_sum = 0.0
                _sub_reward_sum = 0.0
                mrt_rl_info = {}
                for _sub in range(mrt_accum):
                    with perf_phase('mrt_data'):
                        try:
                            mrt_batch = next(mrt_iter, None)
                            if mrt_batch is None:
                                mrt_iter = iter(mrt_loader)
                                mrt_batch = next(mrt_iter)
                        except (RuntimeError, OSError) as e:
                            warnings.warn(f'MRT DataLoader error at epoch {epoch} micro {micro}: {e}')
                            mrt_iter = iter(mrt_loader)
                            mrt_batch = next(mrt_iter)

                    with perf_phase('mrt_enc'):
                        mrt_images, gt_smiles_batch = mrt_batch
                        mrt_images = mrt_images.to(device, non_blocking=True)

                        with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                            img_features = actual_model.image_encoder(mrt_images)
                            img_features = actual_model.pos_enc_2d(img_features)
                        img_features = img_features.float()

                    rl_loss, rl_info = compute_mrt_loss(
                        model=actual_model, img_features=img_features,
                        gt_smiles_list=gt_smiles_batch, vocab=vocab, device=device,
                        max_len=rl_max_len, temperature=rl_temperature,
                        n_samples=rl_n_samples, mrt_alpha=mrt_alpha,
                        reward_validity_weight=reward_validity_weight,
                        reward_similarity_weight=reward_similarity_weight,
                        reward_exact_match_weight=reward_exact_match_weight,
                        reward_mode=reward_mode, frozen_encoder=frozen_encoder,
                        frozen_pos_enc=frozen_pos_enc,
                        visual_reward_encoder=ext_vis_encoder,
                        visual_reward_transforms=ext_vis_transforms,
                        visual_reward_pad_to_square=ext_vis_pad,
                        visual_reward_renderer=visual_reward_renderer,
                        use_amp=use_amp, gt_render_cache=_gt_render_cache,
                        sampling_method=rl_sampling_method,
                        adaptive_n=adaptive_n, adaptive_n_gamma=adaptive_n_gamma,
                        adaptive_n_min=adaptive_n_min,
                        rollout_fast=rollout_fast,
                    )

                    weighted_mrt_loss = (mrt_weight / mrt_accum) * rl_loss
                    with perf_phase('mrt_bwd'):
                        if scaler is not None:
                            scaler.scale(weighted_mrt_loss).backward()
                        else:
                            weighted_mrt_loss.backward()

                    _sub_loss_sum += rl_loss.item()
                    _sub_reward_sum += rl_info['mean_reward']
                    mrt_rl_info = rl_info
                mrt_loss_item = _sub_loss_sum / mrt_accum
                mrt_reward = _sub_reward_sum / mrt_accum
                mrt_steps_done += 1
                _last_mrt_loss = mrt_loss_item
                _last_mrt_reward = mrt_reward
                _did_interleaved = True

            elif extra_mle_active and _micro_last and _group_valid \
                    and ((micro // mle_accum) % _mrt_every == 0):
                try:
                    extra_mle_batch = next(extra_mle_iter, None)
                    if extra_mle_batch is None:
                        extra_mle_iter = iter(extra_mle_loader)
                        extra_mle_batch = next(extra_mle_iter)
                except (RuntimeError, OSError) as e:
                    warnings.warn(f'iMLE DataLoader error at epoch {epoch} micro {micro}: {e}')
                    extra_mle_iter = iter(extra_mle_loader)
                    extra_mle_batch = next(extra_mle_iter)

                e_images = extra_mle_batch['images'].to(device, non_blocking=True)
                e_tgt = extra_mle_batch['tgt_tokens'].to(device, non_blocking=True)
                e_pad = extra_mle_batch['tgt_padding_mask'].to(device, non_blocking=True)
                e_atom_idx = extra_mle_batch['atom_indices'].to(device, non_blocking=True)
                e_atom_m = extra_mle_batch['atom_mask'].to(device, non_blocking=True)
                e_max_atoms = extra_mle_batch['max_atoms']
                e_bonds = extra_mle_batch['bond_matrices_list']

                e_tgt, e_pad, e_atom_idx, e_atom_m = \
                    truncate_sequences(e_tgt, e_pad, e_atom_idx, e_atom_m, max_seq_len=600)
                e_inp = e_tgt[:, :-1].contiguous()
                e_target = e_tgt[:, 1:].contiguous()
                e_inp_mask = e_pad[:, :-1].contiguous()

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        e_tok_logits, e_edge_logits, _h = actual_model(
                            images=e_images, tgt_tokens=e_inp,
                            tgt_key_padding_mask=e_inp_mask,
                            atom_indices=e_atom_idx, atom_mask=e_atom_m,
                            max_atoms=e_max_atoms,
                        )
                        del _h
                        extra_losses = compute_losses(
                            token_logits=e_tok_logits, edge_logits=e_edge_logits,
                            tgt_tokens=e_target, bond_matrices_list=e_bonds,
                            vocab=vocab, bond_class_weights=bond_class_weights,
                        )
                    scaler.scale(mrt_weight * extra_losses['total_loss']).backward()
                else:
                    e_tok_logits, e_edge_logits, _h = actual_model(
                        images=e_images, tgt_tokens=e_inp,
                        tgt_key_padding_mask=e_inp_mask,
                        atom_indices=e_atom_idx, atom_mask=e_atom_m,
                        max_atoms=e_max_atoms,
                    )
                    del _h
                    extra_losses = compute_losses(
                        token_logits=e_tok_logits, edge_logits=e_edge_logits,
                        tgt_tokens=e_target, bond_matrices_list=e_bonds,
                        vocab=vocab, bond_class_weights=bond_class_weights,
                    )
                    (mrt_weight * extra_losses['total_loss']).backward()

                mrt_loss_item = extra_losses['total_loss'].item()
                mrt_steps_done += 1
                _last_mrt_loss = mrt_loss_item
                _last_mrt_reward = 0.0
                _did_interleaved = True

            # Sync + optimizer step + logging: once per optimizer step,
            # only when the group is fully valid (an invalid micro skips the
            # whole update, matching single-batch loader-error semantics)
            if _micro_last and _group_valid:
                with perf_phase('opt_sync'):
                    # Sync accumulated MRT gradients across GPUs (the manual
                    # AVG over already-DDP-averaged MLE grads is a no-op, so it
                    # is safe to include them)
                    if world_size > 1 and _did_interleaved:
                        grads = [p.grad for p in model.parameters()
                                 if p.grad is not None]
                        if grads:
                            flat = torch.cat([g.reshape(-1) for g in grads])
                            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
                            offset = 0
                            for g in grads:
                                n = g.numel()
                                g.copy_(flat[offset:offset + n].view_as(g))
                                offset += n
                            del flat

                    # Optimizer step
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    scheduler.step()

                running_mle_loss += mle_losses['total_loss'].item()
                running_mrt_loss += mrt_loss_item
                running_mrt_reward += mrt_reward
                num_batches += 1
                _epoch_opt_step += 1
                global_step += 1

                if is_main_process():
                    if global_step % (10 if os.environ.get('FAST_TEST', '') == '1' else 50) == 0:
                        perf_flush(f'e{epoch} s{global_step}/{total_steps}')
                    postfix = {
                        'mle': f"{mle_losses['total_loss'].item():.2f}",
                        't': f"{mle_losses['token_loss'].item():.2f}",
                        'b': f"{mle_losses['bond_loss'].item():.2f}",
                    }
                    if mrt_active:
                        postfix['mrt'] = f"{_last_mrt_loss:.3f}"
                        postfix['R'] = f"{_last_mrt_reward:.3f}"
                    elif extra_mle_active:
                        postfix['iMLE'] = f"{_last_mrt_loss:.3f}"
                    train_bar.set_postfix(postfix)

                    writer.add_scalar('Loss/mle_total', mle_losses['total_loss'].item(), global_step)
                    writer.add_scalar('Loss/mle_token', mle_losses['token_loss'].item(), global_step)
                    writer.add_scalar('Loss/mle_bond', mle_losses['bond_loss'].item(), global_step)
                    writer.add_scalar('LR/encoder', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('LR/decoder', optimizer.param_groups[1]['lr'], global_step)

                    if mrt_active and mrt_rl_info:
                        writer.add_scalar('RL/loss', mrt_loss_item, global_step)
                        writer.add_scalar('RL/reward_composite', mrt_reward, global_step)
                        writer.add_scalar('RL/similarity', mrt_rl_info.get('mean_similarity', 0.0), global_step)
                        writer.add_scalar('RL/valid_smiles', mrt_rl_info.get('valid_smiles_count', 0), global_step)
                        writer.add_scalar('RL/exact_matches', mrt_rl_info.get('exact_matches', 0), global_step)
                        writer.add_scalar('RL/render_success', mrt_rl_info.get('render_success', 0), global_step)
                        if mrt_rl_info.get('adaptive_n_mean') is not None:
                            writer.add_scalar('RL/adaptive_n_mean', mrt_rl_info['adaptive_n_mean'], global_step)
                    elif extra_mle_active and _did_interleaved:
                        writer.add_scalar('InterleavedMLE/loss', mrt_loss_item, global_step)

            train_bar.update(1)

        train_bar.close()

        # Epoch summary
        avg_mle_loss = running_mle_loss / max(num_batches, 1)
        avg_mrt_loss = running_mrt_loss / max(mrt_steps_done, 1)
        avg_mrt_reward = running_mrt_reward / max(mrt_steps_done, 1)
        if is_main_process():
            msg = f'Epoch {epoch} [{phase_name}] — avg_mle_loss: {avg_mle_loss:.4f}'
            if mrt_active:
                msg += (f', avg_mrt_loss: {avg_mrt_loss:.4f}'
                        f', avg_reward: {avg_mrt_reward:.4f}'
                        f' ({mrt_steps_done} MRT steps)')
            elif extra_mle_active:
                msg += f', avg_iMLE_loss: {avg_mrt_loss:.4f} ({mrt_steps_done} iMLE steps)'
            print(msg)

        # Validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        primary_val_metrics = training_utils.run_validation(
            model=model, benchmarks=benchmarks, device=device,
            epoch=epoch, writer=writer, global_step=global_step,
            val_max_samples=val_max_samples,
        )

        val_acc = primary_val_metrics.get('exact_match_acc', 0.0) if primary_val_metrics else 0.0
        should_stop, best_val_acc, epochs_no_improve = training_utils.check_early_stopping(
            val_acc, best_val_acc, epochs_no_improve, early_stopping_patience, model, save_path,
        )

        if is_main_process():
            training_utils.save_checkpoint(save_path, model, optimizer, scheduler, scaler,
                            epoch, best_val_acc, epochs_no_improve, log_dir)

        should_stop = training_utils.sync_early_stopping(should_stop, device, world_size)

        if should_stop:
            if is_main_process():
                print(f'Early stopping at epoch {epoch}')
                training_utils.final_save(model, save_path, writer)
            if world_size > 1:
                dist.barrier()
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            cleanup_ddp()
            return []

        if world_size > 1:
            dist.barrier()
        model.train()

    # Final save
    if is_main_process():
        training_utils.final_save(model, save_path, writer)
        print('Joint training complete!')

    gc.collect()
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    cleanup_ddp()
    return []
