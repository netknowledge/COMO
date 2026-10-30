"""
COMO Training — MRT (Minimum Risk Training) Finetuning
=======================================================

RL finetuning on real-world molecule images using MRT sequence-level
loss with composite reward signals.

Reward modes:
  - ``'visual'``        — cycle-consistency cosine similarity
  - ``'tanimoto'``      — Morgan fingerprint Tanimoto
  - ``'edit_distance'`` — normalized Levenshtein

Launch with torchrun::

    torchrun --nproc_per_node=<N> scripts/train_stage2.py

Private module — not included in the pip-publishable ``como`` distribution.
"""

import copy
import gc
import os
import random
from typing import Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from como.vocab import ComoVocab
from como.model import ComoModel
from como.utils import truncate_sequences

from .datasets.collate import collate_fn
from .datasets.real_image_dataset import RealImageDataset, _real_collate_fn

from training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process
from training.losses import compute_losses, get_bond_class_weights
from training.reward import compute_mrt_loss, load_visual_reward_encoder
from . import utils as training_utils


def train_mrt(
    # ===== Data =====
    train_csv_paths: List[str],
    train_image_dirs: List[str],
    # ===== Validation =====
    benchmarks: Optional[List[Dict[str, str]]] = None,
    val_max_samples: Optional[int] = None,
    val_benchmark_dir: str = '',
    val_benchmark_csv: str = '',
    # ===== Model =====
    pretrained_path: str = '',
    save_path: str = '',
    # ===== Architecture =====
    image_size: Tuple[int, int] = (384, 384),
    n_bins: int = 64,
    backbone: str = 'swin_b',
    d_model: int = 256,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    # ===== Training =====
    num_epochs: int = 10,
    batch_size: int = 16,
    encoder_lr: float = 1e-5,
    decoder_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    warmup_ratio: float = 0.02,
    seed: int = 2026,
    early_stopping_patience: int = 10,
    num_workers: int = 4,
    use_amp: bool = True,
    # ===== RL =====
    mrt_max_len: int = 500,
    mrt_temperature: float = 0.8,
    mrt_n_samples: int = 16,
    mrt_sampling_method: str = 'multinomial',
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
    # ===== MLE regularization =====
    mle_weight: float = 0.0,
    mle_smiles_path: Optional[str] = None,
    mle_smiles_num: int = 50000,
    mle_batch_size: int = 8,
    # ===== Resume =====
    resume_from: Optional[str] = None,
) -> List[float]:
    """MRT finetuning on real-world molecule images with MRT composite reward.

    Finetunes a pretrained COMO model using MRT sequence-level MRT loss::

        R = w_v · 𝟙[valid] + w_sim · Similarity + w_e · 𝟙[exact match]

    A frozen copy of the pretrained encoder provides stable visual reward
    signals throughout training.

    Args:
        train_csv_paths: List of CSV paths for real-image MRT data.
        train_image_dirs: Corresponding image directories.
        pretrained_path: Path to pretrained COMO checkpoint.
        save_path: Output directory for checkpoints / logs.
        benchmarks: Validation benchmarks (first is primary).
        reward_mode: ``'visual'``, ``'tanimoto'``, or ``'edit_distance'``.
        visual_reward_renderer: ``'rdkit'`` or ``'drawing_engine'``.
        batch_size: Per-GPU batch size.
    """
    # ===== DDP setup =====
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        setup_ddp(local_rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        raise RuntimeError('CUDA is required for RL finetuning')

    if is_main_process():
        print(f'=== Real-Image RL Finetuning (cycle consistency) ===')
        print(f'Device: {device}, World size: {world_size}')
        print(f'Train CSVs: {train_csv_paths}')
        print(f'mrt_temperature={mrt_temperature}, mrt_n_samples={mrt_n_samples}, '
              f'mrt_sampling_method={mrt_sampling_method}')
        print(f'reward weights: validity={reward_validity_weight}, '
              f'similarity={reward_similarity_weight}, '
              f'exact_match={reward_exact_match_weight}')
        print(f'mrt_alpha={mrt_alpha}, reward_mode={reward_mode}')
        print(f'visual_reward_renderer={visual_reward_renderer}')
        if visual_reward_encoder_path:
            print(f'visual_reward_encoder_path={visual_reward_encoder_path}')

    if visual_reward_renderer not in {'rdkit', 'drawing_engine'}:
        raise ValueError(
            f"Unknown visual_reward_renderer='{visual_reward_renderer}'. "
            "Choose from 'rdkit' or 'drawing_engine'."
        )

    # Normalize benchmark list
    if benchmarks is None:
        if val_benchmark_dir and val_benchmark_csv:
            benchmarks = [{'name': os.path.splitext(os.path.basename(val_benchmark_csv))[0],
                           'dir': val_benchmark_dir, 'csv': val_benchmark_csv}]
        else:
            benchmarks = []
    if is_main_process() and benchmarks:
        print(f'Validation benchmarks ({len(benchmarks)}):')
        for bm in benchmarks:
            print(f'  {bm["name"]}: {bm["csv"]}')

    # Vocab + Model
    vocab = ComoVocab(n_bins=n_bins)

    model = ComoModel(
        vocab=vocab,
        backbone=backbone,
        pretrained=False,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        image_size=image_size,
    ).to(device)

    if is_main_process():
        print(f'Loading pretrained weights from {pretrained_path} ...')
    model.load_model(pretrained_path, device=device)

    # Frozen reference encoder for stable visual reward
    frozen_encoder = copy.deepcopy(model.image_encoder).eval()
    frozen_pos_enc = copy.deepcopy(model.pos_enc_2d).eval()
    frozen_encoder.requires_grad_(False)
    frozen_pos_enc.requires_grad_(False)

    # Optional external visual reward encoder
    ext_vis_encoder = None
    ext_vis_transforms = None
    ext_vis_pad = None
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

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                    gradient_as_bucket_view=True)

    actual_model = model.module if hasattr(model, 'module') else model

    # Dataset
    all_items = []
    for csv_path, img_dir in zip(train_csv_paths, train_image_dirs):
        ds = RealImageDataset(csv_path, img_dir, actual_model.inference_transforms, image_size)
        all_items.extend(ds.items)
        if is_main_process():
            print(f'  Loaded {len(ds)} images from {csv_path}')

    train_dataset = RealImageDataset.__new__(RealImageDataset)
    train_dataset.items = all_items
    train_dataset.transforms = actual_model.inference_transforms
    train_dataset.image_size = image_size

    if is_main_process():
        print(f'Total training images: {len(train_dataset)}')

    # FAST_TEST data limit (the smoke path): strided subset of the whole
    # MRT pool so an epoch ends in minutes. FAST_TEST does NOT shrink the
    # batch recipe.
    if os.environ.get('FAST_TEST', '') == '1':
        from torch.utils.data import Subset
        _mrt_limit = int(os.environ.get('FAST_TEST_MRT_LIMIT', '512'))
        if len(train_dataset.items) > _mrt_limit:
            _stride = max(1, len(train_dataset.items) // _mrt_limit)
            _ind = np.arange(0, len(train_dataset.items), _stride)[:_mrt_limit]
            _orig_len = len(train_dataset.items)
            train_dataset = Subset(train_dataset, _ind.tolist())
            if is_main_process():
                print(f'[FAST_TEST] MRT limited: {len(_ind)}/{_orig_len} '
                      f'(stride {_stride})')

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank,
        shuffle=True, seed=seed,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=_real_collate_fn,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # MLE Regularization DataLoader
    mle_loader = None
    bond_class_weights = None
    if mle_weight > 0 and mle_smiles_path:
        if is_main_process():
            print(f'  MLE Weight: {mle_weight}')
        mle_loader = training_utils.create_mle_dataloader(
            vocab=vocab, mle_smiles_path=mle_smiles_path,
            mle_smiles_num=mle_smiles_num, mle_batch_size=mle_batch_size,
            image_size=image_size, seed=seed, world_size=world_size,
            local_rank=local_rank, num_workers=num_workers, device=device,
            collate_fn_factory=partial(partial, collate_fn),
        )
        bond_class_weights = get_bond_class_weights()

    # Optimizer + Scheduler
    optimizer = training_utils.create_optimizer(model, encoder_lr, decoder_lr, weight_decay)

    steps_per_epoch = len(train_loader)
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

    # TensorBoard
    hparams = {
        'batch_size': batch_size, 'effective_batch_size': batch_size * world_size,
        'encoder_lr': encoder_lr, 'decoder_lr': decoder_lr, 'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio, 'num_epochs': num_epochs,
        'image_size': str(image_size), 'n_bins': n_bins, 'backbone': backbone,
        'd_model': d_model, 'nhead': nhead, 'num_decoder_layers': num_decoder_layers,
        'dim_feedforward': dim_feedforward, 'dropout': dropout,
        'mrt_max_len': mrt_max_len, 'mrt_temperature': mrt_temperature,
        'mrt_n_samples': mrt_n_samples, 'mrt_sampling_method': mrt_sampling_method,
        'reward_mode': reward_mode, 'visual_reward_renderer': visual_reward_renderer,
        'reward_validity_weight': reward_validity_weight,
        'reward_similarity_weight': reward_similarity_weight,
        'reward_exact_match_weight': reward_exact_match_weight,
        'mrt_alpha': mrt_alpha, 'seed': seed,
        'pretrained_path': pretrained_path, 'train_csv_paths': train_csv_paths,
        'train_images_total': len(train_dataset),
        'mle_weight': mle_weight, 'mle_smiles_path': mle_smiles_path,
        'mle_smiles_num': mle_smiles_num if mle_weight > 0 else 0,
        'mle_batch_size': mle_batch_size if mle_weight > 0 else 0,
        'visual_reward_encoder_path': visual_reward_encoder_path or '',
        'visual_reward_embedding_dim': visual_reward_embedding_dim,
    }
    writer, log_dir = training_utils.setup_tensorboard(save_path, resume_log_dir, hparams, log_prefix='rl')
    log_dir = training_utils.broadcast_log_dir(log_dir, device, world_size)

    if is_main_process():
        print(f'\nSteps/epoch: {steps_per_epoch}, Total steps: {total_steps}, '
              f'Warmup steps: {warmup_steps}')
        print(f'Starting from epoch {start_epoch}\n')

    # Training Loop
    mle_iter = None
    if mle_loader is not None:
        mle_iter = iter(mle_loader)

    _gt_render_cache: Dict = {} if reward_mode == 'visual' else {}

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if mle_loader is not None and hasattr(mle_loader, 'sampler') and hasattr(mle_loader.sampler, 'set_epoch'):
            mle_loader.sampler.set_epoch(epoch)

        running_reward = 0.0
        running_loss = 0.0
        running_mle_loss = 0.0
        num_batches = 0

        pbar = (tqdm(train_loader, desc=f'Epoch {epoch}', dynamic_ncols=True)
                if is_main_process() else train_loader)

        for images, gt_smiles_batch in pbar:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                img_features = actual_model.image_encoder(images)
                img_features = actual_model.pos_enc_2d(img_features)

            img_features = img_features.float()

            mrt_loss, mrt_info = compute_mrt_loss(
                model=actual_model,
                img_features=img_features,
                gt_smiles_list=gt_smiles_batch,
                vocab=vocab,
                device=device,
                max_len=mrt_max_len,
                temperature=mrt_temperature,
                n_samples=mrt_n_samples,
                mrt_alpha=mrt_alpha,
                reward_validity_weight=reward_validity_weight,
                reward_similarity_weight=reward_similarity_weight,
                reward_exact_match_weight=reward_exact_match_weight,
                reward_mode=reward_mode,
                frozen_encoder=frozen_encoder,
                frozen_pos_enc=frozen_pos_enc,
                visual_reward_encoder=ext_vis_encoder,
                visual_reward_transforms=ext_vis_transforms,
                visual_reward_pad_to_square=ext_vis_pad,
                visual_reward_renderer=visual_reward_renderer,
                use_amp=use_amp,
                gt_render_cache=_gt_render_cache,
                sampling_method=mrt_sampling_method,
                adaptive_n=adaptive_n,
                adaptive_n_gamma=adaptive_n_gamma,
                adaptive_n_min=adaptive_n_min,
            )

            mrt_reward = mrt_info['mean_reward']

            # MLE regularization forward
            mle_loss_item = 0.0
            mle_token_loss_item = 0.0
            mle_bond_loss_item = 0.0
            mle_loss_tensor = torch.tensor(0.0, device=device)
            if mle_loader is not None:
                mle_batch = next(mle_iter, None)
                if mle_batch is None:
                    mle_iter = iter(mle_loader)
                    mle_batch = next(mle_iter, None)
                if mle_batch is not None and mle_batch.get('images') is not None:
                    mle_images = mle_batch['images'].to(device, non_blocking=True)
                    mle_tgt_tokens = mle_batch['tgt_tokens'].to(device, non_blocking=True)
                    mle_tgt_padding_mask = mle_batch['tgt_padding_mask'].to(device, non_blocking=True)
                    mle_atom_indices = mle_batch['atom_indices'].to(device, non_blocking=True)
                    mle_atom_mask = mle_batch['atom_mask'].to(device, non_blocking=True)
                    mle_max_atoms = mle_batch['max_atoms']
                    mle_bond_matrices = mle_batch['bond_matrices_list']

                    mle_tgt_tokens, mle_tgt_padding_mask, mle_atom_indices, mle_atom_mask = \
                        truncate_sequences(mle_tgt_tokens, mle_tgt_padding_mask,
                                           mle_atom_indices, mle_atom_mask, max_seq_len=600)

                    mle_tgt_input = mle_tgt_tokens[:, :-1].contiguous()
                    mle_tgt_target = mle_tgt_tokens[:, 1:].contiguous()
                    mle_tgt_input_mask = mle_tgt_padding_mask[:, :-1].contiguous()

                    with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                        mle_token_logits, mle_edge_logits, _hidden = actual_model(
                            images=mle_images,
                            tgt_tokens=mle_tgt_input,
                            tgt_key_padding_mask=mle_tgt_input_mask,
                            atom_indices=mle_atom_indices,
                            atom_mask=mle_atom_mask,
                            max_atoms=mle_max_atoms,
                        )
                        del _hidden
                        mle_losses = compute_losses(
                            token_logits=mle_token_logits,
                            edge_logits=mle_edge_logits,
                            tgt_tokens=mle_tgt_target,
                            bond_matrices_list=mle_bond_matrices,
                            vocab=vocab,
                            bond_class_weights=bond_class_weights,
                        )
                    mle_loss_tensor = mle_weight * mle_losses['total_loss']
                    mle_loss_item = mle_losses['total_loss'].item()
                    mle_token_loss_item = mle_losses['token_loss'].item()
                    mle_bond_loss_item = mle_losses['bond_loss'].item()

            # Combined backward + optimizer step
            total_loss = mrt_loss + mle_loss_tensor
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            running_reward += mrt_reward
            running_loss += mrt_loss.item()
            running_mle_loss += mle_loss_item
            num_batches += 1
            global_step += 1

            if is_main_process():
                if hasattr(pbar, 'set_postfix'):
                    postfix = {
                        'reward': f"{mrt_reward:.3f}",
                        'loss': f"{mrt_loss.item():.3f}",
                        'valid': mrt_info.get('valid_smiles_count', 0),
                        'exact': mrt_info.get('exact_matches', 0),
                    }
                    if mle_loader is not None:
                        postfix['mle'] = f"{mle_loss_item:.3f}"
                    pbar.set_postfix(postfix)

                if writer:
                    writer.add_scalar('MRT/reward_composite', mrt_reward, global_step)
                    writer.add_scalar('MRT/similarity', mrt_info.get('mean_similarity', 0.0), global_step)
                    writer.add_scalar('MRT/valid_smiles', mrt_info.get('valid_smiles_count', 0), global_step)
                    writer.add_scalar('MRT/exact_matches', mrt_info.get('exact_matches', 0), global_step)
                    writer.add_scalar('MRT/loss', mrt_loss.item(), global_step)
                    writer.add_scalar('MRT/render_success', mrt_info.get('render_success', 0), global_step)
                    if mrt_info.get('adaptive_n_mean') is not None:
                        writer.add_scalar('MRT/adaptive_n_mean', mrt_info['adaptive_n_mean'], global_step)
                    writer.add_scalar('LR/encoder', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('LR/decoder', optimizer.param_groups[1]['lr'], global_step)
                    if mle_loader is not None:
                        writer.add_scalar('MLE/loss', mle_loss_item, global_step)

        # Epoch summary
        avg_reward = running_reward / max(num_batches, 1)
        avg_loss = running_loss / max(num_batches, 1)
        avg_mle_loss = running_mle_loss / max(num_batches, 1)
        if is_main_process():
            msg = f'\nEpoch {epoch}: avg_reward={avg_reward:.4f}, avg_mrt_loss={avg_loss:.4f}'
            if mle_loader is not None:
                msg += f', avg_mle_loss={avg_mle_loss:.4f}'
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
            val_acc, best_val_acc, epochs_no_improve,
            early_stopping_patience, model, save_path,
        )

        if is_main_process():
            training_utils.save_checkpoint(save_path, model, optimizer, scheduler, scaler,
                            epoch, best_val_acc, epochs_no_improve, log_dir)

        if should_stop:
            if is_main_process():
                print(f'Early stopping at epoch {epoch}')
            break

        if world_size > 1:
            dist.barrier()
        model.train()

    # Final save
    if is_main_process():
        training_utils.final_save(model, save_path, writer)
        print('MRT finetuning complete!')

    gc.collect()
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    cleanup_ddp()
    return []
