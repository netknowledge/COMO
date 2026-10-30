"""
Shared Training Infrastructure for COMO
======================================

Extracted from como_core.py; lives in ``training/utils.py`` (shared
infrastructure for ``train_mle``/``train_mrt``/``train_joint``).  Imports
``como`` (the pip package) and sibling ``training`` modules directly —
no ``como_model`` facade dependency.
"""

import os
import json
import warnings
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from functools import partial


# ---------------------------------------------------------------------------
# DDP helpers (thin wrappers — kept in como_core.py for portability,
# re-exported here for convenience)
# ---------------------------------------------------------------------------

def init_ddp() -> Tuple[torch.device, int, int]:
    """Initialise Distributed Data Parallel from torchrun env vars.

    Returns:
        (device, local_rank, world_size)
    """
    # Lazy import to avoid circular dependency
    from .ddp_utils import setup_ddp, is_main_process

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')

    if world_size > 1:
        setup_ddp(local_rank, world_size)

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if is_main_process():
        if world_size > 1:
            print(f'DDP Training: {world_size} GPUs')
            for i in range(world_size):
                print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print(f'Single GPU training on: {device}')

    return device, local_rank, world_size


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

def create_optimizer(
    model: torch.nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create AdamW with separate encoder / decoder learning rates."""
    actual_model = model.module if hasattr(model, 'module') else model
    encoder_param_ids = {id(p) for p in actual_model.image_encoder.parameters()}
    encoder_params = [p for p in actual_model.parameters() if id(p) in encoder_param_ids]
    decoder_params = [p for p in actual_model.parameters() if id(p) not in encoder_param_ids]

    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': weight_decay},
    ])


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine-annealed LR with linear warmup."""
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# MLE DataLoader factory
# ---------------------------------------------------------------------------

def create_mle_dataloader(
    *,
    vocab,
    mle_smiles_path: str,
    mle_smiles_num: int,
    mle_batch_size: int,
    image_size: Tuple[int, int],
    seed: int,
    world_size: int,
    local_rank: int,
    num_workers: int,
    device: torch.device,
    collate_fn_factory,
) -> Optional[DataLoader]:
    """Create an MLE regularization DataLoader from a PubChem SMILES pickle.

    Returns ``None`` if *mle_smiles_path* is falsy.
    """
    if not mle_smiles_path:
        return None

    import pickle
    from .datasets import MoleculeDataset
    from .ddp_utils import is_main_process

    if is_main_process():
        print(f'\n--- MLE Regularization ---')
        print(f'  Loading SMILES from: {mle_smiles_path}')

    if str(mle_smiles_path).endswith('.txt'):
        # Plain-text list (one SMILES per line), e.g. the pinned
        # pubchem_1M_seed2026.txt released on Hugging Face.
        with open(mle_smiles_path) as f:
            all_smiles = [line.strip() for line in f if line.strip()]
    else:
        with open(mle_smiles_path, 'rb') as f:
            all_smiles = pickle.load(f)

    rng = np.random.default_rng(seed)
    n = min(mle_smiles_num, len(all_smiles))
    indices = rng.choice(len(all_smiles), size=n, replace=False)
    mle_smiles_list = [all_smiles[i] for i in indices]
    del all_smiles

    if is_main_process():
        print(f'  Sampled {len(mle_smiles_list)} SMILES for MLE')

    mle_dataset = MoleculeDataset(
        vocab=vocab,
        smiles_list=mle_smiles_list,
        shuffle_smiles=True,
        image_size=image_size,
        mol_augment=True,
        geo_augment=True,
        img_augment=True,
    )

    mle_sampler = DistributedSampler(
        mle_dataset, num_replicas=world_size, rank=local_rank,
        shuffle=True, seed=seed,
    ) if world_size > 1 else None

    eff_workers = max(1, num_workers // 2)
    loader = DataLoader(
        mle_dataset,
        batch_size=mle_batch_size,
        shuffle=(mle_sampler is None),
        sampler=mle_sampler,
        num_workers=eff_workers,
        pin_memory=device.type == 'cuda',
        collate_fn=collate_fn_factory(vocab=vocab),
        drop_last=True,
        prefetch_factor=4 if eff_workers > 0 else None,
        persistent_workers=eff_workers > 0,
    )

    if is_main_process():
        print(f'  MLE DataLoader: {len(loader)} batches/epoch (batch_size={mle_batch_size})')

    return loader


# ---------------------------------------------------------------------------
# Checkpoint resume / save
# ---------------------------------------------------------------------------

def _remap_compiled_keys(sd: Dict[str, Any], model: torch.nn.Module) -> Dict[str, Any]:
    # Checkpoints saved before/after the image_encoder got torch.compile()-wrapped
    # differ by the ``_orig_mod.`` segment; normalize against the live model so
    # either vintage resumes.
    keys = set(model.state_dict().keys())
    out = {}
    for k, v in sd.items():
        if k in keys:
            pass                                        # exact match
        elif '._orig_mod.' in k and k.replace('._orig_mod.', '.') in keys:
            k = k.replace('._orig_mod.', '.')           # ckpt compiled, model bare
        else:
            head, _, tail = k.partition('.')
            w = f'{head}._orig_mod.{tail}'              # ckpt bare, model compiled
            if w in keys:
                k = w
        out[k] = v
    return out


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    """Load a full training checkpoint and restore all state.

    Returns a dict with keys:
        start_epoch, best_val_acc, epochs_no_improve, resume_log_dir
    """
    from .ddp_utils import is_main_process

    if is_main_process():
        print(f'Resuming from checkpoint: {path}')

    ckpt = torch.load(path, map_location=device, weights_only=False)

    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.load_state_dict(_remap_compiled_keys(ckpt['model_state_dict'], actual_model))

    if ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    saved_epoch = ckpt.get('epoch', 0)
    start_epoch = saved_epoch + 1
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    epochs_no_improve = ckpt.get('epochs_no_improve', 0)
    resume_log_dir = ckpt.get('log_dir', None)

    if is_main_process():
        print(f'  Loaded epoch {saved_epoch}, resuming from epoch {start_epoch}')
        print(f'  best_val_acc={best_val_acc:.4f}, epochs_no_improve={epochs_no_improve}')

    del ckpt
    torch.cuda.empty_cache()

    return {
        'start_epoch': start_epoch,
        'best_val_acc': best_val_acc,
        'epochs_no_improve': epochs_no_improve,
        'resume_log_dir': resume_log_dir,
    }


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_val_acc: float,
    epochs_no_improve: int,
    log_dir: str,
    *,
    save_epoch: bool = True,
) -> None:
    """Save resume checkpoint (and optionally epoch snapshot) on rank 0."""
    actual_model = model.module if hasattr(model, 'module') else model
    ckpt = {
        'epoch': epoch,
        'model_state_dict': actual_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_acc': best_val_acc,
        'epochs_no_improve': epochs_no_improve,
        'log_dir': log_dir,
    }
    torch.save(ckpt, os.path.join(save_path, 'checkpoint_resume.pth'))
    if save_epoch:
        torch.save(actual_model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
        # Full-state epoch snapshot: --resume_from checkpoint_epoch{N}.pth
        # restarts at epoch {N}+1 (any epoch boundary incl. the MRT start).
        torch.save(ckpt, os.path.join(save_path, f'checkpoint_epoch{epoch}.pth'))
    del ckpt


# ---------------------------------------------------------------------------
# TensorBoard setup
# ---------------------------------------------------------------------------

def setup_tensorboard(
    save_path: str,
    resume_log_dir: Optional[str],
    hparams: Dict[str, Any],
    *,
    log_prefix: str = '',
) -> Tuple[Optional[SummaryWriter], Optional[str]]:
    """Create TensorBoard writer + save hparams on rank 0.

    Returns (writer, log_dir).  Both are None on non-main ranks.
    """
    from .ddp_utils import is_main_process

    if not is_main_process():
        return None, None

    os.makedirs(save_path, exist_ok=True)

    if resume_log_dir and os.path.isdir(resume_log_dir):
        log_dir = resume_log_dir
        print(f'TensorBoard logs (resumed): {log_dir}')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        subdir = f'{log_prefix}_{timestamp}' if log_prefix else timestamp
        log_dir = os.path.join(save_path, 'logs', subdir)
        print(f'TensorBoard logs (new): {log_dir}')

    writer = SummaryWriter(log_dir=log_dir)

    # Save hyperparameters as JSON + TensorBoard text
    hparams_path = os.path.join(log_dir, 'hparams.json')
    with open(hparams_path, 'w') as f:
        json.dump(hparams, f, indent=2)
    hparams_md = '| Param | Value |\n|---|---|\n'
    for k, v in hparams.items():
        hparams_md += f'| {k} | {v} |\n'
    writer.add_text('Hyperparameters', hparams_md, global_step=0)

    return writer, log_dir


# ---------------------------------------------------------------------------
# Broadcast log_dir across DDP ranks
# ---------------------------------------------------------------------------

def broadcast_log_dir(
    log_dir: Optional[str],
    device: torch.device,
    world_size: int,
) -> str:
    """Broadcast *log_dir* from rank 0 to all ranks so checkpoints can record it."""
    from .ddp_utils import is_main_process

    if world_size <= 1:
        return log_dir  # type: ignore[return-value]

    if is_main_process():
        log_dir_bytes = log_dir.encode('utf-8')  # type: ignore[union-attr]
        log_dir_len = torch.tensor([len(log_dir_bytes)], dtype=torch.long, device=device)
    else:
        log_dir_len = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(log_dir_len, src=0)

    if is_main_process():
        log_dir_tensor = torch.tensor(list(log_dir_bytes), dtype=torch.uint8, device=device)
    else:
        log_dir_tensor = torch.zeros(int(log_dir_len.item()), dtype=torch.uint8, device=device)

    dist.broadcast(log_dir_tensor, src=0)

    if not is_main_process():
        log_dir = bytes(log_dir_tensor.cpu().tolist()).decode('utf-8')

    return log_dir  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MLE forward + backward step (used inside both training loops)
# ---------------------------------------------------------------------------

def mle_forward_backward(
    *,
    model: torch.nn.Module,
    mle_iter,
    mle_loader: DataLoader,
    mle_weight: float,
    vocab,
    bond_class_weights,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, Any]:
    """Run one MLE regularization forward + backward pass.

    Gradients are accumulated (caller should have already called
    ``optimizer.zero_grad()``).

    Returns ``(mle_loss_item, mle_iter)`` where *mle_iter* may have been
    recycled if the underlying loader was exhausted.
    """
    from .losses import compute_losses
    from como.utils import truncate_sequences

    mle_batch = next(mle_iter, None)
    if mle_batch is None:
        mle_iter = iter(mle_loader)
        mle_batch = next(mle_iter, None)

    if mle_batch is None or mle_batch.get('images') is None:
        return 0.0, mle_iter

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

    actual_model = model.module if hasattr(model, 'module') else model

    with torch.amp.autocast('cuda', enabled=(use_amp and scaler is not None)):
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

    mle_loss_scaled = mle_weight * mle_losses['total_loss']
    if scaler is not None:
        scaler.scale(mle_loss_scaled).backward()
    else:
        mle_loss_scaled.backward()

    return mle_losses['total_loss'].item(), mle_iter


# ---------------------------------------------------------------------------
# Validation + early stopping
# ---------------------------------------------------------------------------

def run_validation(
    *,
    model: torch.nn.Module,
    benchmarks: List[Dict[str, str]],
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    global_step: int,
    val_max_samples: Optional[int] = None,
) -> Optional[Dict]:
    """Run validation on all benchmarks, return primary metrics (first benchmark)."""
    from .ddp_validation import validate

    primary_val_metrics = None
    for bm_idx, bm in enumerate(benchmarks):
        val_metrics = validate(
            model=model,
            benchmark_dir=bm['dir'],
            benchmark_csv_path=bm['csv'],
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            beam_size=1,
            max_samples=val_max_samples,
            benchmark_name=bm['name'],
        )
        if bm_idx == 0:
            primary_val_metrics = val_metrics

    return primary_val_metrics


def check_early_stopping(
    val_acc: float,
    best_val_acc: float,
    epochs_no_improve: int,
    patience: int,
    model: torch.nn.Module,
    save_path: str,
) -> Tuple[bool, float, int]:
    """Check early stopping on rank 0.

    Returns ``(improved, new_best_val_acc, new_epochs_no_improve)``.
    """
    from .ddp_utils import is_main_process

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        if is_main_process():
            actual_model = model.module if hasattr(model, 'module') else model
            torch.save(actual_model.state_dict(), os.path.join(save_path, 'best.pth'))
            print(f'New best model saved (exact_match_acc={best_val_acc:.4f}).')
    else:
        epochs_no_improve += 1
        if is_main_process():
            print(f'No improvement for {epochs_no_improve} epoch(s) (patience={patience}).')

    return (epochs_no_improve >= patience and patience > 0), best_val_acc, epochs_no_improve


def sync_early_stopping(should_stop: bool, device: torch.device, world_size: int) -> bool:
    """Broadcast early-stopping decision from rank 0 to all ranks."""
    if world_size <= 1:
        return should_stop
    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


def final_save(model: torch.nn.Module, save_path: str, writer: Optional[SummaryWriter]) -> None:
    """Save final model weights and close TensorBoard writer."""
    from .ddp_utils import is_main_process
    if is_main_process():
        actual_model = model.module if hasattr(model, 'module') else model
        torch.save(actual_model.state_dict(), os.path.join(save_path, 'final.pth'))
        if writer:
            writer.close()
