"""
COMO Training — MLE (Maximum Likelihood Estimation)
====================================================

Supervised MLE training on synthetic/USPTO molecule data with DDP,
cosine-annealed LR, AMP, and per-epoch validation.

Launch with torchrun::

    torchrun --nproc_per_node=<N_GPUS> scripts/train_stage1.py

Private module — not included in the pip-publishable ``como`` distribution.
"""

import os
import gc
import random
import warnings
from typing import Dict, List, Optional, Tuple
from functools import partial

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

from training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process
from training.losses import compute_losses, get_bond_class_weights
from . import utils as training_utils


def train_mle(
    *,
    # data (synthetic SMILES — dynamically rendered synthetic images)
    synthetic_smiles_list: Optional[List[str]] = None,
    synthetic_smiles_num: Optional[int] = None,
    # data (USPTO patent molecule images — real-world file-based)
    uspto_csv_path: Optional[str] = None,
    uspto_data_dir: Optional[str] = None,
    wild_smiles=False,
    # training 
    save_path: str,
    num_epochs: int,
    batch_size: int,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    early_stopping_patience: int,
    # Validation (list of benchmarks; first one is primary for early stopping)
    benchmarks: Optional[List[Dict[str, str]]] = None,
    val_max_samples: Optional[int] = None,
    # Legacy single-benchmark args (used if benchmarks is None)
    benchmark_dir: str = '',
    benchmark_csv_path: str = '',
    # Molecule
    max_atoms: int = 100,
    mol_augment: bool = True,
    # Vision encoder
    image_size: Tuple[int, int] = (384, 384),
    n_bins: int = 64,
    backbone: str = 'swin_b',
    pretrained: bool = True,
    # decoder
    d_model: int = 512,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    # ===== optimization parameters =====
    num_workers: int = 4,
    use_amp: bool = True,
    use_gradient_checkpointing: bool = False,
    # ===== MLE regularization (synthetic data) =====
    mle_weight: float = 0.0,
    mle_smiles_path: Optional[str] = None,
    mle_smiles_num: int = 50000,
    mle_batch_size: int = 8,
    # ===== resume from checkpoint =====
    resume_from: Optional[str] = None,
    # ===== fine-tune from pretrained weights (model only, fresh optimizer) =====
    finetune_from: Optional[str] = None,
    # ===== alternative ENCODER-ONLY init (MIM-pretrained Swin-B) =====
    encoder_weights: Optional[str] = None,
) -> List[float]:
    """Supervised MLE training with DDP, cosine-annealed LR, and AMP.

    Supports three data configurations:
      1. Synthetic only:  pass *synthetic_smiles_list* + *synthetic_smiles_num*
      2. USPTO only:      pass *uspto_csv_path* + *uspto_data_dir*
      3. Joint:           pass both synthetic and USPTO args

    Args:
        synthetic_smiles_list: Pool of SMILES for synthetic rendering.
        synthetic_smiles_num: Number of SMILES to sample from the pool.
        uspto_csv_path: Path to USPTO CSV with file_path/SMILES/node_coords/edges.
        uspto_data_dir: Root directory for resolving image paths in USPTO CSV.
        save_path: Directory for checkpoints and TensorBoard logs.
        benchmarks: List of dicts with keys 'name', 'dir', 'csv'. First is primary.
        batch_size: **Per-GPU** batch size (effective = batch_size × world_size).
    """
    # ===== DDP setup =====
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
        effective_batch = batch_size * world_size
        print(f'Per-GPU batch size: {batch_size}, Effective batch size: {effective_batch}')
        print(f'Optimizations: AMP={use_amp}, GradCheckpoint={use_gradient_checkpointing}')
        print(f'Optimizer: encoder_lr={encoder_lr}, decoder_lr={decoder_lr}, weight_decay={weight_decay}')
        print(f'cudnn.benchmark={torch.backends.cudnn.benchmark}, cudnn.deterministic={torch.backends.cudnn.deterministic}')

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed + local_rank)
    random.seed(seed + local_rank)

    # Normalize benchmark list
    if benchmarks is None:
        if benchmark_dir and benchmark_csv_path:
            benchmarks = [{'name': os.path.splitext(os.path.basename(benchmark_csv_path))[0],
                           'dir': benchmark_dir, 'csv': benchmark_csv_path}]
        else:
            benchmarks = []
    if is_main_process() and benchmarks:
        print(f'Validation benchmarks ({len(benchmarks)}):')
        for bm in benchmarks:
            print(f'  {bm["name"]}: {bm["csv"]}')
    
    # Create vocab
    vocab = ComoVocab(n_bins=n_bins)
    if is_main_process():
        print(f'Vocab size: {len(vocab)}')
    
    # Prepare training data
    has_synthetic = synthetic_smiles_list is not None
    has_uspto = uspto_csv_path is not None
    if not has_synthetic and not has_uspto:
        raise ValueError('Must provide synthetic_smiles_list and/or uspto_csv_path')

    datasets = []

    if has_uspto:
        if is_main_process():
            print(f'USPTO data: {uspto_csv_path}')
        uspto_dataset = USPTOMolDataset(
            vocab=vocab,
            csv_path=uspto_csv_path,
            data_dir=uspto_data_dir or '',
            image_size=image_size,
            img_augment=True,
            geo_augment=True,
        )
        datasets.append(('USPTO', uspto_dataset))
        if is_main_process():
            print(f'USPTO dataset size: {len(uspto_dataset)}')

    if has_synthetic:
        n = len(synthetic_smiles_list)
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        if synthetic_smiles_num is not None:
            indices = indices[:synthetic_smiles_num]
        sampled_smiles = [synthetic_smiles_list[i] for i in indices]
        if is_main_process():
            print(f'Synthetic data: {len(sampled_smiles)} SMILES')
        synthetic_dataset = MoleculeDataset(
            smiles_list=sampled_smiles,
            wild_smiles=wild_smiles,
            shuffle_smiles=True,
            vocab=vocab,
            image_size=image_size,
            mol_augment=mol_augment,
            geo_augment=True,
            img_augment=True,
        )
        datasets.append(('synthetic', synthetic_dataset))

    if len(datasets) > 1:
        train_dataset = ConcatDataset([d for _, d in datasets])
        if is_main_process():
            parts = ' + '.join(f'{name}: {len(d)}' for name, d in datasets)
            print(f'Joint training dataset: {len(train_dataset)} ({parts})')
    else:
        train_dataset = datasets[0][1]
        if is_main_process():
            print(f'Train size: {len(train_dataset)} ({datasets[0][0]} only)')

    bond_class_weights = get_bond_class_weights()
    
    # DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True,
        seed=seed
    ) if world_size > 1 else None
    
    dl_common = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'collate_fn': partial(collate_fn, vocab=vocab, max_atoms_limit=max_atoms),
        'shuffle': (train_sampler is None),
        'sampler': train_sampler,
        'drop_last': True,
    }
    if num_workers > 0:
        dl_common.update({
            'prefetch_factor': 4,
            'persistent_workers': False,
            'multiprocessing_context': 'forkserver',
            'timeout': 60,
        })
    
    def _make_loader():
        return DataLoader(train_dataset, **dl_common)
    
    train_loader = _make_loader()

    # MLE Regularization DataLoader
    mle_loader = None
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

    # Model
    model = ComoModel(
        vocab=vocab,
        backbone=backbone,
        pretrained=pretrained,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_gradient_checkpointing=use_gradient_checkpointing,
        encoder_weights=encoder_weights,
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=False)

    # Optimizer + Scheduler
    optimizer = training_utils.create_optimizer(model, encoder_lr, decoder_lr, weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    if is_main_process():
        print(f'Steps/epoch: {steps_per_epoch}')
        print(f'Total steps: {total_steps}, Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)')
    scheduler = training_utils.create_cosine_scheduler(optimizer, total_steps, warmup_ratio)

    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    # Resume / fine-tune from checkpoint
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
    elif finetune_from is not None and os.path.isfile(finetune_from):
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

    global_step = (start_epoch - 1) * steps_per_epoch

    # TensorBoard
    hparams = {
        'batch_size': batch_size,
        'effective_batch_size': batch_size * world_size,
        'encoder_lr': encoder_lr,
        'decoder_lr': decoder_lr,
        'weight_decay': weight_decay,
        'warmup_ratio': warmup_ratio,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'image_size': str(image_size),
        'n_bins': n_bins,
        'backbone': backbone,
        'pretrained': pretrained,
        'd_model': d_model,
        'nhead': nhead,
        'num_decoder_layers': num_decoder_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout,
        'max_atoms': max_atoms,
        'mol_augment': mol_augment,
        'use_amp': use_amp,
        'use_gradient_checkpointing': use_gradient_checkpointing,
        'seed': seed,
        'uspto_csv_path': uspto_csv_path,
        'uspto_data_dir': uspto_data_dir,
        'synthetic_smiles_num': synthetic_smiles_num,
        'train_samples': len(train_dataset),
        'resume_from': resume_from,
        'finetune_from': finetune_from,
        'mle_weight': mle_weight,
        'mle_smiles_path': mle_smiles_path,
        'mle_smiles_num': mle_smiles_num,
        'mle_batch_size': mle_batch_size,
    }
    writer, log_dir = training_utils.setup_tensorboard(save_path, resume_log_dir, hparams)
    log_dir = training_utils.broadcast_log_dir(log_dir, device, world_size)

    # MLE iterator
    mle_iter = None
    if mle_loader is not None:
        mle_iter = iter(mle_loader)
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if mle_loader is not None and hasattr(mle_loader, 'sampler') and hasattr(mle_loader.sampler, 'set_epoch'):
            mle_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_mle_loss = 0.0
        num_batches = 0
        
        steps_in_epoch = len(train_loader)
        train_bar = tqdm(total=steps_in_epoch,
                         desc=f'Epoch {epoch} [Train]',
                         disable=not is_main_process())
        train_iter = iter(train_loader)
        
        for _step in range(steps_in_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if is_main_process():
                    warnings.warn(f"DataLoader error at epoch {epoch} step {_step}: {e}")
                batch = None
                del train_iter
                train_loader = _make_loader()
                train_iter = iter(train_loader)
            
            batch_valid = batch is not None
            if world_size > 1:
                valid_tensor = torch.tensor([1 if batch_valid else 0],
                                           dtype=torch.int32, device=device)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                batch_valid = valid_tensor.item() == 1
            
            if not batch_valid:
                train_bar.update(1)
                continue
            
            images = batch['images'].to(device, non_blocking=True).contiguous()
            tgt_tokens = batch['tgt_tokens'].to(device, non_blocking=True).contiguous()
            tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True).contiguous()
            atom_indices = batch['atom_indices'].to(device, non_blocking=True).contiguous()
            atom_mask = batch['atom_mask'].to(device, non_blocking=True).contiguous()
            max_atoms_val = batch['max_atoms']
            bond_matrices_list = batch['bond_matrices_list']

            tgt_tokens, tgt_padding_mask, atom_indices, atom_mask = truncate_sequences(
                tgt_tokens, tgt_padding_mask, atom_indices, atom_mask, max_seq_len=600
            )
            
            tgt_input = tgt_tokens[:, :-1].contiguous()
            tgt_target = tgt_tokens[:, 1:].contiguous()
            tgt_input_mask = tgt_padding_mask[:, :-1].contiguous()
            
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    token_logits, edge_logits, _hidden = model(
                        images=images,
                        tgt_tokens=tgt_input,
                        tgt_key_padding_mask=tgt_input_mask,
                        atom_indices=atom_indices,
                        atom_mask=atom_mask,
                        max_atoms=max_atoms_val
                    )
                    del _hidden
                    
                    losses = compute_losses(
                        token_logits=token_logits,
                        edge_logits=edge_logits,
                        tgt_tokens=tgt_target,
                        bond_matrices_list=bond_matrices_list,
                        vocab=vocab,
                        bond_class_weights=bond_class_weights
                    )
                scaler.scale(losses['total_loss']).backward()
            else:
                token_logits, edge_logits, _hidden = model(
                    images=images,
                    tgt_tokens=tgt_input,
                    tgt_key_padding_mask=tgt_input_mask,
                    atom_indices=atom_indices,
                    atom_mask=atom_mask,
                    max_atoms=max_atoms_val
                )
                del _hidden
                
                losses = compute_losses(
                    token_logits=token_logits,
                    edge_logits=edge_logits,
                    tgt_tokens=tgt_target,
                    bond_matrices_list=bond_matrices_list,
                    vocab=vocab,
                    bond_class_weights=bond_class_weights
                )
                losses['total_loss'].backward()

            # MLE regularization forward + backward
            mle_loss_item = 0.0
            if mle_loader is not None:
                mle_loss_item, mle_iter = training_utils.mle_forward_backward(
                    model=model, mle_iter=mle_iter, mle_loader=mle_loader,
                    mle_weight=mle_weight, vocab=vocab,
                    bond_class_weights=bond_class_weights,
                    scaler=scaler, device=device, use_amp=use_amp,
                )

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            running_loss += losses['total_loss'].item()
            running_mle_loss += mle_loss_item
            num_batches += 1
            global_step += 1
            
            current_lrs = scheduler.get_last_lr()
            
            if is_main_process():
                postfix = {
                    'loss': f"{losses['total_loss'].item():.2f}",
                    't': f"{losses['token_loss'].item():.2f}",
                    'b': f"{losses['bond_loss'].item():.2f}",
                }
                if mle_loader is not None:
                    postfix['mle'] = f"{mle_loss_item:.2f}"
                train_bar.set_postfix(postfix)
                
                writer.add_scalar('Loss/train', losses['total_loss'].item(), global_step)
                writer.add_scalar('Loss/train_token', losses['token_loss'].item(), global_step)
                writer.add_scalar('Loss/train_bond', losses['bond_loss'].item(), global_step)
                writer.add_scalar('LR/encoder', current_lrs[0], global_step)
                writer.add_scalar('LR/decoder', current_lrs[1], global_step)
                if mle_loader is not None:
                    writer.add_scalar('MLE/loss', mle_loss_item, global_step)
            
            train_bar.update(1)
        
        train_bar.close()
        
        avg_loss = running_loss / max(num_batches, 1)
        avg_mle_loss = running_mle_loss / max(num_batches, 1)
        if is_main_process():
            msg = f'Epoch {epoch} [Train] - avg_loss: {avg_loss:.4f}'
            if mle_loader is not None:
                msg += f', avg_mle_loss: {avg_mle_loss:.4f}'
            print(msg)
        
        # Evaluation
        primary_val_metrics = training_utils.run_validation(
            model=model, benchmarks=benchmarks, device=device,
            epoch=epoch, writer=writer, global_step=global_step,
            val_max_samples=val_max_samples,
        )

        val_acc = primary_val_metrics['exact_match_acc'] if primary_val_metrics else 0.0
        should_stop, best_val_acc, epochs_no_improve = training_utils.check_early_stopping(
            val_acc, best_val_acc, epochs_no_improve,
            early_stopping_patience, model, save_path,
        )

        if is_main_process():
            training_utils.save_checkpoint(save_path, model, optimizer, scheduler, scaler,
                            epoch, best_val_acc, epochs_no_improve, log_dir)

        should_stop = training_utils.sync_early_stopping(should_stop, device, world_size)

        if should_stop:
            if is_main_process():
                print(f'Early stopping triggered at epoch {epoch}.')
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
    
    if is_main_process():
        training_utils.final_save(model, save_path, writer)
        print('Training complete!')

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
    
    cleanup_ddp()
    return []
