"""
COMO DDP Utilities
==================

Distributed Data Parallel helpers used across training and evaluation.

Public API:
  - ``setup_ddp``       — initialize NCCL process group with tuned timeouts
  - ``cleanup_ddp``     — best-effort teardown of process group
  - ``is_main_process`` — True for rank 0
  - ``get_rank``        — current process rank
  - ``get_world_size``  — total number of processes
"""

import os
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group with tuned timeouts.

    Timeout budget (defence-in-depth):
      Layer 1: func_set_timeout(5s) in drawing_engine  – catches most hangs
      Layer 2: DataLoader timeout=60s                  – kills hung C-extension workers
      Layer 3: NCCL collective timeout=30 min          – generous for validation inference variance
      Layer 4: NCCL heartbeat env var=1800s            – prevents watchdog kills during recovery
    """
    # Prevent NCCL watchdog from killing processes while DataLoader is recovering
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Best-effort destroy of distributed process group.

    In rare cases NCCL teardown can raise (e.g., transient CUDA OOM during
    communicator shutdown). Since training/evaluation has already finished at
    this point, avoid turning cleanup-time backend issues into hard failures.
    """
    if not dist.is_initialized():
        return

    try:
        dist.destroy_process_group()
    except Exception as e:
        warnings.warn(f"cleanup_ddp() ignored backend shutdown error: {e}")


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1
