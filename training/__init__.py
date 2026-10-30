"""
COMO Training
=============

Training loops for COMO molecule-image recognition.

Contains:
  - ``ddp_utils``    — DDP initialisation and process helpers
  - ``losses``       — MLE loss computation (token + bond)
  - ``reward``       — MRT loss and visual reward encoding
  - ``datasets``     — MLE/MRT dataset loaders (synthetic render, USPTO, real images)
  - ``utils``        — checkpoint/optimizer/TensorBoard shared infrastructure
  - ``train_mle``    — supervised MLE training on synthetic/USPTO data
  - ``train_mrt``    — RL (MRT) finetuning on real images
  - ``train_joint``  — interleaved MLE + MRT joint training

Private package — not part of the pip-publishable ``como`` wheel; released as
repository source code (import ``training`` from the repository root).
Import training modules directly::

    from training.train_mle import train_mle
    from training.train_mrt import train_mrt
    from training.train_joint import train_joint
"""

from .ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_rank,
    get_world_size,
)
from .losses import (
    compute_losses,
    get_bond_class_weights,
)
# Reward/MRT and training loops: imported lazily to avoid circular deps with datasets
# Use: from training.reward import compute_mrt_loss
#      from training.train_mle import train_mle

__all__ = [
    # DDP
    'setup_ddp',
    'cleanup_ddp',
    'is_main_process',
    'get_rank',
    'get_world_size',
    # Losses
    'compute_losses',
    'get_bond_class_weights',
]
