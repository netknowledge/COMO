"""
Path and naming utilities for COMO project artifacts.

Provides canonical artifact naming and filesystem path resolution
for checkpoints/ and results/ directories.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical artifact name mapping
# ---------------------------------------------------------------------------
# Short artifact keys used in code → canonical directory names on disk.

ARTIFACT_ALIASES: dict[str, str] = {
    "stage1_joint":                      "COMO_stage1_joint",
    "stage1_synthetic":                  "COMO_stage1_synthetic",
    "stage1_uspto":                      "COMO_stage1_uspto",
    "stage1_1M436K":                     "COMO_stage1_1M436K",
    "stage1_synthetic_then_436k_uspto":  "COMO_stage1_synthetic_then_436K_USPTO",
    "stage2_mrt":                        "COMO_stage2_mrt",
    "stage2_mrt_from_stage1_synthetic":  "COMO_stage2_mrt_from_stage1_synthetic",
    "stage2_mrt_from_stage1_1M436K":     "COMO_stage2_mrt_from_stage1_1M436K",
    "joint_mle_mrt":                     "COMO_joint",
    "joint_mle_mrt_in1kmim":             "COMO_joint_MIM_IN1K",
    "joint_mle_mrt_mixmim":              "COMO_joint_MIM_MIX",
    "joint_mle_mrt_mix010ink":           "COMO_joint_MIX_010_ink",
    "joint_mle_mrt_mix010uniform":       "COMO_joint_MIX_010_uniform",
    "ablation_continual_mle":            "COMO_ablation_continual_mle",
    "ablation_joint_mle":                "COMO_ablation_joint_mle",
    "mol_discriminator":                 "MolDiscriminator",
    "swinocsr_mrt":                      "SwinOCSR_MRT",
    "img2mol":                           "Img2Mol",
    "molgrapher":                        "MolGrapher",
    "osra":                              "OSRA",
}

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _project_base(project_dir: Path) -> Path:
    """Directory of the python package tree (``COMO_dev/`` internally, flat
    repository root in the public twin)."""
    dev = project_dir / "COMO_dev"
    return dev if dev.is_dir() else project_dir


def _artifact_roots(project_dir: Path) -> tuple[Path, Path]:
    """Resolve (checkpoints_root, results_root) for the active layout."""
    base = _project_base(project_dir)
    return base / "checkpoints", base / "results"


# ---------------------------------------------------------------------------
# Swin-B pretraining layout (Swin_pretraining/<variant>/<run_id>)
# ---------------------------------------------------------------------------

def resolve_swin_pretrain_dir(
    project_dir: Path,
    variant: str,
    run_id: str | None = None,
) -> Path:
    """Resolve a Swin-B pretraining run directory - VERBATIM concatenation.

    Layout: ``COMO_dev/checkpoints/Swin_pretraining/<variant>/<run_id>``.
    The variant name is used exactly as passed (that dir name on disk is the
    single source of truth - no alias tables). ``run_id`` names the EFFECTIVE
    batch (user convention, e.g. ``r224_bs256_e100``) and defaults to
    ``run_<timestamp>``; pass an explicit id to resume/append a run dir.
    """
    from datetime import datetime
    run = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return (_project_base(project_dir) / "checkpoints" / "Swin_pretraining"
            / variant / run)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonical_artifact_name(artifact_key: str) -> str:
    """Return the canonical directory name for *artifact_key*."""
    return ARTIFACT_ALIASES.get(artifact_key, artifact_key)


def ensure_package_root_on_syspath(project_dir: Path) -> Path:
    """Add the python package root to :data:`sys.path` (if not already
    present) and return it.

    Layout-adaptive: the internal repo nests the package in ``COMO_dev/``;
    the public twin keeps it at the repository root.
    """
    package_root = project_dir / "COMO_dev"
    if not package_root.is_dir():
        package_root = project_dir          # public twin: flat root layout
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    return package_root


def resolve_artifact_dir(
    project_dir: Path,
    artifact_key: str,
    *,
    category: str,
    reward_mode: str | None = None,
    visual_reward_encoder: str | None = None,
    experiment_name: str | None = None,
    for_write: bool = False,  # noqa: ARG001 – kept for backward compatibility
) -> Path:
    """Resolve the filesystem path for a *checkpoints* or *results* artifact.

    Parameters
    ----------
    project_dir:
        Repository root (the directory containing ``COMO_dev/`` internally;
        flat repository root in the public twin).
    artifact_key:
        Short key listed in :data:`ARTIFACT_ALIASES` (e.g. ``"stage1_joint"``).
    category:
        Either ``"checkpoints"`` or ``"results"``.
    reward_mode:
        Optional reward variant (``"visual"``, ``"tanimoto"``, …).
    visual_reward_encoder:
        Path to an external visual-encoder checkpoint (used when
        *reward_mode* is ``"visual"``).
    experiment_name:
        Optional experiment sub-directory name.
    """
    if category == "checkpoints":
        root = _artifact_roots(project_dir)[0]
    elif category == "results":
        root = _artifact_roots(project_dir)[1]
    else:
        raise ValueError(f"Unsupported artifact category: {category!r}")
    return root / canonical_artifact_name(artifact_key) / _variant_subpath(
        reward_mode,
        visual_reward_encoder=visual_reward_encoder,
        experiment_name=experiment_name,
    )


def resolve_checkpoint_path(
    project_dir: Path,
    artifact_key: str,
    checkpoint_name: str,
    *,
    reward_mode: str | None = None,
    visual_reward_encoder: str | None = None,
    experiment_name: str | None = None,
) -> Path:
    """Resolve the path to a ``.pth`` checkpoint under ``checkpoints/``."""
    return resolve_artifact_dir(
        project_dir,
        artifact_key,
        category="checkpoints",
        reward_mode=reward_mode,
        visual_reward_encoder=visual_reward_encoder,
        experiment_name=experiment_name,
    ) / f"{checkpoint_name}.pth"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _variant_subpath(
    reward_mode: str | None = None,
    *,
    visual_reward_encoder: str | None = None,
    experiment_name: str | None = None,
) -> Path:
    """Build the nested sub-directory for reward-mode / experiment variants."""
    subpath = Path()
    if reward_mode:
        if reward_mode == "visual":
            encoder_name = (
                Path(visual_reward_encoder).stem
                if visual_reward_encoder
                else "default_visual_encoder"
            )
            subpath = Path("visual") / encoder_name
        else:
            subpath = Path(reward_mode)
    if experiment_name:
        subpath = subpath / experiment_name
    return subpath