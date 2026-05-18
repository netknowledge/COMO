"""
COMO-OCSR — Optical Chemical Structure Recognition with COMO.

COMO (Closed-loop Optical Molecule recOgnition) is a deep learning framework
that recognizes chemical structure diagrams from images and predicts SMILES
strings with atom-level coordinates and bond matrices.  It uses Minimum Risk
Training (MRT) to directly optimize molecular-level, non-differentiable
objectives, closing the gap between token-level training and molecular-level
evaluation.

Quick start::

    import como

    model = como.load_model("path/to/checkpoint.pth", device="cuda")
    smiles = como.predict(model, "molecule.png")
    print(smiles)  # "CC(=O)O"

    # Batch prediction
    results = como.predict_batch(model, ["mol1.png", "mol2.png"])

    # Benchmark evaluation
    metrics = como.evaluate(model, "benchmark/USPTO/", "benchmark/USPTO.csv")

    # Multi-GPU evaluation (use GPUs 0, 1, 2)
    metrics = como.evaluate(model, ..., gpus="0,1,2")
"""

from ._core import ComoModel, ComoVocab, evaluate_benchmarks as _raw_evaluate_benchmarks, _result_to_smiles
from ._chemistry import canonicalize_smiles, canonicalize_tautomer

__version__ = "1.1.1"

__all__ = [
    # Core classes
    "ComoVocab",
    "ComoModel",
    # Convenience functions
    "load_model",
    "predict",
    "predict_batch",
    "evaluate",
    "evaluate_benchmarks",
    # SMILES utilities
    "canonicalize_smiles",
    "canonicalize_tautomer",
    "_result_to_smiles",
    # Version
    "__version__",
]


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    pretrained: bool = True,
    **model_kwargs,
) -> ComoModel:
    """Load a COMO model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` checkpoint (e.g. ``"COMO_joint/tanimoto/final.pth"``).
    device:
        Device string (``"cuda"`` or ``"cpu"``).
    pretrained:
        Whether to use ImageNet-pretrained backbone weights (default: ``True``).
    model_kwargs:
        Additional arguments passed to :class:`ComoModel` (e.g. ``d_model``,
        ``nhead``, ``num_decoder_layers``).

    Returns
    -------
    ComoModel
        The loaded model in evaluation mode.
    """
    vocab = ComoVocab(n_bins=64)
    model = ComoModel(
        vocab=vocab,
        backbone="swin_b",
        pretrained=pretrained,
        **model_kwargs,
    )
    model.load_model(checkpoint_path, device=torch.device(device))
    return model


def predict(
    model: ComoModel,
    image,
    *,
    beam_size: int = 1,
    max_len: int = 500,
    smiles_mode: str = "postprocess",
    device: str | None = None,
) -> str | dict:
    """Predict the SMILES string for a single molecular image.

    Parameters
    ----------
    model:
        A loaded :class:`ComoModel`.
    image:
        One of: a file path (``str``), a NumPy array (H×W×3 or H×W), a PIL
        ``Image``, or a preprocessed ``torch.Tensor``.
    beam_size:
        Beam width (1 = greedy, 3 = beam search).  Greedy is faster and
        produces identical results to beam search on most images.
    max_len:
        Maximum number of tokens to generate.
    smiles_mode:
        SMILES reconstruction mode.  One of ``"postprocess"`` (best quality,
        recommended), ``"graph"``, ``"decoder"``, or ``None`` (return raw
        tokens dict instead of a SMILES string).
    device:
        Optional device override.

    Returns
    -------
    str or dict
        If *smiles_mode* is not ``None``, returns the predicted SMILES string.
        If ``None``, returns the full result dict containing tokens, atom
        symbols, coordinates, bond matrix, etc.
    """
    if device is not None:
        model.to(device)
    result = model.predict(
        image,
        beam_size=beam_size,
        max_len=max_len,
        smiles_mode=smiles_mode,
    )
    if smiles_mode is not None:
        return result["pred_smiles"]
    return result


def predict_batch(
    model: ComoModel,
    images: list,
    *,
    beam_size: int = 1,
    max_len: int = 500,
    smiles_mode: str = "postprocess",
    device: str | None = None,
) -> list[str] | list[dict]:
    """Predict SMILES for a batch of images (single GPU).

    Parameters
    ----------
    model:
        A loaded :class:`ComoModel`.
    images:
        List of file paths, NumPy arrays, PIL Images, or torch Tensors.
    beam_size:
        Beam width (1 = greedy, recommended for batch).
    max_len:
        Maximum number of tokens per image.
    smiles_mode:
        SMILES reconstruction mode (``"postprocess"`` recommended).
    device:
        Optional device override.

    Returns
    -------
    list[str] or list[dict]
        Predicted SMILES strings (or raw result dicts if *smiles_mode* is
        ``None``).
    """
    if device is not None:
        model.to(device)
    results = model.predict_batch(
        images,
        beam_size=beam_size,
        max_len=max_len,
        smiles_mode=smiles_mode,
    )
    if smiles_mode is not None:
        return [r["pred_smiles"] for r in results]
    return results


def evaluate(
    model: ComoModel,
    benchmark_dir: str,
    csv_path: str,
    *,
    beam_size: int = 1,
    postproc_workers: int = 32,
    tautomer_standardize: bool = True,
    gpus: str | None = "0",
) -> dict:
    """Evaluate model on a single benchmark.

    Parameters
    ----------
    model:
        A loaded :class:`ComoModel`.
    benchmark_dir:
        Directory containing benchmark images (``.png`` files).
    csv_path:
        Path to CSV with columns ``image_id`` and ``SMILES``.
    beam_size:
        Beam width for decoding (1 = greedy).
    postproc_workers:
        Number of parallel workers for SMILES post-processing.
    tautomer_standardize:
        If ``True``, additionally compute tautomer-normalized exact match.
    gpus:
        Comma-separated GPU IDs (e.g. ``"0"`` or ``"0,1,2,3"``).  Set to
        ``None`` to use all available GPUs.  Default: ``"0"`` (single GPU).

    Returns
    -------
    dict
        Metrics including ``exact_match_acc``, ``avg_tanimoto``, and
        (optionally) ``tautomer_match_acc`` for each SMILES mode.
    """
    import os as _os
    _prev = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
    try:
        if gpus is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        benchmarks = [{"name": "benchmark", "benchmark_dir": benchmark_dir, "csv_path": csv_path}]
        results = _raw_evaluate_benchmarks(
            model,
            benchmarks,
            beam_size=beam_size,
            postproc_workers=postproc_workers,
            tautomer_standardize=tautomer_standardize,
        )
        return results["benchmark"]
    finally:
        if _prev is None:
            _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            _os.environ["CUDA_VISIBLE_DEVICES"] = _prev


def evaluate_benchmarks(
    model: ComoModel,
    benchmarks: list[dict],
    *,
    beam_size: int = 1,
    postproc_workers: int = 32,
    tautomer_standardize: bool = True,
    gpus: str | None = "0",
) -> dict[str, dict]:
    """Evaluate model on multiple benchmarks.

    Parameters
    ----------
    model:
        A loaded :class:`ComoModel`.
    benchmarks:
        List of dicts, each with keys ``"name"``, ``"benchmark_dir"``, and
        ``"csv_path"``.  Example::

            [{"name": "USPTO", "benchmark_dir": "...", "csv_path": "..."}]
    beam_size:
        Beam width for decoding (1 = greedy).
    postproc_workers:
        Number of parallel workers for SMILES post-processing.
    tautomer_standardize:
        If ``True``, additionally compute tautomer-normalized exact match.
    gpus:
        Comma-separated GPU IDs (e.g. ``"0"`` or ``"0,1,2,3"``).  Set to
        ``None`` to use all available GPUs.  Default: ``"0"`` (single GPU).

    Returns
    -------
    dict[str, dict]
        Mapping from benchmark name to metrics dict.
    """
    import os as _os
    _prev = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
    try:
        if gpus is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        return _raw_evaluate_benchmarks(
            model,
            benchmarks,
            beam_size=beam_size,
            postproc_workers=postproc_workers,
            tautomer_standardize=tautomer_standardize,
        )
    finally:
        if _prev is None:
            _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            _os.environ["CUDA_VISIBLE_DEVICES"] = _prev


# ---------------------------------------------------------------------------
# Internal: torch import for load_model
# ---------------------------------------------------------------------------
import torch  # noqa: E402
