# COMO: Optical Chemical Structure Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/como-ocsr)](https://pypi.org/project/como-ocsr/)

**COMO** (Closed-loop Optical Molecule recOgnition) converts images of chemical structure diagrams into machine-readable SMILES strings, atom-level coordinates, and bond matrices.

Compared to image-to-text OCSR models (e.g., MolScribe, SwinOCSR, Image2Mol), COMO uniquely predicts explicit molecular graphs — atoms with 2D coordinates and bonds — then reconstructs SMILES using cheminformatics post-processing for provably valid, chemically accurate structures.

## 🚀 Quick Start

```python
import como

# 1. Load model
model = como.load_model("COMO_joint.pth", device="cuda")

# 2. Predict a single molecule
smiles = como.predict(model, "molecule.png")         # → "CC(=O)O"
result = como.predict(model, "molecule.png", smiles_mode=None)
# result contains: tokens, atom symbols, coordinates, bond matrix, etc.

# 3. Batch prediction
smiles_list = como.predict_batch(model, ["mol1.png", "mol2.png"])

# 4. Benchmark evaluation
metrics = como.evaluate(model, "benchmark/USPTO/", "benchmark/USPTO.csv")
print(metrics["exact_match_acc"], metrics["avg_tanimoto"])
```

## 📦 Installation

```bash
pip install como-ocsr
```

**Requirements:** Python 3.10+, PyTorch ≥ 2.0, RDKit.

## 🧠 Model Checkpoints

| Checkpoint | Description |
|---|---|
| `COMO_joint.pth` | Full model — MLE + MRT joint training (recommended) |
| `COMO_stage1_synthetic.pth` | Stage 1 only — MLE on synthetic data |

Download from [Hugging Face](https://huggingface.co/Keylab/COMO).

## 📖 API Reference

### `como.load_model(checkpoint_path, device="cuda", pretrained=True, **kwargs)`

Load a COMO model from a checkpoint.

- **checkpoint_path** (`str`): Path to `.pth` checkpoint file.
- **device** (`str`): `"cuda"` or `"cpu"`.
- **pretrained** (`bool`): Use ImageNet-pretrained backbone weights (default: `True`).
- Returns: `ComoModel` in evaluation mode.

### `como.predict(model, image, *, beam_size=1, max_len=500, smiles_mode="postprocess", device=None)`

Predict SMILES for a single image.

- **image**: File path (`str`), NumPy array (H×W×3 or H×W), PIL `Image`, or preprocessed `torch.Tensor`.
- **beam_size** (`int`): 1 = greedy, 3 = beam search.
- **smiles_mode** (`str`):
  - `"postprocess"` — cheminformatics-based SMILES reconstruction (recommended, best accuracy)
  - `"graph"` — graph-traversal SMILES
  - `"decoder"` — raw decoder output
  - `None` — returns full result dict (tokens, atoms, bonds, coordinates)
- Returns: SMILES string (`str`) or full result dict.

### `como.predict_batch(model, images, *, beam_size=1, max_len=500, smiles_mode="postprocess", device=None)`

Predict SMILES for multiple images (single GPU).

- **images**: List of file paths, NumPy arrays, PIL Images, or Tensors.
- Returns: List of SMILES strings or result dicts.

### `como.evaluate(model, benchmark_dir, csv_path, *, beam_size=1, postproc_workers=32, tautomer_standardize=True, gpus="0")`

Evaluate on a benchmark dataset.

- **benchmark_dir**: Directory of `.png` images.
- **csv_path**: CSV with columns `image_id` and `SMILES`.
- **gpus**: Comma-separated GPU IDs (e.g. `"0,1,2,3"`), or `None` for all GPUs.
- Returns: Dict with `exact_match_acc`, `avg_tanimoto`, `tautomer_match_acc`, etc.

### `como.evaluate_benchmarks(model, benchmarks, *, ...)`

Evaluate on multiple benchmarks at once.

- **benchmarks**: List of `{"name": ..., "benchmark_dir": ..., "csv_path": ...}` dicts.
- Returns: `dict[name] → metrics_dict`.

## 🧪 Supported Input Formats

- PNG / JPEG / TIFF images
- Hand-drawn or computer-generated chemical structure diagrams
- Arbitrary aspect ratios and sizes (auto-resized internally)

## 📄 License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Model Weights**: CC BY-NC 4.0

## 📚 Citation

If you use COMO in your research, please cite:

```bibtex
@article{lyu2025como,
  title={Closed-loop Optical Molecule recOgnition with Minimum Risk Training},
  author={Lyu, Zhuoqi and others},
  journal={arXiv},
  year={2025}
}
```
