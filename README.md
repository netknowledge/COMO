# COMO

**COMO** (**C**losed-loop **O**ptical **M**olecule rec**O**gnition) is a deep
learning framework that recognizes chemical structure diagrams from images and
predicts SMILES strings with atom-level coordinates and bond matrices.  It uses
Minimum Risk Training (MRT) to directly optimize molecular-level,
non-differentiable objectives.

## Installation

```bash
pip install como-ocsr
```

## Quick Start

```python
import como

# Load a model checkpoint (on GPU 0)
model = como.load_model("path/to/checkpoint.pth", device="cuda:0")

# Predict SMILES from a single image
smiles = como.predict(model, "molecule.png")
print(smiles)  # "CC(=O)O"

# Batch prediction on a specific GPU
smiles_list = como.predict_batch(model, ["mol1.png", "mol2.png"], device="cuda:1")

# Evaluate on a benchmark (single GPU by default)
metrics = como.evaluate(
    model,
    benchmark_dir="benchmark/USPTO/",
    csv_path="benchmark/USPTO.csv",
)
print(f"Exact Match: {metrics['postprocess/exact_match_acc']:.2%}")

# Multi-GPU, multi-benchmark evaluation
benchmarks = [
    {"name": "USPTO", "benchmark_dir": "benchmark/USPTO/",
     "csv_path": "benchmark/USPTO.csv"},
    {"name": "CLEF",  "benchmark_dir": "benchmark/CLEF/",
     "csv_path": "benchmark/CLEF_corrected.csv"},
]
results = como.evaluate_benchmarks(model, benchmarks, gpus="0,1,2,3")
for name, m in results.items():
    print(f"{name}: {m['postprocess/exact_match_acc']:.2%}")
```

## API Reference

### GPU Selection

All functions accept a ``device`` parameter for single-GPU usage:

```python
model = como.load_model("checkpoint.pth", device="cuda:0")
como.predict(model, "img.png", device="cuda:1")
como.predict_batch(model, [...], device="cuda:2")
```

For **evaluation** (which uses multi-GPU internally via ``mp.spawn``), use the
``gpus`` parameter:

| Function | GPU control |
|----------|-------------|
| ``load_model`` | ``device="cuda:0"`` |
| ``predict`` | ``device="cuda:0"`` |
| ``predict_batch`` | ``device="cuda:0"`` |
| ``evaluate`` | ``gpus="0"`` (default), ``gpus="0,1,2"``, ``gpus=None`` (all) |
| ``evaluate_benchmarks`` | ``gpus="0"`` (default), ``gpus="0,1,2"``, ``gpus=None`` (all) |

---

### `como.load_model(checkpoint_path, device="cuda", pretrained=True, **kwargs)`

Load a COMO model from a `.pth` checkpoint.  Returns a :class:`ComoModel`
instance in evaluation mode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | `str` | *required* | Path to `.pth` checkpoint |
| `device` | `str` | `"cuda"` | ``"cuda"``, ``"cuda:0"``, or ``"cpu"`` |
| `pretrained` | `bool` | `True` | Use ImageNet-pretrained backbone weights |

**Returns:** ``ComoModel``

---

### `como.predict(model, image, *, beam_size=1, max_len=500, smiles_mode="postprocess", device=None)`

Predict the SMILES string for a single molecular image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `image` | `str` / `np.ndarray` / `PIL.Image` / `torch.Tensor` | *required* | Input image (file path, array, PIL, or preprocessed tensor) |
| `beam_size` | `int` | `1` | Beam width (1 = greedy, 3 = beam search) |
| `max_len` | `int` | `500` | Maximum number of tokens to generate |
| `smiles_mode` | `str` or `None` | `"postprocess"` | ``"postprocess"`` (best quality), ``"graph"``, ``"decoder"``, or ``None`` (raw result dict) |
| `device` | `str` or `None` | `None` | Optional device override (e.g. ``"cuda:1"``) |

**Returns:**
- `str` — predicted SMILES string (if *smiles_mode* is not ``None``)
- `dict` — full result dict with keys ``tokens``, ``symbols``, ``coords``, ``bond_mat``, ``decode_smiles``, ``success`` (if ``smiles_mode=None``)

---

### `como.predict_batch(model, images, *, beam_size=1, max_len=500, smiles_mode="postprocess", device=None)`

Batch prediction on a single GPU.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `images` | `list` | *required* | List of file paths, NumPy arrays, PIL Images, or tensors |
| `beam_size` | `int` | `1` | Beam width (1 = greedy, recommended for batch) |
| `max_len` | `int` | `500` | Maximum tokens per image |
| `smiles_mode` | `str` or `None` | `"postprocess"` | SMILES reconstruction mode |
| `device` | `str` or `None` | `None` | Optional device override |

**Returns:**
- `list[str]` — predicted SMILES for each image (if *smiles_mode* is not ``None``)
- `list[dict]` — raw result dicts (if ``smiles_mode=None``)

---

### `como.evaluate(model, benchmark_dir, csv_path, *, beam_size=1, postproc_workers=32, tautomer_standardize=True, gpus="0")`

Evaluate on a single benchmark dataset.  Returns a flat dict of metrics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `benchmark_dir` | `str` | *required* | Directory containing `.png` images |
| `csv_path` | `str` | *required* | CSV with columns ``image_id``, ``SMILES`` |
| `beam_size` | `int` | `1` | Beam width for decoding |
| `postproc_workers` | `int` | `32` | Parallel workers for SMILES post-processing |
| `tautomer_standardize` | `bool` | `True` | Include tautomer-normalized exact match |
| `gpus` | `str` or `None` | `"0"` | GPU IDs (``"0,1"``) or ``None`` for all |

**Returns:** ``dict`` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `decoder/exact_match_acc` | `float` | Exact match accuracy (decoder mode) |
| `decoder/avg_tanimoto` | `float` | Average Tanimoto similarity (decoder) |
| `decoder/tautomer_match_acc` | `float` | Tautomer-normalized exact match (decoder, if `tautomer_standardize=True`) |
| `decoder/failed_predictions` | `int` | Number of failed predictions (decoder) |
| `decoder/valid` | `int` | Number of chemically valid predictions (decoder) |
| `decoder/total` | `int` | Total benchmark samples |
| `graph/exact_match_acc` | `float` | Exact match accuracy (graph mode) |
| `graph/avg_tanimoto` | `float` | Average Tanimoto similarity (graph) |
| `graph/tautomer_match_acc` | `float` | Tautomer-normalized exact match (graph, if `tautomer_standardize=True`) |
| `graph/failed_predictions` | `int` | Number of failed predictions (graph) |
| `graph/valid` | `int` | Number of chemically valid predictions (graph) |
| `graph/total` | `int` | Total benchmark samples |
| `postprocess/exact_match_acc` | `float` | Exact match accuracy (postprocess mode, **primary metric**) |
| `postprocess/avg_tanimoto` | `float` | Average Tanimoto similarity (postprocess) |
| `postprocess/tautomer_match_acc` | `float` | Tautomer-normalized exact match (postprocess, if `tautomer_standardize=True`) |
| `postprocess/failed_predictions` | `int` | Number of failed predictions (postprocess) |
| `postprocess/valid` | `int` | Number of chemically valid predictions (postprocess) |
| `postprocess/records_df` | `DataFrame` | Per-image results with columns ``image_id``, ``gt_smiles``, ``pred_smiles``, ``exact``, ``tautomer``, ``tanimoto`` |
| `postprocess/total` | `int` | Total benchmark samples |
| `total` | `int` | Total benchmark samples |

---

### `como.evaluate_benchmarks(model, benchmarks, *, beam_size=1, postproc_workers=32, tautomer_standardize=True, gpus="0")`

Evaluate on multiple benchmarks in one call.  Returns a nested dict keyed
by benchmark name.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `benchmarks` | `list[dict]` | *required* | Each dict has keys ``"name"``, ``"benchmark_dir"``, ``"csv_path"`` |
| `beam_size` | `int` | `1` | Beam width for decoding |
| `postproc_workers` | `int` | `32` | Parallel workers for SMILES post-processing |
| `tautomer_standardize` | `bool` | `True` | Include tautomer-normalized exact match |
| `gpus` | `str` or `None` | `"0"` | GPU IDs (``"0,1"``) or ``None`` for all |

**Returns:** ``dict[str, dict]`` — mapping from benchmark name to a metrics
dict with the same structure as :func:`evaluate`.  Example::

    {
      "USPTO": {
        "postprocess/exact_match_acc": 0.934,
        "postprocess/avg_tanimoto": 0.987,
        ...
      },
      "CLEF": {
        "postprocess/exact_match_acc": 0.948,
        ...
      },
    }

**Example:**

    benchmarks = [
        {"name": "USPTO", "benchmark_dir": "data/benchmark/real/USPTO",
         "csv_path": "data/benchmark/real/USPTO.csv"},
        {"name": "CLEF",  "benchmark_dir": "data/benchmark/real/CLEF",
         "csv_path": "data/benchmark/real/CLEF_corrected.csv"},
    ]
    results = como.evaluate_benchmarks(model, benchmarks, gpus="0,1")
    for name, metrics in results.items():
        acc = metrics["postprocess/exact_match_acc"]
        tan = metrics["postprocess/avg_tanimoto"]
        print(f"{name}: Exact={acc:.2%}, Tanimoto={tan:.4f}")

---

### `como.canonicalize_smiles(smiles, *, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True)`

Canonicalize a SMILES string using RDKit.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smiles` | `str` | *required* | Input SMILES string |
| `ignore_chiral` | `bool` | `False` | Strip tetrahedral chirality before canonicalization |
| `ignore_cistrans` | `bool` | `False` | Strip cis–trans markers (``/`` and ``\``) before canonicalization |
| `replace_rgroup` | `bool` | `True` | If ``True``, replace R-group tokens (``R``, ``R1``, ``X``, ``Ar``, …) with wildcard ``*`` |

**Returns:** ``tuple[str, bool]`` — ``(canonical_smiles, ok)`` where *ok* is
``True`` if the SMILES is chemically valid and canonicalization succeeded.

---

### `como.canonicalize_tautomer(smiles)`

Canonicalize a SMILES string via RDKit's TautomerEnumerator, normalizing
different tautomeric forms (e.g., keto/enol, lactam/lactim) to the same
canonical representation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smiles` | `str` | *required* | Input SMILES string |

**Returns:** ``tuple[str, bool]`` — ``(tautomer_canonical_smiles, ok)`` where
*ok* is ``False`` if the input SMILES is invalid or tautomer enumeration fails.

---

### `como._result_to_smiles(result, mode="postprocess")`

Low-level: convert a raw prediction result dict (from :func:`predict` with
``smiles_mode=None``) to a canonical SMILES string.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `dict` | *required* | Raw prediction dict with keys ``smiles``, ``symbols``, ``coords``, ``bond_mat``, ``success`` |
| `mode` | `str` | ``"postprocess"`` | SMILES reconstruction mode |

*mode* options:

| Mode | Source | Chirality | Description |
|------|--------|-----------|-------------|
| ``"decoder"`` | Decoder token sequence | ✗ | Raw decoder SMILES, no graph info used. Fastest but lowest quality. |
| ``"graph"`` | Predicted atoms + bonds | ✓ | Reconstructs SMILES entirely from predicted atom symbols, coordinates, and bond matrix. Chirality restored via `_verify_chirality`. |
| ``"postprocess"`` | Decoder + atoms + bonds | ✓ | Starts from decoder SMILES, replaces R-groups/abbreviations, restores chirality from predicted coordinates and bond matrix, then expands functional groups back. Best quality. |

**Returns:** ``str`` or ``None`` — canonical SMILES string, or ``None`` if conversion fails.

## Model Weights

Pre-trained model weights are available on HuggingFace:

| Checkpoint | Reward Mode | Description |
|-----------|-------------|-------------|
| `COMO_joint/tanimoto/final.pth` | Tanimoto | Joint MLE+MRT (Tanimoto reward) |
| `COMO_joint/edit_distance/final.pth` | Edit Distance | Joint MLE+MRT (Edit Distance reward) |
| `COMO_joint/visual/final.pth` | Visual | Joint MLE+MRT (Visual reward) |

Download from: **https://huggingface.co/Keylab/COMO**

## Benchmark Datasets

Benchmark datasets (images + CSV ground truth) are available on HuggingFace Datasets:

| Dataset | Images | Type |
|---------|--------|------|
| USPTO | ~6K | Real patent images |
| USPTO-10K | ~10K | Real patent images |
| CLEF | ~5K | Real patent images |
| JPO | ~3K | Real patent images |
| UOB | ~4K | Real academic images |
| staker | ~1K | Real images |
| acs | ~2K | Real publication images |
| WildMol-10K | ~10K | Real wild images |
| indigo | ~8K | Synthetic (Indigo-rendered) |
| chemdraw | ~8K | Synthetic (ChemDraw style) |

Download from: **https://huggingface.co/Keylab/COMO** (see `benchmarks/` folder)

## Citation

If you use COMO in your research, please cite:

```bibtex
@article{lyu2026closed,
  title={COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training},
  author={Lyu, Zhuoqi and Ke, Qing},
  journal={arXiv preprint arXiv:2604.23546},
  year={2026}
}
```

## License

- **Code** (`como/` package): MIT License
- **Model Weights** (`.pth` files): CC BY-NC 4.0 (non-commercial use only)
- **Benchmark Datasets**: collected from existing public OCSR benchmarks; please refer to their
  original sources for license and attribution:

| Dataset | Source |
|---------|--------|
| USPTO, CLEF, JPO, UOB, Staker | [Rajan et al., 2020](https://github.com/Kohulan/OCSR_Review), [Xiong et al., 2023](https://github.com/jiachengxiong/alpha-Extractor) |
| Indigo, ChemDraw, ACS, Staker | [Qian et al., 2023](https://github.com/thomas0809/MolScribe) |
| USPTO-10K | [Morin et al., 2023](https://huggingface.co/datasets/docling-project/USPTO-30K) |
| WildMol-10K | [Fang et al., 2025](https://github.com/orgs/Chem-Struct-ML/repositories) |

See [LICENSE](LICENSE) for full terms.
