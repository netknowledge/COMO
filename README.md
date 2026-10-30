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

## Training

The complete training pipeline is released in this repository:
supervised MLE training, Minimum Risk Training (MRT), and the interleaved
MLE + MRT joint recipe used for the released checkpoints.

```bash
git clone https://github.com/netknowledge/COMO
cd COMO
pip install -e ".[training]"

# paper recipe: joint MLE + MRT with the visual reward (4 GPUs; single-GPU works too)
# (visual judge = frozen MolDiscriminator, downloaded by prepare_mol_discriminator.py)
python scripts/data/prepare_pubchem_smiles.py
python scripts/data/prepare_uspto_mol.py
python scripts/data/prepare_molparser_mrt.py
python scripts/data/prepare_mol_discriminator.py
python scripts/data/prepare_benchmarks.py

torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt \
    --mrt_every_n_steps 4 \
    --reward_mode visual \
    --visual_reward_encoder data/mol_discriminator/20260321_final.pth
```

Released recipes and full documentation (data licenses, the three reward
modes, the two-stage ablations, custom-dataset MRT finetuning, and the exact
hyperparameters of the released checkpoints):

- **[TRAINING.md](https://github.com/netknowledge/COMO/blob/main/TRAINING.md)** — training guide
- Training data: [Keylab/COMO `training_data/`](https://huggingface.co/Keylab/COMO/tree/main/training_data)
  (PubChem 1M subset + MolParser-SFT images; full dataset provenance and
  download scripts in the guide)

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

# Batch prediction with confidence (see "Prediction confidence" below)
preds = como.predict_batch(model, ["mol1.png", "mol2.png"], return_confidence=True)

# Evaluate on a benchmark (single GPU by default) — file-based
metrics = como.evaluate(
    model,
    benchmark_dir="benchmark/USPTO/",
    csv_path="benchmark/USPTO.csv",
)
print(f"Exact Match: {metrics['postprocess/exact_match_acc']:.2%}")

# Evaluate directly from HuggingFace (no local files needed)
metrics = como.evaluate(
    model,
    hf_dataset="Keylab/OCSR-Benchmarks",
    hf_config="USPTO",
)

# Multi-GPU, multi-benchmark evaluation (mix file-based and HF)
benchmarks = [
    {"name": "USPTO", "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "USPTO"},
    {"name": "CLEF",  "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "CLEF"},
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

### `como.load_model(checkpoint_path, device="cuda", **kwargs)`

Load a COMO model from a `.pth` checkpoint.  Returns a :class:`ComoModel`
instance in evaluation mode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | `str` | *required* | Path to `.pth` checkpoint |
| `device` | `str` | `"cuda"` | ``"cuda"``, ``"cuda:0"``, or ``"cpu"`` |

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
| `return_confidence` | `bool` | `False` | Also compute a scalar confidence in [0, 1] (see "Prediction confidence" below) |

**Returns:**
- `str` — predicted SMILES string (if *smiles_mode* is not ``None``); with *return_confidence* `True`, a `(smiles, confidence)` tuple
- `dict` — full result dict with keys ``tokens``, ``symbols``, ``coords``, ``bond_mat``, ``decode_smiles``, ``success`` plus scoring metrics (if ``smiles_mode=None``; ``confidence`` key also added when *return_confidence* is `True`)

---

### `como.predict_batch(model, images, *, beam_size=1, max_len=500, smiles_mode="postprocess", device=None, return_confidence=False)`

Batch prediction on a single GPU.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `images` | `list` | *required* | List of file paths, NumPy arrays, PIL Images, or tensors |
| `beam_size` | `int` | `1` | Beam width (1 = greedy, recommended for batch) |
| `max_len` | `int` | `500` | Maximum tokens per image |
| `smiles_mode` | `str` or `None` | `"postprocess"` | SMILES reconstruction mode |
| `device` | `str` or `None` | `None` | Optional device override |
| `return_confidence` | `bool` | `False` | Also compute a scalar confidence in [0, 1] per image (see "Prediction confidence" below) |
| `internal_batch_size` | `int` | `0` | If > 0, decode in sub-batches of this size (one flat result list, input order) |
| `prefetch` | `int` | `0` | If > 0 (with `internal_batch_size` > 0), overlap CPU image loading/transform with GPU decode via loader threads (~1.5x on long path-based lists) |
| `preprocess_workers` | `int` | `12` | Loader threads per GPU for the prefetch pool |

**Returns:**
- `list[str]` — predicted SMILES for each image (if *smiles_mode* is not ``None``); with *return_confidence* `True`, a list of `(smiles, confidence)` tuples
- `list[dict]` — raw result dicts (if ``smiles_mode=None``; ``confidence`` key also added when *return_confidence* is `True`)

#### Prediction confidence

Every prediction already carries several decoder-side confidence signals:

| Signal | Meaning | Direction |
|--------|---------|-----------|
| `bond_conf` | mean softmax probability of the predicted bond types; bounded [0, 1] | higher = more confident |
| `confidence` | exp(mean per-token log-prob of the chosen decode); bounded (0, 1] but ⚠ length-biased (penalizes long molecules) | higher = more confident |
| `mean_margin` | mean top-1 minus top-2 token log-prob gap (in nats) | higher = more confident; unbounded, squashed via `m/(1+m)` in the ensemble |
| `mean_entropy` | mean per-position softmax entropy, bounded by `ln V` | lower = more confident |
| `ensemble` | parameter-free bounded fusion: mean of `bond_conf`, `m/(1+m)`, and `1 - H/ln(V)` | higher = more confident, in [0, 1] |

With ``return_confidence=True``, a single scalar in [0, 1] is attached to
each result as ``'confidence'``.  The signal used is selected by
``CONFIDENCE_SIGNAL`` in ``como/model.py`` (currently ``ensemble``, the
bounded fusion of the decoder-side signals above).

```python
preds = como.predict_batch(model, ["mol1.png", "mol2.png"], return_confidence=True)
# -> [("CC(=O)O", 0.97), ("c1ccccc1", 0.99)]
```

---

### `como.evaluate(model, benchmark_dir=None, csv_path=None, *, hf_dataset=None, hf_config=None, hf_split="test", beam_size=1, postproc_workers=32, tautomer_standardize=True, gpus="0")`

Evaluate on a single benchmark dataset.  Returns a flat dict of metrics.

Two mutually exclusive (Hugging Face Priority) input modes are supported:

```python
# File-based
metrics = como.evaluate(model, "benchmark/USPTO/", "benchmark/USPTO.csv")

# HuggingFace dataset (no local files required)
metrics = como.evaluate(model, hf_dataset="Keylab/OCSR-Benchmarks", hf_config="USPTO")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ComoModel` | *required* | A loaded model |
| `benchmark_dir` | `str` or `None` | `None` | Directory containing `.png` images (file-based mode); Ignored if `hf_dataset` is provided |
| `csv_path` | `str` or `None` | `None` | CSV with columns ``image_id``, ``SMILES`` (file-based mode); Ignored if `hf_dataset` is provided |
| `hf_dataset` | `str` or `None` | `None` | HuggingFace dataset repo id, e.g. ``"Keylab/OCSR-Benchmarks"`` |
| `hf_config` | `str` or `None` | `None` | Config / subset name within the HF dataset, e.g. ``"USPTO"`` |
| `hf_split` | `str` | `"test"` | Dataset split to load |
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
| `benchmarks` | `list[dict]` | *required* | List of benchmark spec dicts (see below) |
| `beam_size` | `int` | `1` | Beam width for decoding |
| `postproc_workers` | `int` | `32` | Parallel workers for SMILES post-processing |
| `tautomer_standardize` | `bool` | `True` | Include tautomer-normalized exact match |
| `gpus` | `str` or `None` | `"0"` | GPU IDs (``"0,1"``) or ``None`` for all |
| `prefetch` | `int` | `4` | Preprocessed sub-batches kept in flight per GPU (overlaps image loading with decode) |
| `preprocess_workers` | `int` | `12` | Loader threads per GPU |

Each dict in *benchmarks* must contain ``"name"`` plus one of:

| Mode | Required keys | Optional keys |
|------|--------------|---------------|
| File-based | ``"benchmark_dir"``, ``"csv_path"`` | — |
| HuggingFace | ``"hf_dataset"`` | ``"hf_config"`` (default: benchmark name), ``"hf_split"`` (default: ``"test"``) |

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

**Examples:**

```python
# File-based
benchmarks = [
    {"name": "USPTO", "benchmark_dir": "data/benchmark/real/USPTO",
     "csv_path": "data/benchmark/real/USPTO.csv"},
    {"name": "CLEF",  "benchmark_dir": "data/benchmark/real/CLEF",
     "csv_path": "data/benchmark/real/CLEF_corrected.csv"},
]

# HuggingFace dataset (recommended — no local files required)
benchmarks = [
    {"name": "USPTO", "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "USPTO"},
    {"name": "CLEF",  "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "CLEF"},
    {"name": "JPO",   "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "JPO"},
]

results = como.evaluate_benchmarks(model, benchmarks, gpus="0,1")
for name, metrics in results.items():
    acc = metrics["postprocess/exact_match_acc"]
    tan = metrics["postprocess/avg_tanimoto"]
    print(f"{name}: Exact={acc:.2%}, Tanimoto={tan:.4f}")
```

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
| `result` | `dict` | *required* | Raw prediction dict with keys ``decode_smiles``, ``symbols``, ``coords``, ``bond_mat``, ``success`` |
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

Benchmark datasets are published as a HuggingFace Dataset with one config per benchmark:
**[Keylab/OCSR-Benchmarks](https://huggingface.co/datasets/Keylab/OCSR-Benchmarks)**

| Config | Size | Type |
|--------|-----:|------|
| `CLEF` | 992 | Real (patents) |
| `JPO` | 449 | Real (patents) |
| `UOB` | 5,740 | Real (academic) |
| `USPTO` | 5,719 | Real (patents) |
| `USPTO-10K` | 9,999 | Real (patents) |
| `Staker` | 50,000 | Real |
| `ACS` | 331 | Real (publications) |
| `WildMol-10K` | 9,889 | Real (wild) |
| `Indigo` | 5,719 | Synthetic |
| `ChemDraw` | 5,719 | Synthetic |

Each sample has three fields: `image_id` (str), `image` (PIL), `SMILES` (str).

```python
from datasets import load_dataset

# Load a single benchmark
ds = load_dataset("Keylab/OCSR-Benchmarks", name="USPTO", split="test")
sample = ds[0]
sample["image"].show()   # PIL Image
print(sample["SMILES"])  # ground-truth SMILES

# Iterate over all benchmarks
configs = ["CLEF", "JPO", "UOB", "USPTO", "USPTO-10K",
           "Staker", "ACS", "WildMol-10K", "Indigo", "ChemDraw"]
for name in configs:
    ds = load_dataset("Keylab/OCSR-Benchmarks", name=name, split="test")
    print(f"{name}: {len(ds)} samples")
```

Pre-packaged `.tar.gz` archives are also available for bulk download in the
[COMO model repository](https://huggingface.co/Keylab/COMO/tree/main/benchmarks).

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

- **Code**: MIT License (includes the `training/` pipeline)
- **Model Weights** (`.pth` files): CC BY-NC 4.0 (non-commercial use only)
- **Training Data**: mixed sources — see
  [TRAINING.md](https://github.com/netknowledge/COMO/blob/main/TRAINING.md) for the
  full table:
  - PubChem subset: [PubChem](https://pubchem.ncbi.nlm.nih.gov/) SMILES (public-domain; see PubChem terms)
  - USPTO Mol: MolScribe (`yujieq/MolScribe`); USPTO data is public domain
  - MolParser-SFT real images: [UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M) — **non-commercial use only**
- **Benchmark Datasets**: collected from existing public OCSR benchmarks; please refer to their
  original sources for license and attribution:

| Dataset | Source |
|---------|--------|
| USPTO, CLEF, JPO, UOB, Staker | [Rajan et al., 2020](https://github.com/Kohulan/OCSR_Review), [Xiong et al., 2023](https://github.com/jiachengxiong/alpha-Extractor) |
| Indigo, ChemDraw, ACS, Staker | [Qian et al., 2023](https://github.com/thomas0809/MolScribe) |
| USPTO-10K | [Morin et al., 2023](https://huggingface.co/datasets/docling-project/USPTO-30K) |
| WildMol-10K | [Fang et al., 2025](https://github.com/orgs/Chem-Struct-ML/repositories) |

See [LICENSE](LICENSE) for full terms.