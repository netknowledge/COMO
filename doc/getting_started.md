# Getting Started

End-to-end walkthrough: install COMO, download a checkpoint, run your first
predictions, and evaluate on a benchmark. The full function-level details live
in the [API reference](api_reference.md); training the model is covered in the
[training guide](training.md).

## 1. Install

**Inference only** (PyPI, all platforms):

```bash
pip install como-ocsr
```

**With training** (repository clone, Linux; the synthetic-data renderer uses a
vendored Indigo build):

```bash
git clone https://github.com/netknowledge/COMO
cd COMO
pip install -e ".[training]"
```

Requirements: Python ≥ 3.10 with `torch ≥ 2.0` (GPU strongly recommended for
anything beyond a handful of images).

## 2. Load a checkpoint

### 2.1 Via Hugging Face
Simply:
```python
from como.model import ComoModel
model = ComoModel.from_pretrained("Keylab/COMO-Tanimoto", device="cuda")
```
### 2.2 Via weight files

Released checkpoints (HuggingFace [Keylab/COMO](https://huggingface.co/Keylab/COMO)):

| Checkpoint | Reward Mode | Recipe |
|-----------|-------------|--------|
| `models/tanimoto/final.pth` | Tanimoto | Joint MLE+MRT |
| `models/edit_distance/final.pth` | Edit Distance | Joint MLE+MRT |
| `models/visual/final.pth` | Visual | Joint MLE+MRT (frozen MolDiscriminator judge) |

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("Keylab/COMO", "models/edit_distance/final.pth",
                       local_dir="checkpoints")
```

or with the CLI: `hf download Keylab/COMO models/edit_distance/final.pth --local-dir checkpoints`
then
```python
model = como.load_model(ckpt, device="cuda:0")   # "cpu" also works
```

## 3. First prediction

```python
import como

smiles, conf = como.predict(model, "examples/sample_images/Metformin.png",
                            return_confidence=True)
print(smiles, f"({conf:.2f})")
# -> CN(C)C(=N)N=C(N)N (0.99)
```

Images can be passed as file paths, NumPy arrays, PIL Images, or preprocessed
tensors. Batch decoding (single GPU):

```python
preds = como.predict_batch(
    model,
    ["examples/sample_images/Metformin.png",
     "examples/sample_images/Sertraline.png",
     "examples/sample_images/Atorvastatin.png"],
    internal_batch_size=16,
)
```

A runnable, self-contained version of all this lives in
[`examples/`](../examples) — two small scripts and a notebook that also
render results and compare against ground truth.

## 4. Evaluate on a benchmark

Evaluate on any config of
[Keylab/OCSR-Benchmarks](https://huggingface.co/datasets/Keylab/OCSR-Benchmarks)
without downloading anything manually:

```python
import como

metrics = como.evaluate(model, hf_dataset="Keylab/OCSR-Benchmarks",
                        hf_config="USPTO")
print(f"Exact match: {metrics['postprocess/exact_match_acc']:.2%}")
```

Multi-GPU, multi-benchmark in one call (each spec is either file-based with
`benchmark_dir`/`csv_path`, or HuggingFace with `hf_dataset`/`hf_config`):

```python
benchmarks = [
    {"name": "USPTO", "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "USPTO"},
    {"name": "CLEF",  "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "CLEF"},
    {"name": "JPO",   "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "JPO"},
]
results = como.evaluate_benchmarks(model, benchmarks, gpus="0,1,2,3")
```

The primary metric is `postprocess/exact_match_acc` (decoder + graph
post-processing, chirality-aware; `tautomer_match_acc` additionally
normalizes tautomers). See the [API reference](api_reference.md) for the full
metric dict.

## 5. Troubleshooting

- **`CUDA out of memory`** — lower `internal_batch_size` in `predict_batch`;
  inference needs roughly 2–4 GB for batch 16 at 384×384.
- **HuggingFace download fails / 401** — the checkpoints and benchmarks are
  public (no token needed); check your `HF_HOME` and proxy settings.
- **Slow first call** — the first forward pass compiles decoder kernels;
  subsequent batches are much faster.
- **SMILES is `None` / empty** — the image likely contains no recognizable
  structure; try `smiles_mode=None` to inspect the raw prediction dict and
  per-signal confidences (see "Prediction confidence" in the
  [API reference](api_reference.md)).