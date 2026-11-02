# COMO Examples

Runnable, self-contained inference examples for the pip package
(`pip install como-ocsr`). No training data or repository clone is needed —
only a model checkpoint from
[HuggingFace](https://huggingface.co/Keylab/COMO):

```bash
pip install como-ocsr huggingface_hub
hf download Keylab/COMO models/edit_distance/final.pth --local-dir checkpoints
```

| File | What it shows |
|------|---------------|
| [`predict_single.py`](predict_single.py) | Minimal CLI: one image → SMILES (optional confidence) |
| [`predict_batch.py`](predict_batch.py) | Batch prediction over the bundled samples + exact / tautomer match against ground truth |
| [`como_tutorial.ipynb`](como_tutorial.ipynb) | Guided notebook: download checkpoint → predict → tabulate vs ground truth → RDKit rendering of any failures → tautomer analysis |
| [`sample_images/`](sample_images/) | 10 common drug-molecule depictions (the model input) |
| [`examples.csv`](examples.csv) | Ground-truth SMILES for the samples (columns: `name`, `SMILES`) |

## Run the scripts

```bash
# single image (greedy decode)
python examples/predict_single.py --checkpoint checkpoints/models/edit_distance/final.pth

# with a confidence score
python examples/predict_single.py --checkpoint ... --confidence

# all bundled samples with ground-truth comparison
python examples/predict_batch.py --checkpoint checkpoints/models/edit_distance/final.pth
```

Both scripts run on GPU automatically if available, else CPU. From a
repository clone (without `pip install`), they pick up the local `como`
package themselves.

## Run the notebook

```bash
pip install jupyter
jupyter lab examples/como_tutorial.ipynb
```

The notebook downloads the checkpoint via `huggingface_hub` on first run
(set the `COMO_CHECKPOINT` environment variable to reuse a local copy).

## About the sample images

The PNGs in `sample_images/` are rendered with COMO's bundled
[Indigo](https://github.com/epam/Indigo)-based drawing engine (Apache-2.0)
directly from the SMILES strings in `examples.csv` — so image provenance is
fully self-contained and the ground truth is unambiguous. This is the same
renderer used to synthesize the model's MLE training data (see
[doc/training.md](../doc/training.md)).

For training your own / fine-tuning the released checkpoints, see the
[training guide](../doc/training.md). Full API details:
[doc/api_reference.md](../doc/api_reference.md).