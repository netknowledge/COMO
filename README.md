# COMO

**COMO** (**C**losed-loop **O**ptical **M**olecule rec**O**gnition) is a deep learning framework that recognizes chemical structure diagrams from images and predicts SMILES strings with atom-level coordinates and bond matrices. It uses Minimum Risk Training (MRT) to directly optimize molecular-level, non-differentiable objectives.

[Paper](https://arxiv.org/abs/2604.23546) $\cdot$ [PyPI](https://pypi.org/project/como-ocsr/) $\cdot$ [Hugging Face](https://huggingface.co/Keylab/COMO)

## Installation

```bash
pip install como-ocsr
```

Training requires a repository clone:

```bash
git clone https://github.com/netknowledge/COMO
cd COMO
pip install -e ".[training]"
```

## Quick Start

```python
from como.model import ComoModel
model = ComoModel.from_pretrained("Keylab/COMO-Tanimoto", device="cuda")

# or download checkpoint from HuggingFace:
# huggingface-cli download Keylab/COMO models/tanimoto/final.pth
import como
model = como.load_model("models/tanimoto/final.pth", device="cuda")

smiles = como.predict(model, "examples/sample_images/Metformin.png")
# -> CN(C)C(=N)N=C(N)N
```

See the [getting-started guide](doc/getting_started.md) for downloading checkpoints and the [examples](examples/) folder for runnable scripts and a notebook.

## Training

The complete training pipeline is released in this repository: supervised MLE training, Minimum Risk Training (MRT), and the interleaved MLE + MRT joint recipe used for the released checkpoints.

```bash
# data preparation (idempotent; skips already-complete steps)
python scripts/data/prepare_pubchem_smiles.py
python scripts/data/prepare_uspto_mol.py
python scripts/data/prepare_molparser_mrt.py
python scripts/data/prepare_mol_discriminator.py
python scripts/data/prepare_benchmarks.py

# paper recipe: joint MLE + MRT with the visual reward (4 GPUs)
torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt \
    --mrt_every_n_steps 4 \
    --reward_mode visual \
    --visual_reward_encoder data/mol_discriminator/20260321_final.pth
```

All recipes, data licenses, and the exact hyperparameters of the released checkpoints: the [training guide](doc/training.md). Training data: [Keylab/COMO `training_data/`](https://huggingface.co/Keylab/COMO/tree/main/training_data).

## Documentation

| Document | Contents |
|----------|----------|
| [doc/getting_started.md](doc/getting_started.md) | Install, checkpoint download, first prediction, benchmark evaluation |
| [doc/api_reference.md](doc/api_reference.md) | Full API reference (`predict`, `predict_batch`, `evaluate`, confidence signals, benchmarks) |
| [doc/training.md](doc/training.md) | Training guide: recipes, data pipeline, licenses, reproducibility |
| [examples/](examples/) | Runnable inference scripts + a guided notebook |

## Model Weights

Pre-trained model weights are on HuggingFace
(**[Keylab/COMO](https://huggingface.co/Keylab/COMO)**):

| Checkpoint | Reward Mode | Description |
|-----------|-------------|-------------|
| `models/tanimoto/final.pth` | Tanimoto | Joint MLE+MRT (Tanimoto reward) |
| `models/edit_distance/final.pth` | Edit Distance | Joint MLE+MRT (Edit Distance reward) |
| `models/visual/final.pth` | Visual | Joint MLE+MRT (Visual reward) |

## Benchmark Datasets

Published as a HF Dataset with one config per benchmark:
**[Keylab/OCSR-Benchmarks](https://huggingface.co/datasets/Keylab/OCSR-Benchmarks)**
(`CLEF`, `JPO`, `UOB`, `USPTO`, `USPTO-10K`, `Staker`, `ACS`, `WildMol-10K`,
`Indigo`, `ChemDraw`) — full table and loading snippets in the
[API reference](doc/api_reference.md#benchmark-datasets).

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

**Code and model weights** — all of COMO's source code (including the `training/` pipeline and the examples) and the released `.pth` model weights are licensed under **CC BY-NC 4.0** (non-commercial use only). See [LICENSE](LICENSE) for the full license text.

**Training Data**: mixed sources — see
  [doc/training.md](doc/training.md) for the full table:
  - PubChem subset: [PubChem](https://pubchem.ncbi.nlm.nih.gov/) SMILES (public-domain; see PubChem terms)
  - USPTO Mol: MolScribe (`yujieq/MolScribe`); USPTO data is public domain
  - MolParser-SFT real images: [UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M) — **non-commercial use only**

**Benchmark Datasets**: collected from existing public OCSR benchmarks; please refer to their original sources for license and attribution (table in [doc/training.md](doc/training.md#2-data-preparation)).