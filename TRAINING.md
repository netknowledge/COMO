# Training COMO

This guide reproduces the full COMO training pipeline:

- **MLE training** — supervised pre-training from scratch (synthetic PubChem
  renders + real USPTO patent images).
- **Minimum Risk Training (MRT)** — candidate sampling, molecular reward
  computation (Edit Distance, Tanimoto, Visual), and the joint MLE + MRT
  optimization.
- **Custom-dataset fine-tuning** — adapting COMO to your own molecular
  structure images.
- **Reproducibility** — exact configurations used for the released
  checkpoints.

All training entry points live under `scripts/`, the training loops in
`training/`, and the dataset loaders in `training/datasets/`.

---

## 1. Installation

The pip package (`como-ocsr`) covers *inference/evaluation*. Training requires
the repository source:

```bash
git clone https://github.com/netknowledge/COMO
cd COMO
conda create -n como python=3.10 -y && conda activate como   # any >=3.10
pip install -e ".[training]"
```

Notes:

- Training needs CUDA. The released checkpoints were trained on 4 GPUs with an
  effective batch of 256 (`batch_size=64` per GPU × 4 GPUs).
- The synthetic-data renderer uses a vendored [Indigo](https://github.com/epam/Indigo)
  distribution (Apache-2.0) with Linux shared libraries in `training/indigo/`
  — **training runs on Linux** (the inference package itself is OS-independent).
- DDP is used for multi-GPU (`torchrun`); every script also runs as plain
  `python scripts/…` on a single GPU.

## 2. Data preparation

Four datasets together reproduce the paper recipe (1M synthetic + 652K
USPTO MLE + 83K MolParser-SFT MRT + benchmarks for validation):

```bash
# (a) PubChem SMILES subset (59 MB) — the pinned 1M subset from the paper
python scripts/data/prepare_pubchem_smiles.py

# (b) USPTO Mol (680K real patent renders, from yujieq/MolScribe) → 652K normalised
python scripts/data/prepare_uspto_mol.py
python scripts/data/normalise_rgroup.py \
    --input  data/uspto_mol/train_680k.csv \
    --output data/uspto_mol/train_680k_normalised.csv

# (c) MolParser-SFT real images (~83K; MRT data, non-commercial license)
python scripts/data/prepare_molparser_mrt.py

# (d) MolDiscriminator — frozen visual-reward judge for the visual arm (§7)
python scripts/data/prepare_mol_discriminator.py

# (e) Benchmarks for validation (file-based layout used by the training loops)
python scripts/data/prepare_benchmarks.py          # default: USPTO + JPO
```

After preparation:

```text
data/
├── pubchem_smiles/
│   └── pubchem_1M_seed2026.txt        # 1,000,000 SMILES (sha256 pinned)
├── uspto_mol/
│   ├── train_680k.csv                      # original MolScribe set
│   └── train_680k_normalised.csv           # 652K used by COMO (R-group normalised)
├── molparser_sft_real/
│   ├── sft_real.csv                        # image_id, file_path, SMILES (82,980)
│   └── images/<image_id>.png
└── benchmark/
    ├── real/{USPTO,JPO,...}/  + <name>.csv
    └── synthetic/{indigo,chemdraw}/ + <name>.csv
```

Data sources & licenses:

| Dataset | Source | License |
|---|---|---|
| Synthetic MLE (rendered on the fly) | [PubChem SMILES](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) | public-domain (see PubChem terms) |
| USPTO Mol images | [yujieq/MolScribe](https://huggingface.co/yujieq/MolScribe) | USPTO data (public domain); see [MolScribe](https://github.com/thomas0809/MolScribe) |
| MolParser-SFT real images | [UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M) | **non-commercial use only** |
| MolDiscriminator (visual judge, §7) | trained by us | weights CC BY-NC 4.0 |
| Benchmarks | [Keylab/OCSR-Benchmarks](https://huggingface.co/datasets/Keylab/OCSR-Benchmarks) | see original sources |
| Indigo renderer (vendored) | [EPAM Indigo](https://github.com/epam/Indigo) | Apache-2.0 |

Everything is also mirrored in the Hugging Face repo
**[Keylab/COMO](https://huggingface.co/Keylab/COMO)** under `training_data/`
(the prepared MolParser-SFT subset, the pinned PubChem subset, the full
122M-SMILES pool, and the MolDiscriminator visual-judge checkpoint).

## 3. Recommended recipe: joint MLE + MRT from scratch

`scripts/train/train_joint_mle_mrt.py` — interleaves supervised MLE on
synthetic + USPTO data with Minimum Risk Training on real images:
MLE-only warmup (default 5 epochs), then every 4th optimizer step carries an
MRT backward with `L = L_mle + 0.1 · L_mrt`.

```bash
# Paper recipe, visual reward (judge = frozen MolDiscriminator)
torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt \
    --mrt_every_n_steps 4 \
    --reward_mode visual \
    --visual_reward_encoder data/mol_discriminator/20260321_final.pth

# Tanimoto / edit-distance reward arms
torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt \
    --mrt_every_n_steps 4 \
    --reward_mode tanimoto

torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt \
    --mrt_every_n_steps 4 \
    --reward_mode edit_distance

# Resume after interruption
torchrun --nproc_per_node=4 scripts/train/train_joint_mle_mrt.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt --resume

# 3-epoch smoke test (real batch shapes to stress VMEM, but tiny total data amount) on one GPU
python scripts/train/train_joint_mle_mrt.py --reward_mode tanimoto --fast_test
```

Checkpoints land in `checkpoints/COMO_joint/<reward_mode>/` (`best.pth`,
`final.pth`, `checkpoint_resume.pth`), TensorBoard logs under `logs/`.

Key MRT options:

| Flag | Default | Meaning |
|---|---|---|
| `--mrt_warmup_epochs` | 5 | MLE-only epochs before MRT kicks in |
| `--mrt_weight` | 0.1 | λ weight of the MRT loss |
| `--mrt_batch_size` | 16 | MRT images per GPU batch |
| `--mrt_sampling_method` | multinomial | candidate sampling method (`multinomial` / `beam_search`) |
| `--mrt_n_samples` | 32 | candidates N per image (beam width for beam mode) |
| `--mrt_alpha` | 1.0 | risk sharpening exponent α |
| `--mrt_every_n_steps` | 0 (auto) | one MRT backward every N optimizer steps |
| `--reward_mode` | visual | `visual` / `tanimoto` / `edit_distance` |
| `--adaptive_n` | off | adaptive-N MRT (beam mode): `N(x)=clip(γ·ESS, N_min, N_max)` |
| `--visual_reward_encoder` | COMO encoder | path to an external SiameseNetwork encoder (see §7) |

## 4. Reproducing the released checkpoints

The released `COMO_joint/{visual,tanimoto,edit_distance}/final.pth` weights on
[Hugging Face](https://huggingface.co/Keylab/COMO) were produced by the
command in §3 with **all defaults**:

| Hyperparameter | Value |
|---|---|
| MLE data | 1M PubChem synthetic (rendered on the fly) + 652K USPTO patent images |
| MRT data | 82,980 MolParser-SFT real images |
| Total MLE samples / epoch | 1,652,330 |
| Epochs | 30 (5-epoch warmup, `mrt_every_n_steps=4`) |
| Effective MLE batch | 256 = `--batch_size 64` × 4 GPUs |
| MRT batch | 16 per GPU; sampled N=32, multinomial, temperature 0.5 |
| Reward | weights validity 0.1 / similarity 0.5 / exact 0.4, α = 1.0, λ = 0.1 |
| Visual judge | frozen **MolDiscriminator** SiameseNetwork (`20260321_final` checkpoint); Tanimoto / edit-distance arms use no visual judge |
| Optimizer | AdamW; encoder_lr = decoder_lr = 4e-4; weight decay 1e-6 |
| LR schedule | cosine with 2% linear warmup |
| Architecture | Swin-B@384 encoder, 2D positional encoding, 6-layer decoder (d_model 256, nhead 8, FF 1024), bond predictor, n_bins 64 for coordinate quantization|
| Loss precision | mixed-precision (AMP); torch.compile on the encoder (`COMO_COMPILE=0` to disable) |
| Seed | 2026 |

Validation during training uses the file-based benchmarks (USPTO + JPO by
default, first listed = early-stopping metric) and saves `best.pth`.

The three released arms differ only in the reward: `--reward_mode visual` plus
the external MolDiscriminator judge, or `--reward_mode tanimoto` /
`--reward_mode edit_distance` with no visual encoder. The two-stage pipeline
below reproduces the ablation history
(`COMO_stage1_* → COMO_stage2_mrt*`).

## 5. Legacy two-stage pipeline (ablations)

Supervised pre-training first, then MRT fine-tuning on the real images.
The Stage-2 MRT recipe (standalone finetune): 6 epochs, MRT batch 16,
N=32 candidates (multinomial, temperature 0.5), α = 1.0, reward weights
0.1 / 0.5 / 0.4 (validity / similarity / exact); optimizer: encoder and
decoder at 1e-5 (~40× below the MLE-stage 4e-4 — right when only the MRT
loss drives the update, with no MLE anchor), weight decay 1e-6, warmup 2%,
patience 10. Optional: MLE regularization on PubChem synthetic data with
`--mle_weight 0.3` (recommended to prevent drift; default off).

```bash
# Stage 1 — MLE pre-training (joint = 1M synthetic + 680K USPTO, default)
torchrun --nproc_per_node=4 scripts/train/train_stage1.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt

torchrun --nproc_per_node=4 scripts/train/train_stage1.py --data_mode synthetic --smiles_list ...

# Stage 2 — MRT finetune on MolParser-SFT (defaults to Stage-1 joint final.pth)
torchrun --nproc_per_node=4 scripts/train/train_stage2_83K_MolParser.py --reward_mode visual
torchrun --nproc_per_node=4 scripts/train/train_stage2_83K_MolParser.py \
    --mle_weight 0.3 \
    --mle_smiles_path data/pubchem_smiles/pubchem_1M_seed2026.txt
```

The MLE-regularized Stage-2 variant (`--mle_weight 0.3`) interleaves teacher
-forced MLE steps on PubChem synthetic data to prevent distribution drift.

### Ablation: interleaved MLE without MRT

`scripts/train/train_ablation_joint_mle.py` keeps the joint schedule but replaces
the MRT loss with teacher-forced MLE on the same interleaved MolParser data —
isolating the effect of the MRT mechanism from the extra data:

```bash
torchrun --nproc_per_node=4 scripts/train/train_ablation_joint_mle.py \
    --smiles_list data/pubchem_smiles/pubchem_1M_seed2026.txt
```

## 6. Fine-tuning on custom data

### MRT fine-tuning on your own molecule images

MRT needs (image, ground-truth SMILES) pairs in a simple CSV:

```csv
image_id,file_path,SMILES
0,mydata/images/0.png,CC(=O)Oc1ccccc1C(=O)O
1,mydata/images/1.png,C1=CC=C(C=C1)C=O
```

Point a training run at it (path columns resolve relative to the data dir;
images may live anywhere):

```python
from training.train_mrt import train_mrt

train_mrt(
    # data: list of CSVs + their image dirs
    train_csv_paths=["mydata/labels.csv"],
    train_image_dirs=["mydata/images"],

    # continue from a pretrained checkpoint (fresh optimizer)
    pretrained_path="checkpoints/COMO_joint/edit_distance/final.pth",

    # output dir for best/final/resume checkpoints + TensorBoard logs
    save_path="checkpoints/my_mrt_finetune/",

    # architecture must match the checkpoint
    image_size=(384, 384), n_bins=64, backbone='swin_b',
    d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024,

    # MRT settings — 
    # in the joint pipeline MRT gradients share the MLE optimizer (4e-4, λ=0.1); 
    # while the (legacy two-stage-style) MRT finetune recipe
    # run an MRT-only finetune at a much lower LR (6 epochs, batch 16, lr 1e-5), which is used in SwinOCSR MRT fine-tuning experiment.
    num_epochs=6, batch_size=16,
    encoder_lr=1e-5, decoder_lr=1e-5,       # ~40x below the MLE-stage LR (4e-4)
    rl_temperature=0.5, rl_n_samples=32,
    reward_mode="edit_distance",
)
```

A runnable template with resume, validation-on-benchmarks, AMP and DDP
wiring is `scripts/train/train_stage2_83K_MolParser.py` — point its
`mrt_csv_paths` / `mrt_image_dirs` (two lines) at your data.

### MLE from scratch on your own synthetic data

MLE trains on SMILES lists rendered on the fly by
`training/datasets/molecule_dataset.py` (`MoleculeDataset`), so the simplest
custom route is to pass your own SMILES list:

```python
from training.train_mle import train_mle
train_mle(synthetic_smiles_list=my_smiles, ...)
```

If your images already exist on disk, use the USPTO-style CSV
(`file_path, SMILES, node_coords, [edges]`) with `train_mle(uspto_csv_path=...)`
(`USPTOMolDataset`; `node_coords`/`edges` optional, but then the bond head
trains without GT edges), or subclass `MoleculeDataset` /
`training.datasets` loaders and combine different sources with
`torch.utils.data.ConcatDataset` inside your own entry script that calls
`train_mle` — the training loop itself is dataset-agnostic on the MLE side.

### Markush / patent-style labels

GT labels containing R-group and decoration tokens (e.g. `[R]`, `[CH3]`,
Ar-abbreviations) are supported natively — that is exactly the label style of
the MolParser-SFT data used for MRT (see
`convert_esmiles.py` for reproducing those labels from
[UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M)
E-SMILES). For rendering such wild SMILES internally, the drawing engine
(`training/drawing_engine.py`) keeps the labels as displayed atoms.

## 7. The visual reward in detail

For `reward_mode=visual`:

1. **Both sides are rendered**: the ground-truth SMILES *and* every sampled
   prediction are rendered back to images (`--visual_reward_renderer`:
   the Indigo-based drawing engine by default, `rdkit` as a lighter
   alternative). Ground-truth renders are cached per run.
2. **A frozen image encoder embeds the renders** and the reward similarity is
   the cosine similarity between the embeddings:
   `Sim = cos(g(Render(pred)), g(Render(GT)))`.
3. **The paper's judge (`g`) is the MolDiscriminator** — a Siamese
   EfficientNet-V2-S encoder (InfoNCE, CReM hard negatives) trained
   contrastively on rendered molecule pairs. The exact checkpoint used by the
   released COMO-Visual model is on Hugging Face:
   `training_data/mol_discriminator/20260321_final.pth` (fetch with
   `python scripts/data/prepare_mol_discriminator.py`) and is passed via
   `--visual_reward_encoder data/mol_discriminator/20260321_final.pth` — this
   is how the released `COMO_joint/visual` checkpoints were trained, and the
   command in §3 for `--reward_mode visual` includes it. To re-train the
   encoder yourself: `python scripts/train/train_MolDiscriminator.py`
   (500K PubChem SMILES, 3 epochs, batch 48/GPU, lr 3e-4, 3% warmup by
   default; optional `crem` hard negatives; internal DDP via `mp.spawn` —
   do NOT wrap in torchrun).
4. **Code fallback**: if `--visual_reward_encoder` is omitted, the judge
   becomes COMO's own image encoder (a frozen copy taken at MRT start,
   spatially pooled). This is a built-in alternative that needs no extra
   checkpoint, but it is *not* the configuration used for the released
   checkpoints. The Tanimoto and edit-distance arms use no visual judge at all.

## 8. Evaluating a trained checkpoint

```python
import como

model = como.load_model("checkpoints/COMO_joint/visual/final.pth")
metrics = como.evaluate(
    model,
    hf_dataset="Keylab/OCSR-Benchmarks",
    hf_config="USPTO",
)
print(f"Exact match: {metrics['postprocess/exact_match_acc']:.2%}")

# Multi-benchmark, multi-GPU
benchmarks = [
    {"name": "USPTO", "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "USPTO"},
    {"name": "JPO",   "hf_dataset": "Keylab/OCSR-Benchmarks", "hf_config": "JPO"},
]
results = como.evaluate_benchmarks(model, benchmarks, gpus="0,1")
```