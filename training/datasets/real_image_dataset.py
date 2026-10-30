"""
COMO Real Image Dataset
=======================

Dataset for loading real-world molecule images + GT SMILES from CSV.

Used for RL finetuning (MRT) on real patent images.

CSV columns: image_id, file_path, SMILES
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RealImageDataset(Dataset):
    """Dataset that loads real-world molecule images + GT SMILES from CSV.

    Returns (image_tensor, gt_smiles_str) pairs for RL finetuning.
    """

    def __init__(self, csv_path: str, image_dir: str,
                 transforms, image_size: Tuple[int, int] = (384, 384),
                 max_samples: Optional[int] = None):
        import pandas as pd
        label_df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples < len(label_df):
            label_df = label_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.items = []
        for _, row in label_df.iterrows():
            img_id = str(row['image_id'])
            img_path = os.path.join(image_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                continue
            gt_smi = str(row['SMILES'])
            self.items.append({'img_path': img_path, 'gt_smiles': gt_smi})

        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item['img_path']).convert('RGB')
            tensor = self.transforms(image=np.array(img))['image']
        except Exception:
            tensor = torch.zeros(3, *self.image_size)
        return tensor, item['gt_smiles']


def _real_collate_fn(batch):
    """Simple collate: stack images, collect SMILES strings."""
    images = torch.stack([b[0] for b in batch])
    smiles = [b[1] for b in batch]
    return images, smiles
