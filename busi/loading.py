"""Common classification inputs for the baseline CNN and pretrained networks."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from . import CLASSES


def rgb(root, rel):
    with Image.open(root / rel) as image:
        return image.convert("RGB")


def normalization(root, train):
    sums = np.zeros(3)
    squares = np.zeros(3)
    pixels = 0
    for rel in train.filepath:
        image = rgb(root, rel).resize((224, 224), Image.Resampling.BILINEAR)
        x = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255
        sums += x.sum(axis=0)
        squares += (x*x).sum(axis=0)
        pixels += len(x)
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 0))
    return mean.tolist(), np.where(std > 0, std, 1).tolist()


class BUSIDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root = root
        self.samples = list(zip(rows.filepath, rows["class"]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(rgb(self.root, path)), CLASSES.index(label)


def loaders(root, split_path, weights=None, batch_size=32, seed=42):
    root, split_path = Path(root), Path(split_path)
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest != split_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Split checksum mismatch")
    rows = pd.read_csv(split_path)
    if weights is None:
        mean, std = normalization(root, rows[rows.split == "train"])
    else:
        preset = weights.transforms()
        mean, std = list(preset.mean), list(preset.std)
    config = dict(size=[224, 224], channels="RGB", interpolation="bilinear",
                  mean=mean, std=std, classes=list(CLASSES), split_sha256=digest)
    base = [T.Resize((224, 224))]
    augment = [T.RandomHorizontalFlip(), T.RandomRotation(10, interpolation=T.InterpolationMode.BILINEAR),
               T.RandomResizedCrop(224, scale=(.85, 1), ratio=(1, 1)),
               T.ColorJitter(brightness=.1, contrast=.1)]
    post = [T.ToTensor(), T.Normalize(mean, std)]
    result = {}
    for name in ("train", "validation", "test"):
        transform = T.Compose(base + (augment if name == "train" else []) + post)
        ds = BUSIDataset(root, rows[rows.split == name], transform)
        result[name] = DataLoader(ds, batch_size=batch_size, shuffle=name == "train",
                                  generator=torch.Generator().manual_seed(seed))
    return result, config
