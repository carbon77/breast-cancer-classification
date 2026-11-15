"""Проверки на полном датасете: запуск через python -m tests.verify_inputs."""

import os
from pathlib import Path

import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from busi.loading import loaders

if __name__ == "__main__":
    torch.set_num_threads(2)
    split = Path("artifacts/splits/split_v2.csv")
    frame = pd.read_csv(split)
    manifest = pd.read_csv("artifacts/manifest.csv")
    excluded = (
        pd.read_csv("artifacts/audit.csv").query("issue == 'conflicting_labels'").files
    )
    # Split должен содержать все допустимые снимки ровно по одному разу,
    # точные копии исключены.
    assert frame.filepath.is_unique
    assert set(frame.filepath) == set(
        manifest[~manifest.filepath.isin(excluded)].drop_duplicates("sha256").filepath
    )
    assert (
        not manifest[manifest.filepath.isin(frame.filepath)].sha256.duplicated().any()
    )
    for weights in (None, ResNet18_Weights.DEFAULT, ViT_B_16_Weights.DEFAULT):
        # Проверяем входы базовой модели и нормализацию для CNN/ViT.
        batches, config = loaders(
            os.environ.get("BUSI_ROOT", "data/busi"), split, weights=weights
        )
        for name, loader in batches.items():
            x, y = next(iter(loader))
            assert x.shape == (32, 3, 224, 224) and torch.isfinite(x).all()
            assert y.min() >= 0 and y.max() <= 2
            if name != "train":
                # Повторная итерация оценки не должна менять входные тензоры.
                assert torch.equal(x, next(iter(loader))[0])
        print(weights, config)
