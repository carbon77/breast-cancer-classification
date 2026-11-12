"""Full-dataset acceptance checks. Run python -m tests.verify_inputs."""
from pathlib import Path

import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from busi.loading import loaders


if __name__ == "__main__":
    torch.set_num_threads(2)
    split = Path("artifacts/splits/split_v1.csv")
    frame = pd.read_csv(split)
    manifest = pd.read_csv("artifacts/manifest.csv")
    excluded = pd.read_csv("artifacts/audit.csv").query("issue == 'conflicting_labels'").files
    assert frame.filepath.is_unique
    assert set(frame.filepath) == set(manifest.filepath) - set(excluded)
    assert frame.groupby("group_id").split.nunique().max() == 1
    for weights in (None, ResNet18_Weights.DEFAULT, ViT_B_16_Weights.DEFAULT):
        batches, config = loaders("data/busi", split, weights=weights)
        for name, loader in batches.items():
            x, y = next(iter(loader))
            assert x.shape == (32, 3, 224, 224) and torch.isfinite(x).all()
            assert y.min() >= 0 and y.max() <= 2
            if name != "train":
                assert torch.equal(x, next(iter(loader))[0])
        print(weights, config)
