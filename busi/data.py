"""Dataset audit, duplicate-safe splits and shared PyTorch input pipeline.

Paths in manifests are relative to DATA_ROOT, never machine-specific paths.
No download or preparation happens on import.
"""

import hashlib
import importlib.metadata
import json
import platform
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ("normal", "benign", "malignant")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
SEED = 42
IMAGE_SIZE = 224
DATASET_SLUG = "aryashah2k/breast-ultrasound-images-dataset"
MASK_PATTERN = re.compile(r"_mask.*$", re.IGNORECASE)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment():
    packages = ("numpy", "pandas", "Pillow", "scipy", "scikit-learn", "matplotlib", "torch", "torchvision")
    return {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in packages
    }}


def find_data_root(path):
    """Accept either Dataset_BUSI_with_GT or its Kaggle wrapper directory."""
    path = Path(path).resolve()
    candidates = [path] + sorted(p for p in path.rglob("Dataset_BUSI_with_GT") if p.is_dir())
    matches = [p for p in candidates if all((p / c).is_dir() for c in CLASSES)]
    if len(matches) != 1:
        raise ValueError(f"Expected one BUSI root with {CLASSES}, found {matches}")
    return matches[0]


def phash(image):
    """64-bit pHash: grayscale 32x32 -> 2D DCT -> 8x8 median threshold.

    Matches the conventional ImageHash pHash construction (including DC).
    """
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(pixels, type=2)[:8, :8]
    bits = low > np.median(low)
    return f"{int(''.join('1' if b else '0' for b in bits.ravel()), 2):016x}"


def scan_dataset(root):
    """Inventory ALL files, masks included; quarantine invalid original images."""
    root = find_data_root(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else ""
        row = {"filepath": rel.as_posix(), "class": label, "is_mask": bool(MASK_PATTERN.search(path.stem)),
               "sha256": sha256(path), "status": "ok", "error": "", "width": None,
               "height": None, "mode": "", "channels": None, "phash": "", "mask_count": 0,
               "mean_intensity": None, "std_intensity": None, "dark_fraction": None, "bright_fraction": None}
        try:
            if label not in CLASSES or len(rel.parts) != 2:
                raise ValueError("Unexpected directory; expected class/file.png")
            if path.suffix.lower() != ".png":
                raise ValueError("Expected PNG extension")
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError("File is not PNG")
                image.load()  # Fully decode; Image.open alone does not detect truncated data.
                if min(image.size) <= 0:
                    raise ValueError("Empty image")
                row.update(width=image.width, height=image.height, mode=image.mode, channels=len(image.getbands()))
                if not row["is_mask"]:
                    row["phash"] = phash(image)
                    gray = np.asarray(image.convert("L"), dtype=float) / 255
                    row.update(mean_intensity=float(gray.mean()), std_intensity=float(gray.std()),
                               dark_fraction=float((gray < 0.02).mean()), bright_fraction=float((gray > 0.98).mean()))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            row.update(status="invalid", error=str(exc))
        rows.append(row)
    if not rows:
        raise ValueError("Dataset contains no files")
    frame = pd.DataFrame(rows)
    for idx, row in frame[~frame.is_mask].iterrows():
        image_path = Path(row.filepath)
        masks = frame[frame.is_mask & (frame["class"] == row["class"])]
        frame.loc[idx, "mask_count"] = sum(
            MASK_PATTERN.sub("", Path(p).stem).lower() == image_path.stem.lower()
            for p in masks.filepath
        )
    for label in CLASSES:
        if frame[(frame["class"] == label) & ~frame.is_mask & (frame.status == "ok")].empty:
            raise ValueError(f"No valid original images in class {label}")
    return frame


def group_images(inventory, patient_csv=None, threshold=4):
    """Connected components of pHash <= threshold, exact hashes and optional patients.

    patient_csv must cover every valid original: filepath,patient_id. A filename
    number is NOT a patient identifier. Mixed-label components need manual review.
    """
    frame = inventory[(inventory.status == "ok") & ~inventory.is_mask].copy()
    frame = frame.sort_values("filepath").reset_index(drop=True)
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    if patient_csv is not None:
        patients = pd.read_csv(patient_csv, dtype=str, keep_default_na=False)
        if not {"filepath", "patient_id"} <= set(patients.columns):
            raise ValueError("Patient CSV requires filepath,patient_id")
        if patients.filepath.duplicated().any() or patients.patient_id.str.strip().eq("").any():
            raise ValueError("Patient CSV contains duplicate paths or empty IDs")
        mapping = patients.set_index("filepath").patient_id
        frame["patient_id"] = frame.filepath.map(mapping)
        if frame.patient_id.isna().any():
            raise ValueError("Patient CSV must cover every valid original image")
        for indices in frame.groupby("patient_id").groups.values():
            for idx in indices:
                union(indices[0], idx)

    hashes = [int(value, 16) for value in frame.phash]
    pairs = []
    for i in range(len(frame)):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            exact = frame.at[i, "sha256"] == frame.at[j, "sha256"]
            if exact or distance <= threshold:
                union(i, j)
                pairs.append({"filepath_a": frame.at[j, "filepath"], "filepath_b": frame.at[i, "filepath"],
                              "hamming": distance, "exact": exact})
    frame["group_id"] = [f"g{find(i):05d}" for i in range(len(frame))]
    conflicts = frame.groupby("group_id")["class"].nunique()
    frame["needs_review"] = frame.group_id.isin(conflicts[conflicts > 1].index)
    return frame, pd.DataFrame(pairs, columns=["filepath_a", "filepath_b", "hamming", "exact"])


def create_split(grouped, seed=SEED):
    """Stratify homogeneous components; search group splits nearest 70/15/15.

    GroupShuffleSplit samples groups, not images. Candidate selection minimizes
    image-count error within each class. Indivisible groups imply approximate ratios.
    """
    frame = grouped[~grouped.needs_review].copy().sort_values("filepath").reset_index(drop=True)
    frame["split"] = ""
    targets = np.array([0.70, 0.15, 0.15])
    for label in CLASSES:
        subset = frame[frame["class"] == label]
        if subset.group_id.nunique() < 3:
            raise ValueError(f"{label}: need >=3 independent groups after quarantine; inspect conflicts")
        best, best_score = None, float("inf")
        group_count = subset.group_id.nunique()
        # Leave at least two groups for validation and test, including tiny fixtures.
        train_fraction = min(0.7, (group_count - 2) / group_count + 1e-9)
        first = GroupShuffleSplit(n_splits=128, train_size=train_fraction, random_state=seed)
        for train_idx, rest_idx in first.split(subset, groups=subset.group_id):
            rest = subset.iloc[rest_idx]
            if rest.group_id.nunique() < 2:
                continue
            second = GroupShuffleSplit(n_splits=32, test_size=0.5, random_state=seed)
            for val_idx, test_idx in second.split(rest, groups=rest.group_id):
                parts = (subset.iloc[train_idx].index, rest.iloc[val_idx].index, rest.iloc[test_idx].index)
                score = float(np.square(np.array([len(p) for p in parts]) / len(subset) - targets).sum())
                if score < best_score:
                    best, best_score = parts, score
        if best is None:
            raise ValueError(f"Cannot split {label}; inspect group sizes")
        for name, indices in zip(("train", "validation", "test"), best):
            frame.loc[indices, "split"] = name
    validate_split(frame)
    return frame[["filepath", "class", "group_id", "split"]]


def validate_split(frame):
    if frame.filepath.duplicated().any() or frame.empty:
        raise ValueError("Duplicate paths or empty split")
    if set(frame["class"]) != set(CLASSES) or set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Invalid class/split names")
    if (frame.groupby("group_id").split.nunique() != 1).any():
        raise ValueError("Group leakage across splits")
    if (pd.crosstab(frame["class"], frame.split).reindex(CLASSES, fill_value=0) == 0).any().any():
        raise ValueError("Every split must contain all three classes")


def freeze_split(grouped, inventory, output_dir, version="v1", seed=SEED, threshold=4):
    """Never silently overwrite a split; verify bytes, membership and data on rerun."""
    if not re.fullmatch(r"v[1-9]\d*", version):
        raise ValueError("Use split version v1, v2, ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"split_{version}.csv"
    metadata_path = path.with_suffix(".json")
    fingerprint_columns = ["filepath", "class", "sha256", "status", "is_mask"]
    dataset_hash = hashlib.sha256(inventory.sort_values("filepath")[fingerprint_columns].to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    grouping_columns = ["filepath", "class", "group_id", "needs_review"]
    grouping_hash = hashlib.sha256(grouped.sort_values("filepath")[grouping_columns].to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    identity = {"dataset_sha256": dataset_hash, "grouping_sha256": grouping_hash,
                "seed": seed, "phash_threshold": threshold, "class_to_idx": CLASS_TO_IDX,
                "split_algorithm": "per-class-group-search-128x32-v1", "ratios": [0.7, 0.15, 0.15]}
    if path.exists() or metadata_path.exists():
        if not (path.exists() and metadata_path.exists()):
            raise ValueError("Incomplete split artifacts; restore matching CSV and JSON")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in identity.items()) or sha256(path) != metadata["split_sha256"]:
            raise ValueError("Data, grouping, config or split changed. Restore artifacts or use a NEW version.")
        result = pd.read_csv(path, dtype=str)
        validate_split(result)
        expected = grouped[~grouped.needs_review][["filepath", "class", "group_id"]].sort_values("filepath").reset_index(drop=True)
        if not result.sort_values("filepath")[expected.columns].reset_index(drop=True).equals(expected):
            raise ValueError("Split membership differs from audited data")
        return result, metadata
    result = create_split(grouped, seed)
    result.to_csv(path, index=False, lineterminator="\n")
    metadata = {**identity, "split_sha256": sha256(path), "environment": environment(),
                "source": DATASET_SLUG, "patient_ids_available": "patient_id" in grouped.columns,
                "limitations": "pHash groups are not patients; patient leakage remains possible without reliable IDs."}
    write_json(metadata_path, metadata)
    return result, metadata


def train_statistics(root, split):
    """Pixel-weighted mean/std of RGB [0,1] after resize, TRAIN only, no augment."""
    total, squares, pixels = np.zeros(3), np.zeros(3), 0
    train = split[split.split == "train"]
    if train.empty:
        raise ValueError("Empty training split")
    for relative in train.filepath:
        with Image.open(Path(root) / relative) as image:
            values = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.float64) / 255
        total += values.sum(axis=(0, 1))
        squares += np.square(values).sum(axis=(0, 1))
        pixels += values.shape[0] * values.shape[1]
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean ** 2, 0))
    if (std < 1e-8).any():
        raise ValueError("Zero variance in training channels; inspect dataset")
    return {"mean": mean.tolist(), "std": std.tolist(), "images": len(train), "pixels": pixels}


def preprocessing_config(stats, split_sha256):
    return {"split_sha256": split_sha256, "size": [IMAGE_SIZE, IMAGE_SIZE], "channels": 3,
            "color_mode": "RGB", "resize_interpolation": "bilinear", "scale": "[0,1]",
            "class_to_idx": CLASS_TO_IDX,
            "baseline": {"mean": stats["mean"], "std": stats["std"]},
            "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "augmentation": {"horizontal_flip_p": 0.5, "rotation_degrees": 10,
                             "crop_scale": [0.85, 1.0], "crop_ratio": [1.0, 1.0],
                             "brightness": 0.1, "contrast": 0.1}}


def convert_rgb(image):
    """Named function so transforms remain picklable for Windows workers."""
    return image.convert("RGB")


def make_transform(config, split="validation", normalization="baseline"):
    from torchvision import transforms as T

    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if normalization not in ("baseline", "imagenet"):
        raise ValueError("Unknown normalization")
    size = tuple(config["size"])
    # Resize is shared with train_statistics; stochastic transforms follow resize.
    operations = [T.Lambda(convert_rgb), T.Resize(size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if split == "train":
        aug = config["augmentation"]
        operations += [T.RandomHorizontalFlip(aug["horizontal_flip_p"]),
                       T.RandomRotation(aug["rotation_degrees"], interpolation=T.InterpolationMode.BILINEAR),
                       T.RandomResizedCrop(size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"]),
                                           interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                       T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"])]
    parameters = config[normalization]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(parameters["mean"], parameters["std"])])


def seed_everything(seed=SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(root, split, config, normalization="baseline", batch_size=32, seed=SEED, num_workers=0):
    import torch
    from torch.utils.data import DataLoader

    validate_split(split)
    loaders = {}
    for name in ("train", "validation", "test"):
        dataset = BUSIDataset(root, split[split.split == name], make_transform(config, name, normalization))
        generator = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
                                   num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    return loaders


class BUSIDataset:
    """Map-style PyTorch dataset; no masks or patient identifiers enter the model."""

    def __init__(self, root, rows, transform):
        self.root = Path(root).resolve()
        self.rows = rows.reset_index(drop=True).copy()
        self.transform = transform
        for relative in self.rows.filepath:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or MASK_PATTERN.search(path.stem):
                raise ValueError("Unsafe path or mask in classification manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.root / row.filepath) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_TO_IDX[row["class"]]


def class_weights(split):
    counts = split[split.split == "train"]["class"].value_counts().reindex(CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError("Training data missing a class")
    return (counts.sum() / (len(CLASSES) * counts)).to_numpy(dtype=np.float32)
