"""Audit BUSI, group duplicate scans, and publish a deterministic split."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from . import CLASSES

MASK = re.compile(r"(.+)_mask(?:_\d+)?$", re.I)


def audit(root):
    rows, issues = [], []
    for label in CLASSES:
        folder = root / label
        if not folder.is_dir():
            raise ValueError(f"Missing class directory: {folder}")
        scans, masks = {}, {}
        for path in sorted(folder.iterdir()):
            rel = path.relative_to(root).as_posix()
            if not path.is_file() or path.suffix.lower() != ".png":
                issues.append(("unexpected_entry", rel))
                continue
            match = MASK.fullmatch(path.stem)
            try:
                with Image.open(path) as im:
                    if im.format != "PNG":
                        raise ValueError("format")
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    if match:
                        masks.setdefault(match[1], []).append((rel, im.size))
                        continue
                    thumb = im.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
                    low = dctn(np.asarray(thumb, dtype=float), type=2)[:8, :8]
                    bits = np.packbits(low > np.median(low)).tobytes().hex()
                    scans[path.stem] = dict(filepath=rel, width=im.width, height=im.height,
                        channels=len(im.getbands()), mode=im.mode, phash=bits,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except (OSError, ValueError, SyntaxError):
                issues.append(("invalid_png", rel))
        for name, scan in scans.items():
            associated = masks.pop(name, [])
            if not associated:
                issues.append(("missing_mask", scan["filepath"]))
            for rel, size in associated:
                if size != (scan["width"], scan["height"]):
                    issues.append(("mask_size_mismatch", rel))
            rows.append({**scan, "class": label, "masks": json.dumps([p for p, _ in associated])})
        for items in masks.values():
            issues.extend(("orphan_mask", p) for p, _ in items)
    return pd.DataFrame(rows), issues


def group_scans(frame, issues, patients=None):
    parents = list(range(len(frame)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    hashes = [int(h, 16) for h in frame.phash]
    paths = frame.filepath.tolist()
    digests = frame.sha256.tolist()
    for i, path in enumerate(paths):
        for j in range(i):
            near = (hashes[i] ^ hashes[j]).bit_count() <= 4
            same_patient = patients and patients.get(path) and patients[path] == patients.get(paths[j])
            if near or same_patient:
                parents[find(i)] = find(j)
            if near:
                kind = "exact_duplicate" if digests[i] == digests[j] else "near_duplicate"
                issues.append((kind, f"{paths[j]} | {path}"))
    frame["group_id"] = [paths[find(i)] for i in range(len(frame))]
    mixed = frame.groupby("group_id")["class"].transform("nunique") > 1
    issues.extend(("conflicting_labels", path) for path in frame.loc[mixed, "filepath"])
    return frame.loc[~mixed].copy()


def stratify(frame):
    rng = random.Random(42)
    assignments = {}
    names = ["train", "validation", "test"]
    for label in CLASSES:
        groups = list(frame.loc[frame["class"] == label].groupby("group_id").size().items())
        rng.shuffle(groups)
        groups.sort(key=lambda pair: -pair[1])
        targets = np.array([.70, .15, .15]) * sum(n for _, n in groups)
        counts = np.zeros(3)
        for group, size in groups:
            index = int(np.argmax(targets - counts))
            assignments[group] = names[index]
            counts[index] += size
    return frame.assign(split=frame.group_id.map(assignments))[["filepath", "class", "group_id", "split"]]


def prepare(root, out, patients=None):
    frame, issues = audit(root)
    eligible = group_scans(frame, issues, patients)
    split = stratify(eligible)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    path = out / "splits/split_v1.csv"
    content = split.to_csv(index=False, lineterminator="\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Existing split_v1 differs. Use a new output directory to review the dataset revision.")
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    frame.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(issues, columns=["issue", "files"]).to_csv(out / "audit.csv", index=False)
    stats = dict(classes=frame["class"].value_counts().to_dict(), issues=pd.Series([k for k, _ in issues]).value_counts().to_dict(),
                 split=split.groupby(["class", "split"]).size().unstack(fill_value=0).to_dict(), seed=42, split_sha256=digest)
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/busi"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--patients", type=Path, help="CSV with filepath,patient_id")
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    if args.kaggle:
        import kagglehub
        import shutil
        source = Path(kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset"))
        source = next(p.parent for p in source.rglob("benign") if p.is_dir())
        shutil.copytree(source, args.root)
    patients = None
    if args.patients:
        ids = pd.read_csv(args.patients, dtype=str).dropna()
        patients = ids.set_index("filepath", verify_integrity=True).patient_id.to_dict()
    print(json.dumps(prepare(args.root, args.out, patients), indent=2))
