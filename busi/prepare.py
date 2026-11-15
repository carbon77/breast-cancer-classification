"""Проверка BUSI, учёт точных копий и воспроизводимое разбиение данных."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from . import CLASSES

# У одного снимка может быть несколько масок: name_mask.png, name_mask_1.png и т. д.
MASK = re.compile(r"(.+)_mask(?:_\d+)?$", re.I)


def audit(root):
    """Вернуть манифест читаемых снимков и список проблем с файлами и масками.

    root — Path с каталогами классов. Маски связываются со снимками по имени
    внутри класса; сами маски в строки манифеста не попадают.
    """
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
                # После verify() файл нужно открыть заново для загрузки пикселей.
                with Image.open(path) as im:
                    im.load()
                    if match:
                        masks.setdefault(match[1], []).append((rel, im.size))
                        continue
                    scans[path.stem] = dict(
                        filepath=rel,
                        width=im.width,
                        height=im.height,
                        channels=len(im.getbands()),
                        mode=im.mode,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
            except (OSError, ValueError, SyntaxError):
                issues.append(("invalid_png", rel))
        for name, scan in scans.items():
            # pop убирает сопоставленные маски; оставшиеся ниже считаются сиротами.
            associated = masks.pop(name, [])
            if not associated:
                issues.append(("missing_mask", scan["filepath"]))
            for rel, size in associated:
                if size != (scan["width"], scan["height"]):
                    issues.append(("mask_size_mismatch", rel))
            rows.append(
                {
                    **scan,
                    "class": label,
                    "masks": json.dumps([p for p, _ in associated]),
                }
            )
        for items in masks.values():
            issues.extend(("orphan_mask", p) for p, _ in items)
    return pd.DataFrame(rows), issues


def eligible_scans(frame, issues):
    """Оставить одну копию файла; одинаковые файлы с разными метками исключить."""
    duplicates = frame[frame.sha256.duplicated(keep=False)]
    excluded = []
    for _, copies in duplicates.groupby("sha256"):
        paths = copies.filepath.tolist()
        issues.append(("exact_duplicate", " | ".join(paths)))
        if copies["class"].nunique() > 1:
            issues.extend(("conflicting_labels", path) for path in paths)
            excluded.extend(paths)
        else:
            excluded.extend(paths[1:])
    return frame[~frame.filepath.isin(excluded)]


def stratify(frame):
    """Перемешать пути внутри каждого класса и разделить в пропорции 70/15/15."""
    rng = random.Random(42)
    rows = []
    for label in CLASSES:
        paths = sorted(frame.loc[frame["class"] == label, "filepath"])
        rng.shuffle(paths)
        train_end = round(0.70 * len(paths))
        validation_end = train_end + round(0.15 * len(paths))
        for name, subset in (
            ("train", paths[:train_end]),
            ("validation", paths[train_end:validation_end]),
            ("test", paths[validation_end:]),
        ):
            rows.extend((path, label, name) for path in subset)
    return pd.DataFrame(rows, columns=["filepath", "class", "split"])


def prepare(root, out):
    """Провести аудит, сохранить манифест, split и отчёты; вернуть статистику.

    root и out — Path. Уже сохранённый split_v2 разрешено записать повторно
    только при побайтовом совпадении, чтобы не подменить разбиение эксперимента.
    """
    frame, issues = audit(root)
    eligible = eligible_scans(frame, issues)
    split = stratify(eligible)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    path = out / "splits/split_v2.csv"
    content = split.to_csv(index=False, lineterminator="\n").encode()
    # Явный перевод строки обеспечивает одинаковые байты CSV на разных ОС.
    if path.exists() and path.read_bytes() != content:
        raise ValueError(
            "Existing split_v2 differs. Use a new output directory to review the dataset revision."
        )
    path.write_bytes(content)
    # Загрузчик сверяет эту сумму перед использованием разбиения.
    digest = hashlib.sha256(content).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    frame.to_csv(out / "manifest.csv", index=False)
    # Манифест включает и снимки, исключённые из split из-за конфликта меток.
    pd.DataFrame(issues, columns=["issue", "files"]).to_csv(
        out / "audit.csv", index=False
    )
    stats = dict(
        classes=frame["class"].value_counts().to_dict(),
        issues=pd.Series([k for k, _ in issues]).value_counts().to_dict(),
        split=split.groupby(["class", "split"]).size().unstack(fill_value=0).to_dict(),
        seed=42,
        split_sha256=digest,
    )
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/busi"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    if args.kaggle:
        import shutil

        import kagglehub

        source = Path(
            kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
        )
        # В архиве Kaggle каталоги классов могут находиться во вложенной папке.
        source = next(p.parent for p in source.rglob("benign") if p.is_dir())
        shutil.copytree(source, args.root)
    print(json.dumps(prepare(args.root, args.out), indent=2))
