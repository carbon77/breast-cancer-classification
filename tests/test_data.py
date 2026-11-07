"""Contract checks on generated images, without downloading or running the notebook."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from busi.data import (
    BUSIDataset, CLASSES, class_weights, create_split, freeze_split, group_images,
    make_loaders, make_transform, preprocessing_config, scan_dataset,
    seed_everything, train_statistics, validate_split,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "Dataset_BUSI_with_GT"
        rng = np.random.default_rng(8)
        for label in CLASSES:
            directory = self.root / label
            directory.mkdir(parents=True)
            for idx in range(10):
                Image.fromarray(rng.integers(0, 256, (40, 48), dtype=np.uint8)).save(directory / f"{label} ({idx}).png")
            for suffix in ("_mask", "_mask_1"):
                Image.new("L", (48, 40)).save(directory / f"{label} (0){suffix}.png")

    def test_audit_masks_duplicates_corruption_and_conflicts(self):
        (self.root / "normal/broken.png").write_bytes(b"broken")
        original = self.root / "normal/normal (0).png"
        (self.root / "normal/copy.png").write_bytes(original.read_bytes())
        (self.root / "benign/conflict.png").write_bytes(original.read_bytes())
        Image.new("L", (48, 40)).save(self.root / "normal/normal (0)_mask_extra.png")
        audit = scan_dataset(self.root)
        self.assertEqual(int(audit.is_mask.sum()), 7)
        self.assertEqual(int((audit.status == "invalid").sum()), 1)
        self.assertEqual(audit.set_index("filepath").at["normal/normal (0).png", "mask_count"], 3)
        grouped, pairs = group_images(audit)
        originals = grouped[grouped.filepath.isin(["normal/normal (0).png", "normal/copy.png", "benign/conflict.png"])]
        self.assertEqual(originals.group_id.nunique(), 1)
        self.assertTrue(originals.needs_review.all())
        self.assertEqual(int(pairs.exact.sum()), 3)
        self.assertFalse(grouped.filepath.str.contains("mask|broken").any())

    def test_transitive_duplicates_and_patient_groups(self):
        audit = scan_dataset(self.root)
        images = audit[~audit.is_mask].copy()
        # A~B, B~C, A!~C must still form one component.
        images.loc[images.index[:3], "phash"] = ["0000000000000000", "000000000000000f", "00000000000000ff"]
        grouped, _ = group_images(images)
        self.assertEqual(grouped.iloc[:3].group_id.nunique(), 1)
        patients = images[["filepath"]].copy()
        patients["patient_id"] = [f"patient-{i}" for i in range(len(patients))]
        patients.loc[patients.index[:2], "patient_id"] = "same-patient"
        path = Path(self.tmp.name) / "patients.csv"
        patients.to_csv(path, index=False)
        grouped, _ = group_images(audit, path)
        first_two = grouped[grouped.filepath.isin(patients.filepath.iloc[:2])]
        self.assertEqual(first_two.group_id.nunique(), 1)
        patients.iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "cover"):
            group_images(audit, path)

    def test_split_freeze_and_tamper_detection(self):
        audit = scan_dataset(self.root)
        grouped, _ = group_images(audit)
        folder = Path(self.tmp.name) / "splits"
        split, metadata = freeze_split(grouped, audit, folder)
        validate_split(split)
        again, same_metadata = freeze_split(grouped, audit, folder)
        pd.testing.assert_frame_equal(split, again)
        self.assertEqual(metadata, same_metadata)
        self.assertEqual(set(split.filepath), set(grouped.filepath))
        pd.testing.assert_frame_equal(split, create_split(grouped.sample(frac=1, random_state=99)))
        changed = audit.copy()
        changed.loc[0, "sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed"):
            freeze_split(grouped, changed, folder)
        with self.assertRaisesRegex(ValueError, "changed"):
            freeze_split(grouped, audit, folder, seed=43)
        path = folder / "split_v1.csv"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            freeze_split(grouped, audit, folder)

    def test_small_split_and_leakage_rejected(self):
        rows = [{"filepath": f"{c}/{i}.png", "class": c, "group_id": f"{c}-{i}", "needs_review": False}
                for c in CLASSES for i in range(3)]
        split = create_split(pd.DataFrame(rows))
        validate_split(split)
        split.loc[0, "group_id"] = split.loc[1, "group_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(split)

    def test_train_only_statistics_and_loaders(self):
        import torch

        audit = scan_dataset(self.root)
        grouped, _ = group_images(audit)
        # Fixed synthetic manifest for this test; splitting itself is tested above.
        split = grouped[["filepath", "class", "group_id"]].copy()
        split["split"] = ["train" if int(Path(p).stem.split("(")[1][:-1]) < 6
                          else "validation" if int(Path(p).stem.split("(")[1][:-1]) < 8
                          else "test" for p in split.filepath]
        stats = train_statistics(self.root, split)
        for path in split[split.split != "train"].filepath:
            Image.new("L", (48, 40), 255).save(self.root / path)
        self.assertEqual(stats, train_statistics(self.root, split))
        config = preprocessing_config(stats, "fixture-hash")
        self.assertEqual(tuple(make_transform(config)(Image.new("L", (48, 40)) ).shape), (3, 224, 224))
        Image.new("L", (48, 40), 120).save(self.root / "normal/normal (9).png")
        validation = BUSIDataset(self.root, split[split.split == "test"], make_transform(config))
        self.assertTrue(torch.equal(validation[0][0], validation[0][0]))
        self.assertEqual(tuple(validation[0][0].shape), (3, 224, 224))
        for normalization in ("baseline", "imagenet"):
            seed_everything()
            first = next(iter(make_loaders(self.root, split, config, normalization, batch_size=4)["train"]))
            seed_everything()
            second = next(iter(make_loaders(self.root, split, config, normalization, batch_size=4)["train"]))
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            self.assertTrue(torch.isfinite(first[0]).all())
        np.testing.assert_allclose(class_weights(split), [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "mask"):
            BUSIDataset(self.root, pd.DataFrame([{"filepath": "normal/normal (0)_mask.png"}]), make_transform(config))

    def test_notebook_format_and_python_cells(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/01_busi_data_eda.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")


if __name__ == "__main__":
    unittest.main()
