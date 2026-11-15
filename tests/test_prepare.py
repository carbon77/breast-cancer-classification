import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from busi import CLASSES
from busi.prepare import audit, eligible_scans, stratify


class AuditTest(unittest.TestCase):
    """Проверки аудита и разбиения на небольшом синтетическом датасете."""

    def test_corruption_masks_and_conflicts(self):
        """Проверить битый PNG, несколько масок, конфликт меток и покрытие выборок."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            for label in CLASSES:
                (root / label).mkdir()
                for i in range(8):
                    image = Image.fromarray(
                        rng.integers(0, 256, (64, 64), dtype=np.uint8)
                    )
                    image.save(root / label / f"{i}.png")
            source = root / "normal/0.png"
            # Та же картинка в другом классе создаёт конфликт меток,
            # а две копии с суффиксом mask проверяют связь нескольких масок.
            for rel in ("benign/copy.png", "normal/0_mask.png", "normal/0_mask_1.png"):
                (root / rel).write_bytes(source.read_bytes())
            (root / "benign/broken.png").write_bytes(b"bad")
            frame, issues = audit(root)
            self.assertEqual(len(frame), 25)
            self.assertIn(
                "0_mask_1.png",
                frame.loc[frame.filepath == "normal/0.png", "masks"].item(),
            )
            eligible = eligible_scans(frame, issues)
            self.assertIn(("invalid_png", "benign/broken.png"), issues)
            self.assertIn(("conflicting_labels", "normal/0.png"), issues)
            self.assertIn(("conflicting_labels", "benign/copy.png"), issues)
            self.assertTrue(any(k == "exact_duplicate" for k, _ in issues))
            split = stratify(eligible)
            # Ни один допустимый снимок не теряется, пути не пересекают
            # выборки, а повторный запуск даёт то же разбиение.
            self.assertEqual(set(split.filepath), set(eligible.filepath))
            self.assertTrue(split.filepath.is_unique)
            self.assertTrue(split.equals(stratify(eligible)))
