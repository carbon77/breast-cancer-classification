"""Verify DVC recovery in a fresh workspace and cache using a local remote."""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def dvc(folder, *args):
    subprocess.run([sys.executable, "-m", "dvc", *map(str, args)], cwd=folder, check=True)


def hashes(folder):
    return {p.relative_to(folder): hashlib.sha256(p.read_bytes()).digest()
            for p in folder.rglob("*") if p.is_file()}


if __name__ == "__main__":
    project = Path.cwd()
    remote = project / ".dvc-store"
    dvc(project, "push", "-r", "local", "data/busi.dvc")
    with tempfile.TemporaryDirectory() as tmp:
        publisher, consumer = (Path(tmp) / name for name in ("publisher", "consumer"))
        for folder in (publisher, consumer):
            folder.mkdir()
            dvc(folder, "init", "--no-scm")
        dvc(publisher, "remote", "add", "-d", "test", Path(tmp) / "model-store")
        model = publisher / "sample.pt"
        digest = (project / "artifacts/splits/split_v1.sha256").read_text().strip()
        torch.save({"state_dict": torch.nn.Linear(3, 1).state_dict(), "split_sha256": digest}, model)
        dvc(publisher, "add", model.name)
        dvc(publisher, "push")
        shutil.copy(publisher / "sample.pt.dvc", consumer)
        shutil.copy(project / "data/busi.dvc", consumer)
        dvc(consumer, "remote", "add", "data", remote)
        dvc(consumer, "remote", "add", "model", Path(tmp) / "model-store")
        dvc(consumer, "pull", "-r", "data", "busi.dvc")
        dvc(consumer, "pull", "-r", "model", "sample.pt.dvc")
        expected = hashes(project / "data/busi")
        assert expected and expected == hashes(consumer / "busi")
        assert model.read_bytes() == (consumer / model.name).read_bytes()
        assert torch.load(consumer / model.name)["split_sha256"] == digest
        print(f"Verified {len(expected)} dataset files and checkpoint.")
