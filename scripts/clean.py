"""Remove local data and generated artifacts, retaining split versions and DVC pointers."""
import argparse
import shutil
from pathlib import Path


def clean(root, dry_run=False):
    for name in ("artifacts", "data"):
        directory = root / name
        if directory.resolve() != directory:
            raise ValueError(f"Unexpected cleanup path: {directory}")
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or path.suffix == ".dvc" or path == root / "artifacts/splits":
                continue
            if path.resolve() != path:
                raise ValueError(f"Unexpected cleanup path: {path}")
            print(path.relative_to(root))
            if not dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    clean(Path(__file__).resolve().parents[1], parser.parse_args().dry_run)
