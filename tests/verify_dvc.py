"""Проверка восстановления данных и бинарного артефакта через DVC в чистом рабочем каталоге."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def dvc(folder, *args):
    """Запустить DVC текущего Python в folder и прервать проверку при ошибке."""
    subprocess.run(
        [sys.executable, "-m", "dvc", *map(str, args)], cwd=folder, check=True
    )


def hashes(folder):
    """Вернуть SHA-256 каждого файла с путём относительно folder для сравнения копий."""
    return {
        p.relative_to(folder): hashlib.sha256(p.read_bytes()).digest()
        for p in folder.rglob("*")
        if p.is_file()
    }


if __name__ == "__main__":
    project = Path.cwd()
    remote = project / ".dvc-store"
    # Данные публикуются в настроенное локальное хранилище до проверки восстановления.
    dvc(project, "push", "-r", "local", "data/busi.dvc")
    with tempfile.TemporaryDirectory() as tmp:
        # Раздельные проекты исключают восстановление из кеша издателя:
        # потребитель получает только DVC-указатели и доступ к хранилищам.
        publisher, consumer = (Path(tmp) / name for name in ("publisher", "consumer"))
        for folder in (publisher, consumer):
            folder.mkdir()
            dvc(folder, "init", "--no-scm")
        dvc(publisher, "remote", "add", "-d", "test", Path(tmp) / "model-store")
        artifact = publisher / "sample.bin"
        digest = (project / "artifacts/splits/split_v2.sha256").read_text().strip()
        payload = b"BUSI\x00" + bytes.fromhex(digest)
        artifact.write_bytes(payload)
        pointer = artifact.name + ".dvc"
        dvc(publisher, "add", artifact.name)
        dvc(publisher, "push")
        shutil.copy(publisher / pointer, consumer)
        shutil.copy(project / "data/busi.dvc", consumer)
        dvc(consumer, "remote", "add", "data", remote)
        dvc(consumer, "remote", "add", "model", Path(tmp) / "model-store")
        dvc(consumer, "pull", "-r", "data", "busi.dvc")
        dvc(consumer, "pull", "-r", "model", pointer)
        expected = hashes(project / "data/busi")
        # Сверяем содержимое всех файлов, байты артефакта и связь со split.
        assert expected and expected == hashes(consumer / "busi")
        assert (consumer / artifact.name).read_bytes() == payload
        print(f"Verified {len(expected)} dataset files and binary artifact.")
