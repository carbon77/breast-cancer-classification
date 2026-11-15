# BUSI в Colab и Kaggle

В облачном ноутбуке запускайте код из корня клона. Для подготовки нужны NumPy, pandas и Pillow, для EDA — matplotlib, для загрузчика — совместимая пара PyTorch и torchvision. В Python-ядре с PyTorch проверьте зависимости. При отсутствии пакетов установите их в то же ядро:

```python
%pip install numpy pandas pillow matplotlib torch torchvision
```

## Google Colab

Откройте [EDA-ноутбук](../notebooks/01_eda.ipynb) в Colab и добавьте ячейку перед первой ячейкой кода. Подключите Drive и создайте рабочую копию один раз:

```python
from google.colab import drive
from pathlib import Path
import subprocess

drive.mount("/content/drive")
project = Path("/content/drive/MyDrive/colab/breast-cancer-classification")
project.parent.mkdir(parents=True, exist_ok=True)
if not project.exists():
    subprocess.run(["git", "clone", "https://github.com/carbon77/breast-cancer-classification.git", str(project)], check=True)
root = Path("/content/drive/MyDrive/colab/data/busi")
art = project / "artifacts"
```

Скопируйте распакованный BUSI в `root` так, чтобы непосредственно в нём находились три каталога классов. Выполните общую ячейку подготовки ниже **до** исходных ячеек EDA: первая из них проверяет наличие split. Если выбрали другие пути, исправьте также `project` и `root` в первой исходной ячейке. Затем выполните ячейки EDA. Отчёты и графики в `artifacts/eda/` сохраняются в рабочей копии на Drive (CSV-отчёты лежат в самом `artifacts/`).

## Kaggle Notebooks

Подключите датасет [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) через Add Input. Включите интернет для клонирования репозитория. В ячейке ноутбука выполните:

```python
from pathlib import Path
import subprocess

project = Path("/kaggle/working/breast-cancer-classification")
if not project.exists():
    subprocess.run(["git", "clone", "https://github.com/carbon77/breast-cancer-classification.git", str(project)], check=True)
roots = [p.parent for p in Path("/kaggle/input").rglob("benign")
         if all((p.parent / name).is_dir() for name in ("normal", "benign", "malignant"))]
if len(roots) != 1:
    raise ValueError(f"Проверьте подключённый BUSI и выберите один корень: {roots}")
root = roots[0]
art = Path("/kaggle/working/artifacts")
```

## Подготовка и проверка одного батча

Выполните эту ячейку после настройки путей на выбранной платформе:

```python
import os
import sys

os.chdir(project)
subprocess.run([sys.executable, "-m", "busi.prepare", "--root", str(root),
                "--out", str(art)], check=True)

from busi.loading import loaders

batches, preprocessing = loaders(root, art / "splits/split_v2.csv")
images, labels = next(iter(batches["train"]))
assert images.shape[1:] == (3, 224, 224) and len(images) == len(labels)
print(images.shape, preprocessing["classes"], preprocessing["split_sha256"])
```

На Kaggle сохраните версию ноутбука с выполнением ячеек (Save Version → Save & Run All). После завершения проверьте `artifacts/splits/split_v2.csv` в Output сохранённой версии и скачайте нужные результаты. В Colab проверьте тот же файл на Drive.

## Обучение

Архитектура baseline и метрики описаны в [плане экспериментов](baseline-experiment-plan.md). Точка входа `modeling.train` появится вместе с реализацией обучения. При обучении сохраняйте checkpoint вместе со словарём `preprocessing` и метриками в каталог результатов на Drive или в `/kaggle/working/artifacts` для сохранения в Output.
