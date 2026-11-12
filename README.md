# Классификация опухолей молочной железы по данным УЗИ

## Подготовка BUSI

Из корня репозитория в активном окружении Python выполните `python -m pip install -r requirements.txt`. Для новой копии [BUSI из Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) используйте `python -m busi.prepare --kaggle`. При наличии распакованных данных укажите каталог с папками normal, benign, malignant:

```shell
python -m busi.prepare --root data/busi
```

В Colab после подключения Drive можно передать `--root "/content/drive/MyDrive/colab/data/busi"`. Для анализа другой версии данных задайте новый каталог через `--out`: команда проверяет уже существующий split перед записью. Для проверенных идентификаторов пациентов предусмотрен `--patients patients.csv` с колонками filepath,patient_id. В обработанной копии BUSI таких идентификаторов нет.

## Аудит и EDA

[Манифест](artifacts/manifest.csv) содержит размеры, каналы, хэши снимков и пути всех масок. [Аудит](artifacts/audit.csv) перечисляет проблемные файлы и пары дубликатов. В 780 снимках обнаружены одна пара побайтовых дубликатов и 185 других пар с расстоянием pHash ≤ 4. Все 798 масок сопоставлены со снимками. Декодирование всех PNG прошло успешно.

Откройте [ноутбук](notebooks/01_eda.ipynb) из рабочей копии проекта и выполните все ячейки. Он читает сохранённый манифест и записывает [четыре графика](artifacts/eda). Для нестандартного расположения задайте BUSI_PROJECT (репозиторий) и BUSI_ROOT (датасет).

В выборке 437 benign, 210 malignant и 133 normal. Соотношение самого частого и самого редкого классов — 3,29. В [подборке для проверки качества](artifacts/eda/quality.png) заметны разные пропорции кадров и подпись аппарата в верхнем левом углу malignant (38). Прямой resize до квадрата сильнее всего изменяет геометрию узких кадров, например malignant (136).

## Разбиение и входы моделей

[Split v1](artifacts/splits/split_v1.csv) создан с seed 42. Связные группы pHash целиком относятся к одной выборке. 18 снимков из групп с разными метками вынесены в аудит для ручной проверки. Без идентификаторов пациентов сохраняется риск пересечения пациентов между выборками.

| Класс | Train | Validation | Test |
|---|---:|---:|---:|
| normal | 92 | 20 | 19 |
| benign | 299 | 64 | 64 |
| malignant | 143 | 31 | 30 |
| Всего | 534 | 115 | 113 |

```python
from busi.loading import loaders

batches, preprocessing = loaders("data/busi", "artifacts/splits/split_v1.csv")
images, labels = next(iter(batches["train"]))
```

Для предобученных CNN и ViT передайте соответствующий torchvision enum через `weights=`. В baseline среднее и СКО вычисляются по train. Сохраните возвращённый словарь preprocessing в checkpoint вместе со state_dict: в нём есть порядок классов, RGB, размер, интерполяция, нормализация и SHA-256 split.

## DVC

Git хранит [указатель на BUSI](data/busi.dvc), метаданные и split. DVC хранит исходные файлы и checkpoint. Локальный remote по умолчанию — каталог .dvc-store в корне проекта. Для восстановления в другом checkout предоставьте ему копию хранилища либо задайте путь через `dvc remote modify --local local url /path/to/store`. После выбора нужной версии Git выполните `dvc pull data/busi.dvc`.

Для регистрации обученной модели сохраните checkpoint с preprocessing, затем выполните `dvc add models/model.pt` и `dvc push models/model.pt.dvc`. Включите указатель и созданный DVC файл .gitignore в тот же Git-коммит, что и соответствующий split. Для восстановления используйте `dvc pull models/model.pt.dvc`.

## Проверки

```shell
python -m unittest tests.test_prepare
python -m tests.verify_inputs
python -m tests.verify_dvc
```

Последняя проверка публикует BUSI в локальное хранилище и восстанавливает данные и тестовый checkpoint в отдельном временном рабочем каталоге с пустым кешем.
