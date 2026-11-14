# Классификация опухолей молочной железы по данным УЗИ

Локальное окружение включает подготовку BUSI, DVC и ONNX Runtime для CPU. EDA и обучение выполняются в Google Colab. Команда предсказания и экспорт появятся вместе с обученной моделью.

## Подготовка BUSI

При установленном GNU Make обычная последовательность команд — `make install`, `make pull`, `make prepare`. Для первой загрузки из Kaggle используйте `make download`: эта цель также выполняет подготовку. Список команд выводит `make help`.

`make clean-preview` показывает содержимое верхнего уровня `data/` и `artifacts/`, которое удалит `make clean`. Очистка сохраняет каталог версий split, DVC-указатели и скрытые служебные файлы. Она удаляет в том числе локальный BUSI, отчёты и графики. Для восстановления данных и отчётов используйте команды `make pull`, `make prepare`. Без Make скрипт запускается через `uv run --no-sync scripts/clean.py`; для предварительного просмотра добавьте `--dry-run`.

Из корня репозитория выполните `uv sync --locked`. Для новой копии [BUSI из Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) используйте `uv run python -m busi.prepare --kaggle`. При наличии распакованных данных укажите каталог с папками normal, benign, malignant:

```shell
uv run python -m busi.prepare --root data/busi
```

В Colab после подключения Drive можно передать `--root "/content/drive/MyDrive/colab/data/busi"`. Для анализа другой версии данных задайте новый каталог через `--out`: команда проверяет уже существующий split перед записью.

## Аудит и EDA

[Манифест](artifacts/manifest.csv) содержит размеры, каналы, хэши снимков и пути всех масок. [Аудит](artifacts/audit.csv) перечисляет проблемные файлы и пары дубликатов. В 780 снимках обнаружены одна пара побайтовых дубликатов с разными метками. Все 798 масок сопоставлены со снимками. Декодирование всех PNG прошло успешно.

### Запуск в новой сессии Google Colab

1. Скопируйте текущую рабочую версию проекта в `My Drive/colab/breast-cancer-classification`, включая `busi/`, `tests/`, ноутбук и `artifacts/splits/split_v2.csv` с файлом контрольной суммы. Распакованный BUSI с каталогами классов поместите в `My Drive/colab/data/busi`.
2. Откройте [EDA-ноутбук](notebooks/01_eda.ipynb) в Colab. При другом расположении исправьте пути `project` и `root` в первой ячейке кода. Выберите «Выполнить все» и разрешите подключение Drive.
3. Ноутбук запускает общий `busi.prepare`, читает манифест и сохраняет четыре графика в `artifacts/eda/` рабочей копии на Drive.
4. Для проверки PyTorch-загрузчиков выполните в дополнительной ячейке после подготовки:

```python
subprocess.run([sys.executable, "-m", "tests.verify_inputs"], cwd=project,
               env={**os.environ, "BUSI_ROOT": str(root)}, check=True)
```

В выборке 437 benign, 210 malignant и 133 normal. Соотношение самого частого и самого редкого классов — 3,29. В [подборке для проверки качества](artifacts/eda/quality.png) заметны разные пропорции кадров и подпись аппарата в верхнем левом углу malignant (38). Прямой resize до квадрата сильнее всего изменяет геометрию узких кадров, например malignant (136).

## Разбиение и входы моделей

[Split v2](artifacts/splits/split_v2.csv) делит снимки каждого класса в пропорции 70/15/15 с seed 42. Два побайтово идентичных файла с разными метками исключены до ручной проверки. Похожие кадры и снимки одной пациентки могут попасть в разные выборки.

| Класс | Train | Validation | Test |
|---|---:|---:|---:|
| normal | 93 | 20 | 20 |
| benign | 305 | 65 | 66 |
| malignant | 146 | 31 | 32 |
| Всего | 544 | 116 | 118 |

В Colab после настройки путей используйте загрузчики для обучения:

```python
from busi.loading import loaders

batches, preprocessing = loaders(root, art / "splits/split_v2.csv")
images, labels = next(iter(batches["train"]))
```

Для предобученных CNN и ViT передайте соответствующий torchvision enum через `weights=`. В baseline среднее и СКО вычисляются по train. Сохраните возвращённый словарь preprocessing в checkpoint вместе со state_dict: в нём есть порядок классов, RGB, размер, интерполяция, нормализация и SHA-256 split.

## DVC

Git хранит [указатель на BUSI](data/busi.dvc), метаданные и split. DVC хранит исходные файлы и checkpoint. Основной DVC remote `storage` указывает на `s3://zakatov-dash-app/dvc` в Yandex Object Storage. В [.dvc/config](.dvc/config) заданы endpoint `https://storage.yandexcloud.net` и регион `ru-central1`.

Установите зависимости командой `uv sync --locked`. Для доступа к бакету задайте `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` в окружении или выберите настроенный профиль через `AWS_PROFILE`. Для восстановления нужен доступ на чтение объектов и просмотр префикса, для публикации — также на запись.

Команда `make pull` восстанавливает BUSI, а `make push` публикует его в S3. Без Make используйте `uv run --locked dvc pull data/busi.dvc` и `uv run --locked dvc push data/busi.dvc`. Для работы в Colab распакованный датасет можно скопировать из восстановленного каталога `data/busi` в папку Drive, указанную в ноутбуке.

Локальное хранилище по-прежнему доступно через `-r local`, например `uv run dvc pull -r local data/busi.dvc`.

Для регистрации обученной модели сохраните checkpoint с preprocessing, затем выполните `uv run dvc add models/model.pt` и `uv run dvc push models/model.pt.dvc`. Включите указатель и созданный DVC файл .gitignore в тот же Git-коммит, что и соответствующий split. Для восстановления используйте `uv run dvc pull models/model.pt.dvc`.

## Самостоятельная локальная проверка

Из корня проекта с восстановленным BUSI выполните:

```shell
uv sync --locked
uv run --locked python -c "import onnxruntime as ort; assert 'CPUExecutionProvider' in ort.get_available_providers(); print(ort.get_available_providers())"
uv run --locked python -m unittest discover -s tests
uv run --locked python -m busi.prepare --root data/busi
uv run --locked python -c "from pathlib import Path; import hashlib; p=Path('artifacts/splits/split_v2.csv'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '5042084b87743cadaabd42319e03f7fbe87fb2976995039d1daf596217aa8ed7'"
uv run --locked python -m tests.verify_dvc
```

Подготовка сравнивает заново вычисленный split с существующим CSV перед записью. Проверка DVC восстанавливает BUSI и бинарный артефакт с контрольной суммой split во временном каталоге и сверяет содержимое.
