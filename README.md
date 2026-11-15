# Классификация опухолей молочной железы по данным УЗИ

Локальное окружение включает подготовку BUSI, DVC и ONNX Runtime для CPU. Для обучения и оценки трёх моделей откройте [ноутбук экспериментов](notebooks/02_train_colab.ipynb) в Google Colab с GPU T4. Режим `new` начинает запуск, `resume` продолжает его с Google Drive, `test` оценивает лучший checkpoint.

## Быстрый старт

Понадобятся Git, uv и Python 3.11 или новее. Из терминала выполните:

```sh
git clone https://github.com/carbon77/breast-cancer-classification.git
cd breast-cancer-classification
make install
make test
```

Тесты используют временные данные. Без GNU Make выполните после клонирования `uv sync --locked` и `uv run --locked python -m unittest discover -s tests`.

## Окружение разработчика

Для разработки uv выбирает Python 3.11 по файлу `.python-version`. Установите зависимости из lock-файла и подключите хуки в каждом новом клоне:

```sh
uv sync --locked
uv run --locked pre-commit install
```

При коммите хуки проверяют изменённые Python-файлы. Для проверки всего проекта используйте `uv run --locked pre-commit run --all-files`. Проверки можно запустить отдельно:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
```

## Карта проекта

| Каталог или файл | Назначение |
|---|---|
| [busi/](busi/) | Аудит BUSI, разбиение и загрузчики изображений |
| [data/busi.dvc](data/busi.dvc) | DVC-указатель на исходный датасет |
| [modeling/](modeling/README.md) | Обучение и оценка моделей |
| [serving/](serving/README.md) | API инференса |
| [frontend/](frontend/README.md) | Интерфейс загрузки УЗИ и просмотра результата |
| [configs/](configs/README.md) | Параметры экспериментов и сервиса |
| [notebooks/](notebooks/) | Исследовательский анализ данных |
| [tests/](tests/) | Тесты и проверки входов и DVC |
| [docs/](docs/README.md) | Архитектура, эксперименты и инструкции |

## Подготовка BUSI

Выберите сценарий по источнику данных. Все команды запускаются из корня проекта после установки зависимостей.

| Источник | GNU Make | Эквивалент через uv |
|---|---|---|
| Настроен доступ к DVC | `make pull`, затем `make prepare` | `uv run --locked dvc pull data/busi.dvc`, затем `uv run --locked python -m busi.prepare` |
| Первая загрузка из [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | `make download` | `uv run --locked python -m busi.prepare --kaggle` |
| BUSI уже распакован | `make prepare` для стандартного пути `data/busi` | `uv run --locked python -m busi.prepare --root "<каталог>"` |

Для `--root` укажите родительскую папку каталогов `normal`, `benign` и `malignant`. Загрузка из Kaggle создаёт `data/busi` и сразу запускает подготовку, поэтому каталог назначения должен отсутствовать. Для другой версии данных задайте отдельный каталог результатов через `--out`.

`make clean-preview` показывает содержимое верхнего уровня `data/` и `artifacts/`, которое удалит `make clean`. Очистка сохраняет каталог версий split, DVC-указатели и скрытые служебные файлы. Она удаляет в том числе локальный BUSI, отчёты и графики. Для восстановления данных и отчётов используйте команды `make pull`, `make prepare`. Без Make скрипт запускается через `uv run --no-sync scripts/clean.py`; для предварительного просмотра добавьте `--dry-run`.

## Аудит и EDA

После подготовки `busi.prepare` записывает `artifacts/manifest.csv`. Манифест содержит размеры, каналы, хэши снимков и пути всех масок. Созданный той же командой `artifacts/audit.csv` перечисляет проблемные файлы и пары дубликатов. В 780 снимках обнаружены одна пара побайтовых дубликатов с разными метками. Все 798 масок сопоставлены со снимками. Декодирование всех PNG прошло успешно.

В [инструкции для Colab и Kaggle](docs/colab-kaggle.md) описаны подключение BUSI, запуск EDA в Colab и проверка первого батча.

В выборке 437 benign, 210 malignant и 133 normal. Соотношение самого частого и самого редкого классов — 3,29. В подборке `artifacts/eda/quality.png`, которую сохраняет ячейка проверки качества в [EDA-ноутбуке](notebooks/01_eda.ipynb), заметны разные пропорции кадров и подпись аппарата в верхнем левом углу malignant (38). Прямой resize до квадрата сильнее всего изменяет геометрию узких кадров, например malignant (136).

## Разбиение и входы моделей

Команда подготовки создаёт `artifacts/splits/split_v2.csv` и соседний файл `.sha256`. Split v2 делит снимки каждого класса в пропорции 70/15/15 с seed 42. Два побайтово идентичных файла с разными метками исключены до ручной проверки. Похожие кадры и снимки одной пациентки могут попасть в разные выборки.

| Класс | Train | Validation | Test |
|---|---:|---:|---:|
| normal | 93 | 20 | 20 |
| benign | 305 | 65 | 66 |
| malignant | 146 | 31 | 32 |
| Всего | 544 | 116 | 118 |

В [ноутбуке обучения](notebooks/02_train_colab.ipynb) BUSI скачивается напрямую из Kaggle. Ячейки создают PyTorch DataLoader для каждой выборки. Для baseline среднее и СКО вычисляются по train, для предобученных моделей используются параметры исходных весов. Эти параметры сохраняются в checkpoint.

## DVC

Git хранит [указатель на BUSI](data/busi.dvc) и настройки DVC. DVC хранит исходные файлы BUSI. Разбиение и его контрольная сумма создаются локально при подготовке. Основной DVC remote `storage` указывает на `s3://zakatov-dash-app/dvc` в Yandex Object Storage. В [.dvc/config](.dvc/config) заданы endpoint `https://storage.yandexcloud.net` и регион `ru-central1`.

Установите зависимости командой `uv sync --locked`. Для доступа к бакету задайте `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` в окружении или выберите настроенный профиль через `AWS_PROFILE`. Для восстановления нужен доступ на чтение объектов и просмотр префикса, для публикации — также на запись.

Для локального файла настроек скопируйте [.env.example](.env.example) в `.env`. Оставьте только используемые строки: либо `AWS_PROFILE` с именем профиля, либо пару `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` со значениями. Передайте переменные через окружение оболочки или загрузите файл явно: `uv run --locked --env-file .env dvc pull data/busi.dvc`. Переменная `BUSI_ROOT` указывает путь к распакованным данным для `tests.verify_inputs`; при стандартном пути `data/busi` строку можно удалить. Индивидуальные параметры DVC можно хранить в `.dvc/config.local`.

Команда `make pull` восстанавливает BUSI, а `make push` публикует его в S3. Без Make используйте `uv run --locked dvc pull data/busi.dvc` и `uv run --locked dvc push data/busi.dvc`. Для работы в Colab распакованный датасет можно скопировать из восстановленного каталога `data/busi` в папку Drive, указанную в ноутбуке.

Локальное хранилище по-прежнему доступно через `-r local`, например `uv run dvc pull -r local data/busi.dvc`.

Для регистрации обученной модели сохраните checkpoint с preprocessing, затем выполните `uv run dvc add models/model.pt` и `uv run dvc push models/model.pt.dvc`. Включите указатель и созданный DVC файл .gitignore в тот же Git-коммит, что и метаданные версии split. Для восстановления используйте `uv run dvc pull models/model.pt.dvc`.

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
