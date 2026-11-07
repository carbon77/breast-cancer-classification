# breast-cancer-classification
Классификация опухолей молочной железы по данным УЗИ

## Данные и EDA (UNI-24)

[Ноутбук подготовки BUSI](notebooks/01_busi_data_eda.ipynb) содержит загрузку
версии Kaggle или подключение локального датасета, аудит, EDA, поиск дубликатов,
split 70/15/15 и Dataset/DataLoader для CNN и ViT. Поставляется без результатов
выполнения; обучение моделей — следующий этап.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-data.txt
.venv\Scripts\python -m jupyterlab notebooks/01_busi_data_eda.ipynb
```

В ноутбуке задайте `LOCAL_DATA_ROOT` или оставьте `None` для `kagglehub`.
Версия Kaggle задаётся явно (`KAGGLE_VERSION = 1`). При необходимости выполните
авторизацию штатными средствами Kaggle, не записывая токен в код/outputs.
В Colab notebook умеет клонировать ветку PR; после merge можно указать `master`.
Если зависимости отсутствуют, включите `INSTALL_DEPENDENCIES`, установите их,
перезапустите kernel и верните флаг в `False`. CPU достаточно.

Артефакты сохраняются в `artifacts/data/v1/`, split — в `artifacts/splits/`.
До запуска в Colab измените `ARTIFACTS` на постоянное хранилище или восстановите
его содержимое из предыдущего запуска. Новый временный Colab runtime сам по себе
не сохраняет split. Повторный запуск сверяет SHA-256 данных и split;
при изменениях используйте новую версию и повторно обучите все модели.
Сырые данные, маски, токены и крупные артефакты не коммитятся.

Общая реализация находится в [busi/data.py](busi/data.py), контрактные проверки
на синтетических изображениях запускаются без Kaggle и исполнения notebook:

```powershell
python -m unittest discover -s tests -v
```

Политика split и аугментаций: [план экспериментов](docs/baseline-experiment-plan.md).
Без надёжных patient ID группировка по pHash не исключает patient leakage.
При наличии patient metadata задайте `PATIENT_CSV` с колонками
`filepath,patient_id` и полным покрытием оригиналов. Компоненты со смешанными
метками изолируются до ручной проверки. Notebook сохраняет списки invalid-файлов,
конфликтов и похожих пар, PNG-графики, статистику train, class weights,
preprocessing config и manifest SHA-256.

Версионирование реализовано манифестами, без DVC: каждый raw-файл имеет SHA-256,
split привязан к fingerprint данных, отчёты — к split и code commit. Это проверка
целостности, а не backup: raw dataset нужной версии и весь `artifacts` нужно хранить
на постоянном диске. Для checkpoint будущие эксперименты должны сохранять
SHA-256 весов, split и preprocessing metadata рядом с весами.
