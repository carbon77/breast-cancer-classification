# Архитектура системы

## Решение

Обучение и serving разделены. Обучающий код выполняется в GPU-среде (Google Colab или Kaggle Notebook) и не запускается при веб-запросе. Веб-прототип состоит из Streamlit UI и отдельного FastAPI backend: Streamlit не загружает модель и не вызывает inference-модуль напрямую.

```text
Пользователь
    |
    v
Streamlit UI
(загрузка, preview, результат, disclaimer)
    |
    | HTTP multipart/form-data
    v
FastAPI backend
(/health, /model-info, /predict)
    |
    v
Inference service
(валидация -> preprocessing -> predict -> форматирование)
    |
    v
Загруженный checkpoint выбранной модели
```

При запуске backend загружает checkpoint и preprocessing-конфигурацию один раз, переводит модель в evaluation/inference режим и при PyTorch-инференсе отключает вычисление градиентов. Загруженные изображения не сохраняются без необходимости.

## Логические компоненты

| Каталог | Ответственность |
| --- | --- |
| `data/` | Локальная структура и инструкции получения датасета; крупные данные не хранятся в Git. |
| `src/data/` | Валидация датасета, split, загрузчики. |
| `src/preprocessing/` | Общие преобразования и аугментации. |
| `src/models/` | Архитектуры и фабрика загрузки модели. |
| `src/training/` | Обучение, validation, сохранение checkpoint. |
| `src/evaluation/` | Метрики, confusion matrix, сравнение экспериментов. |
| `src/inference/` | Загрузка checkpoint, preprocessing, предсказание. |
| `app/backend/` | FastAPI endpoints, схемы и обработка ошибок. |
| `app/frontend/` | Streamlit UI, обращающийся к backend по HTTP. |
| `artifacts/` | Конфигурации, метрики и веса; крупные файлы могут храниться вне Git. |
| `tests/` | Тесты preprocessing, контракта предсказания и ошибок загрузки. |

Это целевая логическая структура, а не утверждение, что все каталоги уже существуют.

## HTTP API

| Endpoint | Назначение | Успешный ответ |
| --- | --- | --- |
| `GET /health` | Проверка готовности сервиса. | Состояние сервиса. |
| `GET /model-info` | Получение имени/версии модели и поддерживаемых классов. | Метаданные модели. |
| `POST /predict` | Принимает одно изображение через `multipart/form-data` и запускает инференс. | JSON с `predicted_class`, `probabilities`, `model_version`. |

Backend ограничивает допустимый формат и размер файла, проверяет декодирование и возвращает подходящий HTTP status code с понятным сообщением. Ответ не должен содержать traceback или внутренние пути. CORS настраивается только если он действительно необходим.
