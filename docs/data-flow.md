# Data flow

## Конвейер данных

```text
Kaggle Breast Ultrasound Images Dataset
                |
                v
       Data validation и EDA
 (структура, метки, размеры, файлы,
  дубликаты и дисбаланс классов)
                |
                v
              Data split
       (train / validation / test)
                |
       +--------+---------+
       |                  |
       v                  v
 Training transforms   Validation/test transforms
 (augmentation)        (без случайных аугментаций)
       |                  |
       +--------+---------+
                |
                v
      Обучение и сравнение моделей
 (CNN / transfer-learning CNN / ViT)
                |
                v
      Оценка и выбор лучшей модели
                |
                v
 Checkpoint + preprocessing config + metrics
                |
                v
 FastAPI: validate -> preprocess -> predict
                |
                v
 Streamlit: класс + оценки + версия + disclaimer
```

## Правила разделения

- Сначала создаются train, validation и test; аугментации применяются только после этого и только к train.
- Validation и test не используются для оптимизации весов.
- При наличии надёжного идентификатора пациентки или исследования split выполняется на его уровне, чтобы снимки одного случая не оказались в разных подвыборках.
- Если такую связь восстановить нельзя, это документируется как риск возможной утечки и завышенных метрик.
- Фиксируются random seed и состав/индексы split, чтобы эксперимент можно было повторить.
- Конкретные пропорции (70 / 15 / 15), защита от почти одинаковых снимков (pHash-группы), seed и версионирование `split_v1` описаны в [baseline-experiment-plan.md](baseline-experiment-plan.md#1-единый-trainvalidationtest-split).

## Preprocessing

Общий pipeline: `decode -> resize -> преобразование каналов -> tensor -> normalize`. Его параметры зависят от выбранной архитектуры и сохраняются рядом с checkpoint. Для ImageNet-предобученных моделей используются соответствующие преобразование каналов и нормализация. Validation, test и инференс применяют одну и ту же детерминированную конфигурацию.

## Артефакты эксперимента

Для каждого эксперимента сохраняются архитектура и параметры, seed и split, размер изображения, preprocessing, аугментации, loss, optimizer, learning rate, batch size, число эпох, метрики по эпохам и на test, время обучения, latency, размер весов, версии зависимостей и идентификатор checkpoint.
