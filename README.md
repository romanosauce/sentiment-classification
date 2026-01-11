# IMDB Sentiment Classifier

Проект классификации тональности отзывов на фильмы с использованием нейронных сетей (BiLSTM).

## Бизнес-цель

Автоматизировать анализ 100,000+ отзывов в месяц для улучшения клиентского сервиса.

## Целевые метрики

### Продакшен-метрики (SLA)

| Метрика | Целевое значение |
|---------|------------------|
| Среднее время отклика сервиса | не более 200 мс |
| P99 время отклика | не более 500 мс |
| Доля неуспешных запросов | не более 1% |
| Использование памяти | не более 2 GB |
| Использование CPU | не более 80% |

### ML-метрики

| Метрика | Целевое значение |
|---------|------------------|
| Accuracy | не менее 85% |
| F1-score | не менее 0.85 |
| ROC-AUC | не менее 0.90 |

### Бизнес-метрики

| Метрика | Целевое значение |
|---------|------------------|
| Время обработки 100K отзывов | не более 6 часов |
| Снижение нагрузки на операторов | не менее 70% |

## Набор данных

**IMDB Movie Reviews Dataset**

- Источник: Stanford AI Lab (https://ai.stanford.edu/~amaas/data/sentiment/)
- Размер: 50,000 отзывов (25K train / 25K test)
- Баланс классов: 50% положительных / 50% отрицательных
- Средняя длина отзыва: около 230 слов

Структура данных:
```
data/
├── train/
│   ├── pos/  # 12,500 положительных отзывов
│   └── neg/  # 12,500 отрицательных отзывов
└── test/
    ├── pos/  # 12,500 положительных отзывов
    └── neg/  # 12,500 отрицательных отзывов
```

## Версионирование данных (DVC)

Проект использует [DVC](https://dvc.org/) для версионирования данных и моделей.

### Расположение данных и моделей

| Артефакт | Локальный путь | DVC файл | Описание |
|----------|---------------|----------|----------|
| Сырые данные | `data/` | `data.dvc` | IMDB датасет (50K отзывов, ~65MB) |
| Подготовленные данные | `processed/` | `dvc.lock` | Токенизированные данные и словарь |
| Обученная модель | `models/sentiment_model/` | `dvc.lock` | Веса BiLSTM модели |
| Метрики | `metrics.json` | `dvc.lock` | Результаты валидации |

### Быстрый старт с DVC

```bash
# Клонирование и восстановление данных
git clone https://github.com/your-username/sentiment-classifier.git
cd sentiment-classifier
pip install dvc
dvc pull

# Воспроизведение пайплайна
dvc repro
```

### DVC Pipeline

Пайплайн состоит из трёх стадий:

```
┌──────────┐     ┌─────────┐     ┌──────────┐
│ prepare  │ ──▶ │  train  │ ──▶ │ evaluate │
└──────────┘     └─────────┘     └──────────┘
```

1. **prepare** — загрузка и предобработка данных, построение словаря
2. **train** — обучение BiLSTM модели
3. **evaluate** — валидация на тестовой выборке, расчёт метрик

Команды для работы с пайплайном:
```bash
# Запуск всего пайплайна
dvc repro

# Запуск отдельной стадии
dvc repro prepare
dvc repro train
dvc repro evaluate

# Просмотр статуса
dvc status

# Просмотр метрик
dvc metrics show

# Визуализация пайплайна
dvc dag
```

### Переключение версий

```bash
# Переключение на определённую версию данных/модели
git checkout <commit-hash>
dvc checkout

# Возврат к текущей версии
git checkout main
dvc checkout
```

## Трекинг экспериментов (MLflow)

Проект использует [MLflow](https://mlflow.org/) для трекинга экспериментов.

### Просмотр результатов

```bash
# Запуск MLflow UI (локально)
mlflow ui --port 5000

# Открыть в браузере: http://localhost:5000
```

### Что логируется

| Категория | Данные |
|-----------|--------|
| **Параметры** | epochs, batch_size, learning_rate, hidden_dim, dropout, и др. |
| **Метрики** | train_loss, val_loss, accuracy, f1, roc_auc (по эпохам) |
| **Артефакты** | model/, config/, dvc.lock, data.dvc, logs/ |
| **Теги** | dvc_data_hash, config_file |

### Запуск обучения с MLflow

```bash
# Стандартный запуск (логи в mlruns/)
python -m src.train --config configs/train_config.yaml

# С указанием имени эксперимента
python -m src.train --config configs/train_config.yaml \
    --experiment-name "my-experiment" \
    --run-name "baseline-v1"

# С удалённым сервером MLflow
python -m src.train --config configs/train_config.yaml \
    --mlflow-tracking-uri http://mlflow-server:5000
```

### Интеграция DVC + MLflow

При каждом запуске автоматически:
- Сохраняется `dvc.lock` как артефакт (воспроизводимость пайплайна)
- Сохраняется `data.dvc` как артефакт (версия данных)
- Логируется `dvc_data_hash` как тег (быстрый поиск по версии данных)

### Настройка удалённого сервера (опционально)

```bash
# Запуск MLflow сервера с PostgreSQL и S3
MLFLOW_SERVER_ALLOWED_HOSTS="*" mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlruns
```

### Структура mlruns/

```
mlruns/
├── 0/                          # Default experiment
├── <experiment_id>/
│   ├── <run_id>/
│   │   ├── artifacts/
│   │   │   ├── model/          # Сохранённая модель
│   │   │   ├── config/         # Конфигурация
│   │   │   ├── dvc/            # dvc.lock, data.dvc
│   │   │   └── logs/           # Логи обучения
│   │   ├── metrics/            # Метрики по эпохам
│   │   ├── params/             # Гиперпараметры
│   │   └── tags/               # Теги (DVC хеши)
│   └── meta.yaml
└── models/                     # Model Registry (если используется)
```

## План экспериментов

**Этап 1: Baseline (текущий)**
- Архитектура: BiLSTM с Embedding слоем
- Токенизация: словарный токенизатор
- Ожидаемый результат: Accuracy около 85%

**Этап 2: Улучшение архитектуры**
- Добавление Attention механизма
- Эксперименты с размером embedding
- Регуляризация и dropout

## Структура проекта

```
sentiment-classifier/
├── .dvc/                       # DVC конфигурация
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── configs/
│   ├── model_config.yaml       # Архитектура модели
│   └── train_config.yaml       # Параметры обучения
├── data/                       # Сырые данные (DVC tracked)
├── processed/                  # Подготовленные данные (DVC tracked)
├── models/                     # Сохраненные модели (DVC tracked)
├── mlruns/                     # MLflow эксперименты
├── logs/                       # Логи обучения
├── src/
│   ├── data_loader.py          # Загрузка и валидация данных
│   ├── dataset.py              # PyTorch Dataset
│   ├── download_data.py        # Скрипт загрузки данных
│   ├── evaluate.py             # Оценка модели (DVC stage)
│   ├── inference.py            # Инференс для API
│   ├── model.py                # Архитектура нейросети
│   ├── prepare.py              # Подготовка данных (DVC stage)
│   ├── preprocessing.py        # Предобработка текста
│   ├── train.py                # Обучение модели (DVC stage)
│   ├── utils.py                # Утилиты
│   └── validate.py             # Валидация модели
├── tests/                      # Тесты
├── data.dvc                    # DVC файл для данных
├── dvc.yaml                    # DVC пайплайн
├── dvc.lock                    # DVC lock файл
├── metrics.json                # Метрики модели
├── requirements.txt            # Зависимости
└── README.md
```

## Установка и запуск

### Требования

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (опционально, для GPU)

### Установка

```bash
git clone https://github.com/your-username/sentiment-classifier.git
cd sentiment-classifier

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Загрузка данных

```bash
python -m src.download_data
```

### Обучение модели

```bash
python -m src.train --config configs/train_config.yaml

python -m src.train --config configs/train_config.yaml --verbose

python -m src.train --config configs/train_config.yaml --epochs 5 --lr 0.0005
```

### Валидация модели

```bash
python -m src.validate --model-path models/sentiment_model --config configs/train_config.yaml
```

### Использование в коде

```python
from src.inference import SentimentPredictor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = SentimentPredictor.from_pretrained("models/sentiment_model", device)

result = predictor.predict("This movie was absolutely fantastic!")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence:.2%}")
```

### Запуск тестов

```bash
pytest tests/ -v

pytest tests/ -v --cov=src
```

## Конфигурация

Основные параметры в `configs/train_config.yaml`:

```yaml
data:
  max_vocab_size: 20000
  max_seq_length: 256

training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001
  random_seed: 42

model:
  embedding_dim: 128
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: true
```

## Тестирование

Проект содержит тесты для всех ключевых компонентов:

- `test_preprocessing.py` — очистка текста, токенизация, построение словаря
- `test_data_loader.py` — загрузка данных, валидация формата, разбиение на выборки
- `test_model.py` — инициализация модели, forward pass, сохранение и загрузка
- `test_inference.py` — преобразование логитов в предсказания, валидация входных данных

При каждом коммите автоматически запускаются:
- Все unit-тесты (pytest)
- Линтер (ruff)
- Проверка форматирования (black)

## Docker

Проект можно запустить в Docker-контейнере для batch-инференса.

### Сборка образа

```bash
docker build -t ml-app:v1 .
```

Размер образа: ~800 MB (используется CPU-only версия PyTorch).

### Запуск контейнера

```bash
# Базовый запуск
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
    ml-app:v1 --input_path /app/input/data.csv --output_path /app/output/preds.csv

# С указанием batch size
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
    ml-app:v1 --input_path /app/input/data.csv --output_path /app/output/preds.csv --batch_size 64
```

### Формат входных данных

Входной файл должен быть в формате CSV с колонкой `text`:

```csv
text
This movie was absolutely fantastic!
Terrible waste of time, boring and predictable.
An average film, nothing special.
```

Можно указать другое имя колонки через `--text_column`:

```bash
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
    ml-app:v1 --input_path /app/input/data.csv --output_path /app/output/preds.csv --text_column review
```

### Формат выходных данных

Выходной CSV содержит исходные данные плюс колонки с предсказаниями:

| text | prediction | confidence | prob_positive | prob_negative |
|------|------------|------------|---------------|---------------|
| This movie was fantastic! | positive | 0.9234 | 0.9234 | 0.0766 |
| Terrible waste of time... | negative | 0.8912 | 0.1088 | 0.8912 |

### Пример полного workflow

```bash
# 1. Подготовка данных
mkdir -p input output
echo "text" > input/data.csv
echo "This movie was great!" >> input/data.csv
echo "Worst film I have ever seen." >> input/data.csv

# 2. Сборка образа
docker build -t ml-app:v1 .

# 3. Запуск предсказаний
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
    ml-app:v1 --input_path /app/input/data.csv --output_path /app/output/preds.csv

# 4. Проверка результатов
cat output/preds.csv
```

### Скрипт src/predict.py

Скрипт выполняет batch-инференс:

1. Загружает обученную модель с диска (`models/sentiment_model/`)
2. Читает входной CSV файл с текстами
3. Выполняет предсказания для каждого текста
4. Сохраняет результаты в выходной CSV

Аргументы командной строки:

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `--input_path` | Путь к входному CSV | (обязательный) |
| `--output_path` | Путь к выходному CSV | (обязательный) |
| `--model_path` | Путь к модели | `models/sentiment_model` |
| `--batch_size` | Размер батча | 32 |
| `--text_column` | Имя колонки с текстом | `text` |

## TorchServe (Online API)

Модель можно развернуть как REST API сервис с помощью TorchServe.

### Структура serve/

```
serve/
├── Dockerfile           # Docker образ для TorchServe
├── config.properties    # Конфигурация TorchServe
├── handler.py           # Кастомный обработчик запросов
├── export_model.py      # Экспорт модели в TorchScript
├── build_mar.sh         # Скрипт сборки .mar архива
├── sample_input.json    # Пример входных данных
└── model-store/
    └── sentiment.mar    # Архив модели для TorchServe
```

### Сборка .mar архива

```bash
# Установка зависимостей
pip install torch-model-archiver torchserve

# Сборка архива (экспорт + архивация)
./serve/build_mar.sh
```

### Сборка Docker образа

```bash
docker build -t sentiment-serve:v1 -f serve/Dockerfile .
```

### Запуск контейнера

```bash
# Запуск в фоновом режиме
docker run -d -p 8080:8080 -p 8081:8081 -p 8082:8082 \
    --name sentiment-server sentiment-serve:v1

# Проверка логов
docker logs sentiment-server
```

### REST API

#### Inference API (порт 8080)

**POST /predictions/sentiment** — получить предсказание

```bash
# Запрос
curl -X POST http://localhost:8080/predictions/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'

# Ответ
{
  "prediction": "positive",
  "confidence": 0.9167,
  "probabilities": {
    "negative": 0.0833,
    "positive": 0.9167
  }
}
```

#### Management API (порт 8081)

```bash
# Список моделей
curl http://localhost:8081/models

# Статус модели
curl http://localhost:8081/models/sentiment

# Масштабирование (увеличить workers)
curl -X PUT "http://localhost:8081/models/sentiment?min_worker=2"
```

#### Metrics API (порт 8082)

```bash
# Prometheus метрики
curl http://localhost:8082/metrics
```

### Формат входных данных

JSON с полем `text`:

```json
{
  "text": "Your review text here"
}
```

### Формат ответа

```json
{
  "prediction": "positive|negative",
  "confidence": 0.0-1.0,
  "probabilities": {
    "positive": 0.0-1.0,
    "negative": 0.0-1.0
  }
}
```

### Конфигурация TorchServe

Файл `serve/config.properties`:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `inference_address` | `http://0.0.0.0:8080` | Адрес Inference API |
| `management_address` | `http://0.0.0.0:8081` | Адрес Management API |
| `metrics_address` | `http://0.0.0.0:8082` | Адрес Metrics API |
| `default_workers_per_model` | `1` | Число workers на модель |
| `default_response_timeout` | `120` | Таймаут ответа (сек) |
| `disable_token_authorization` | `true` | Отключить авторизацию |

### Кастомный handler.py

Handler выполняет:
1. **Preprocess**: очистка текста, токенизация, padding
2. **Inference**: forward pass через TorchScript модель
3. **Postprocess**: softmax, формирование JSON ответа

## Выводы

1. **Выбранный подход**: BiLSTM с предобученными эмбеддингами обеспечивает хороший баланс между качеством и скоростью инференса для задачи бинарной классификации тональности.

2. **Достижимость метрик**: целевые ML-метрики (Accuracy 85%, F1 0.85) достижимы на данном датасете с выбранной архитектурой. Для достижения более высоких показателей рекомендуется переход на трансформерные модели.

3. **Продакшен-готовность**: код организован модульно, покрыт тестами, воспроизводим через фиксацию random_seed и версий библиотек. Модель сохраняется в формате, совместимом с Hugging Face.

4. **Масштабируемость**: архитектура позволяет обрабатывать батчи отзывов, что обеспечивает выполнение SLA по времени отклика (менее 200 мс на запрос).
