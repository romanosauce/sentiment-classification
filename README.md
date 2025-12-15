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
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── configs/
│   ├── model_config.yaml       # Архитектура модели
│   └── train_config.yaml       # Параметры обучения
├── data/                       # Данные
├── logs/                       # Логи обучения
├── models/                     # Сохраненные модели
├── src/
│   ├── data_loader.py          # Загрузка и валидация данных
│   ├── dataset.py              # PyTorch Dataset
│   ├── download_data.py        # Скрипт загрузки данных
│   ├── inference.py            # Инференс для API
│   ├── model.py                # Архитектура нейросети
│   ├── preprocessing.py        # Предобработка текста
│   ├── train.py                # Скрипт обучения
│   ├── utils.py                # Утилиты
│   └── validate.py             # Валидация модели
├── tests/                      # Тесты
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
python src/download_data.py
```

### Обучение модели

```bash
python src/train.py --config configs/train_config.yaml

python src/train.py --config configs/train_config.yaml --verbose

python src/train.py --config configs/train_config.yaml --epochs 5 --lr 0.0005
```

### Валидация модели

```bash
python src/validate.py --model-path models/sentiment_model --config configs/train_config.yaml
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

## Выводы

1. **Выбранный подход**: BiLSTM с предобученными эмбеддингами обеспечивает хороший баланс между качеством и скоростью инференса для задачи бинарной классификации тональности.

2. **Достижимость метрик**: целевые ML-метрики (Accuracy 85%, F1 0.85) достижимы на данном датасете с выбранной архитектурой. Для достижения более высоких показателей рекомендуется переход на трансформерные модели.

3. **Продакшен-готовность**: код организован модульно, покрыт тестами, воспроизводим через фиксацию random_seed и версий библиотек. Модель сохраняется в формате, совместимом с Hugging Face.

4. **Масштабируемость**: архитектура позволяет обрабатывать батчи отзывов, что обеспечивает выполнение SLA по времени отклика (менее 200 мс на запрос).
