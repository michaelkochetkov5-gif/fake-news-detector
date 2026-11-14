# 🔍 Fake News Detector

**Веб-приложение для проверки достоверности новостей и утверждений на русском языке.**

Система ищет источники в интернете, сопоставляет с ними утверждение и выносит один из трёх вердиктов — с указанием, на чём именно основано решение.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-NLI-FFD21E)](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 Что делает

```
Текст пользователя
      │
      ├─► FactChecker ── DuckDuckGo · Wikipedia · RSS ──► источники + цитаты
      │        │
      │        └─ уверен? ──► ДА ──────────────────────► вердикт
      │                        │
      │                        НЕТ
      │                        ▼
      └─► FLUD ── NLI cross-encoder + numeric checker ──► вердикт
                                 │
                                 ▼
                        combine() — итоговое решение
```

| Вердикт | Значение |
|---|---|
| ✅ **ПРАВДА** | Утверждение подтверждено найденными источниками |
| ❌ **ФЕЙК** | Найдено противоречие с источниками |
| ⚠️ **НЕИЗВЕСТНО** | Данных недостаточно для надёжного вывода |

---

## 🧩 Архитектура

Система двухступенчатая. Вторая ступень включается только тогда, когда первая не уверена.

### Ступень 1 — FactChecker (`main_code.py`)

Поиск и первичный анализ:

- **DuckDuckGo** — веб-поиск с несколькими стратегиями запросов
- **Wikipedia API** — поиск статей и загрузка полного текста
- **RSS-ленты** — РИА Новости, ТАСС, Интерфакс, Lenta.ru, BBC Russian
- разбиение сложного текста на отдельные утверждения (`split_into_claims`)
- извлечение ключевых терминов, стран, годов, чисел (`extract_key_terms`, `extract_countries`, `extract_years`)
- проверка числовых отношений и хронологических конфликтов (`check_numeric_relation`, `check_year_conflict`)
- оценка надёжности домена (`get_domain_trust`)
- поиск подтверждающих и опровергающих фрагментов (`find_best_sentences`, `estimate_support`)

### Ступень 2 — FLUD (`nli_checker.py`)

**Fact Lie Utility Detector** — семантическая проверка на CPU.

| Компонент | Назначение |
|---|---|
| `cross-encoder/nli-deberta-v3-base` | NLI-модель: contradiction / entailment / neutral |
| `numeric_checker.py` | сверка чисел между утверждением и источником |

Как принимается решение:

1. Из `details` FactChecker извлекаются цитаты источников — **premise**
2. Каждая premise сравнивается с утверждением через NLI
3. Берётся максимум по `contradiction` и максимум по `entailment` среди всех premise
4. Если `contradiction ≥ entailment` → ФЕЙК, иначе ПРАВДА; уверенность = больший из двух
5. `numeric_checker` может скорректировать вероятность при числовом расхождении
6. Если итоговая уверенность ниже `MIN_CONFIDENCE = 0.50` → НЕИЗВЕСТНО

### Сведение результатов — `combine()`

| Ситуация | Источник решения |
|---|---|
| FactChecker уверен | `factchecker` |
| Обе ступени согласны | `factchecker+flud` |
| Вердикты противоположны | `uncertain` |
| FactChecker не уверен, FLUD уверен | `flud` |
| Обе не уверены | `uncertain` |

Служебные значения: `nli`, `nli_uncertain`, `nli_no_premise`, `nli_unavailable`, `nli_error`.

---

## 📦 Модули

| Файл | Строк | Роль |
|---|---:|---|
| `site.py` | 603 | Streamlit-интерфейс, кэш, вызов пайплайна |
| `main_code.py` | 2605 | FactChecker: поиск, анализ, вердикт |
| `nli_checker.py` | 425 | FLUD: NLI + `combine()` + `is_confident()` |
| `numeric_checker.py` | 356 | Сверка чисел, `check_numbers()`, `adjust_nli_confidence()` |
| `flud_checker.py` | 382 | LoRA-обёртка `FLUDChecker` (см. «Ограничения») |
| `query_extractor.py` | 283 | Извлечение поисковых фраз через pymorphy3 |
| `pages/test_factchecker.py` | 440 | Страница пакетного тестирования |
| `pages/debug_page.py` | 174 | Диагностика: запросы, модели, `.pkl` |

---

## 🖥 Интерфейс

| Страница | Что показывает |
|---|---|
| **Главная** | Ввод текста, вердикт, уверенность, источники, детали проверки, результат FLUD |
| **🧪 Тест фактчекера и FLUD** | Пакетный прогон на списке утверждений, таблица результатов |
| **🔧 Отладка фактчекера** | Поисковые запросы, состояние моделей, наличие `.pkl` |

История проверок хранится в `cache.json` (`CACHE_VERSION = "v8-fix"`).

---

## 📂 Данные

| Файл | Назначение |
|---|---|
| `dataset_binary_checkpoint_868.json` | Размеченный набор: 868 строк, 425 ПРАВДА / 443 ФЕЙК |
| `factcheck_dataset.jsonl` | Набор утверждений для фактчекинга |
| `russian_fake_news_model_lemma_best.pkl` | Стилевая ML-модель (используется только `debug_page.py`) |
| `russian_fake_news_vectorizer_lemma.pkl` | Векторизатор для неё |
| `cache.json` | Кэш поиска и проверок |

---

## ⚙️ Настройки

| Параметр | Где | Значение |
|---|---|---|
| `MIN_CONFIDENCE` | `nli_checker.py` | `0.50` — порог уверенности FLUD |
| `NLI_MODEL_NAME` | `nli_checker.py` | `cross-encoder/nli-deberta-v3-base` |
| `GATE` / `SHRINK` | `numeric_checker.py` | `0.9` / `0.5` — правила коррекции |
| `CACHE_VERSION` | `main_code.py` | `v8-fix` — версия поискового кэша |
| `RSS_FEEDS` | `main_code.py` | 5 новостных лент |

---

## ⚠️ Ограничения

Честно о том, что стоит знать перед использованием:

- **Не оракул.** Система сопоставляет текст с источниками, а не проверяет факты напрямую. Нет источника — нет подтверждения.
- **Уверенность не калибрована.** Softmax NLI-модели систематически переуверен: 100% в интерфейсе не означает нулевую вероятность ошибки.
- **Решает одна premise.** Итог зависит от самого «противоречащего» фрагмента; подтверждающие цитаты на вердикт не влияют.
- **Свежие новости проверяются плохо.** Wikipedia не успевает за событиями дня; RSS покрывает только 5 изданий.
- **Числа прописью не парсятся.** «два», «три» — вне возможностей `numeric_checker`.
- **`flud_checker.py` не подключён.** Модуль `FLUDChecker` (LoRA, путь `lum-factchecker-lora-v2`) в репозитории есть, но `site.py` использует `NLIChecker` из `nli_checker.py`. LoRA-веса в пайплайне не участвуют.

---

## 🗂 Структура

```
fake-news-detector/
├── site.py                      # Streamlit-интерфейс
├── main_code.py                 # FactChecker
├── nli_checker.py               # FLUD: NLI + combine
├── numeric_checker.py           # Сверка чисел
├── flud_checker.py              # LoRA-обёртка (не подключена)
├── query_extractor.py           # Поисковые фразы
├── pages/
│   ├── test_factchecker.py
│   └── debug_page.py
├── requirements.txt
├── cache.json
└── *.pkl                        # Стилевая модель
```

---

## 🛠 Технологии

`Python` · `Streamlit` · `PyMorphy3` · `Transformers` · `PyTorch` · `scikit-learn` · `DuckDuckGo Search` · `Wikipedia API` · `Feedparser` · `BeautifulSoup`

---

## 📄 Лицензия

MIT — см. [LICENSE](LICENSE).

---

<p align="center">
<sub>Сделано для проверки русскоязычных новостей и утверждений</sub>
</p>
