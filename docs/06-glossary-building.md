# Offline: сборка глоссариев

## Назначение

Глоссарии — основа keyword-фильтров (этап 1 classification). Они строятся
offline из реальных данных чатов и хранятся в SQLite. Процесс сборки —
это отдельный pipeline, не связанный с runtime-классификацией.

## Два типа offline tooling

### 1. Полный glossary pipeline (buyer)

CLI-инструмент для сборки полноценного глоссария с определениями, валидацией
и web-research.

**Точка входа:** `python run.py build` → [glossary_builder/cli.py](../glossary_builder/cli.py) → [glossary_builder/pipeline.py](../glossary_builder/pipeline.py)

**Стадии:**

```
Stage 1    — Rule-based candidate extraction (candidates.py)
             Извлекает кандидатов-терминов из корпуса: акронимы, латиница,
             частотный анализ. Фильтрация по hard_skip, min_frequency.

Stage 1b   — LLM-assisted phrase extraction (phrases.py)
             Извлекает многословные фразы и русские/украинские термины
             через LLM. Результат → таблица intent_glossary в SQLite.

Stage 2    — Self-definition mining (self_definitions.py)
             Ищет в корпусе фрагменты, где авторы сами объясняют термины
             («X — это когда...», «X означает...»).

Stage 2b   — Brand mining (brand_mining.py)
             Извлекает названия брендов/компаний из текстов.

Stage 2c   — Gazetteer scan (gazetteer.py)
             Сканирует словарь внешних терминов.

Stage 3    — LLM inference (inference.py)
             Генерирует определения для каждого кандидата через LLM.
             Определяет: relevant? confidence? definition? context clues?

Stage 4    — Adversarial validation (validator.py)
             Второй LLM-проход: критикует определения, ищет ошибки,
             дедуплицирует синонимы.

Stage 5    — Web research (research.py, Tavily)
             Опциональный веб-поиск для верификации терминов.

Stage 6    — Deduplication & cleanup (dedup.py, lemma.py)
             Лемматизация, кластеризация, дедупликация parent-child.
```

**Артефакты:** каждая стадия пишет промежуточный JSON в `data/output/`:
`01_corpus_stats.json`, `02_candidates.json`, ... — pipeline инспектируемый
и возобновляемый.

**Хранилище:** [glossary_builder/storage.py](../glossary_builder/storage.py) — SQLite с таблицами `terms`,
`senses`, `examples`, `critique`.

### 2. Stage 1b + inference (отдельные скрипты)

Для пайплайнов Provider, Traffic и Crypto Cards используется упрощённый
процесс: только Stage 1b (phrase extraction) + Stage 3 (inference/enrichment).

**Этап 1: phrase extraction**

| Пайплайн | Скрипт | Domain data | Выход |
|----------|--------|-------------|-------|
| Buyer | [run_stage1b.py](../run_stage1b.py) | `glossary_builder/domain_data/payments_igaming.py` | `data/output/glossary.db` → `intent_glossary` |
| Crypto Buyers | [run_stage1b_crypto_buyers.py](../run_stage1b_crypto_buyers.py) | `glossary_builder_crypto_buyers/domain_data/crypto_otc_buyers.py` | аналогично |
| Provider | [run_provider_glossary.py](../run_provider_glossary.py) | `payments_igaming_providers.py` | `PSP_providers_glossary_builder/data/provider_glossary.db` |
| Traffic | [run_traffic_glossary.py](../run_traffic_glossary.py) | `traffic_domain_data.py` | `Traffic_glossary_builder/data/traffic_glossary.db` |

Все скрипты:
- Читают .xlsx корпус (Telegram-экспорт)
- Используют `glossary_builder.phrases.extract_phrases()` с domain-specific seed block
- Пишут результат в таблицу `intent_glossary` своей SQLite БД

**Этап 2: LLM enrichment (inference)**

| Пайплайн | Скрипт | Выход |
|----------|--------|-------|
| Provider | [run_provider_inference.py](../run_provider_inference.py) | `provider_glossary_enriched` таблица |
| Traffic | [run_traffic_inference.py](../run_traffic_inference.py) | `traffic_glossary_enriched` таблица |

Enrichment добавляет к каждому термину:
- `relevant: bool` — настоящий доменный термин или шум?
- `senses: JSON` — LLM-определение + context clues
- `confidence: float` — уверенность LLM

Классификатор предпочитает enriched-таблицу (с фильтром `relevant = 1`)
и падает на raw `intent_glossary` если enriched нет.

## RAG-индекс

**Скрипт:** [build_provider_rag.py](../build_provider_rag.py)

Строит ChromaDB-индекс из размеченных примеров (text + label LEAD/NOT_LEAD).
Используется buyer-пайплайном для few-shot retrieval при классификации.

Embeddings: `perplexity/pplx-embed-v1-0.6b` через OpenRouter.

## Domain data файлы

Файлы-справочники с доменными терминами, примерами и seed blocks для phrase
extraction:

| Файл | Назначение |
|------|-----------|
| [glossary_builder/domain_data/payments_igaming.py](../glossary_builder/domain_data/payments_igaming.py) | Buyer: PSP/iGaming домен |
| [glossary_builder_crypto_buyers/domain_data/crypto_otc_buyers.py](../glossary_builder_crypto_buyers/domain_data/crypto_otc_buyers.py) | Crypto buyers: OTC/P2P домен |
| [payments_igaming_providers.py](../payments_igaming_providers.py) | Provider: seller-side PSP домен |
| [traffic_domain_data.py](../traffic_domain_data.py) | Traffic: affiliate/media buying домен |
| [crypto_cards_domain_data.py](../crypto_cards_domain_data.py) | Crypto Cards: крипто-карты домен |

Каждый файл содержит `phrase_seed_block` — примеры терминов для LLM,
чтобы phrase extraction знал, что искать в конкретном домене.

## Дупликация кода

`glossary_builder_crypto_buyers/` — почти полный форк `glossary_builder/`.
Оба содержат одинаковые модули для pipeline-сборки (`candidates.py`, `fusion.py`,
`gazetteer.py`, `lemma.py`, `phrases.py`, `pipeline.py`, `rag.py`, `research.py`,
`self_definitions.py`, `storage.py`, `validator.py`).

Отличаются: `lead_prompt_versions.py`, `domain_data/`, `config.py`,
`lead_extraction.py` (crypto-specific lead definition).

## Утилиты

- [retry_failed_leads.py](../retry_failed_leads.py) — перезапуск сообщений, завершившихся
  с `extraction failed`, из предыдущего batch-run
- `run_crypto_buyers.py` — batch classifier для crypto buyers (offline evaluation)
