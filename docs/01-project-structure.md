# Структура проекта

## Обзор

Проект — FastAPI-сервис классификации сообщений из Telegram-групп
iGaming/payments тематики. Выделяет бизнес-сигналы (лиды, провайдеры,
трафик, крипто-карты) и записывает результаты в PostgreSQL.

Код делится на две части:
- **Runtime** — webhook-сервер и классификаторы, обрабатывающие сообщения в реальном времени
- **Offline tooling** — CLI-инструменты для сборки глоссариев и batch-обработки

## Дерево директорий

```
glossary_builder_v1/
│
├── webhook.py                          # Главная точка входа: FastAPI app, POST /classify
├── scam_detector.py                    # Pre-classification: скам-детектор (keyword + LLM)
├── banned_usernames_cache.py           # Pre-classification: кеш забаненных юзернеймов
├── requirements.txt                    # Python-зависимости
├── .gitignore
│
├── glossary_builder/                   # Buyer pipeline + offline glossary tooling
│   ├── single_classifier.py            #   Runtime: классификатор одного сообщения (extract → judge)
│   ├── lead_extraction.py              #   Runtime: LeadExtractor, LeadDecision, glossary-injected RAG
│   ├── lead_prompt_versions.py         #   Runtime: версии промптов для lead extraction
│   ├── geo_pipeline.py                 #   Runtime: нормализация/расширение гео после классификации
│   ├── llm.py                          #   Shared: LLM-клиент (Anthropic / OpenAI / OpenRouter)
│   ├── config.py                       #   Shared: конфиги (LLMConfig, PipelineConfig, CandidateConfig)
│   ├── loader.py                       #   Shared: Message dataclass, загрузка данных
│   ├── rag.py                          #   Runtime: ChromaDB RAG-индекс для few-shot
│   ├── fusion.py                       #   Runtime: FusionExtractor (мульти-провайдер extraction)
│   ├── cli.py                          #   Offline: click CLI (build, list, show, export, extract-leads, judge-leads)
│   ├── pipeline.py                     #   Offline: оркестратор полного pipeline сборки глоссария
│   ├── candidates.py                   #   Offline: извлечение кандидатов-терминов из корпуса
│   ├── inference.py                    #   Offline: LLM-генерация определений для терминов
│   ├── research.py                     #   Offline: веб-поиск (Tavily) для верификации
│   ├── validator.py                    #   Offline: adversarial validation определений
│   ├── brand_mining.py                 #   Offline: извлечение брендов из текстов
│   ├── gazetteer.py                    #   Offline: сканирование словаря
│   ├── lemma.py                        #   Offline: лемматизация и кластеризация кандидатов
│   ├── phrases.py                      #   Offline: извлечение фраз
│   ├── dedup.py                        #   Offline: дедупликация терминов
│   ├── self_definitions.py             #   Offline: извлечение самоопределений из корпуса
│   ├── storage.py                      #   Offline/Shared: SQLite GlossaryStore
│   ├── main.py                         #   Offline: альтернативная точка входа
│   ├── webhook.py                      #   (Legacy? дубль основного webhook — не используется)
│   └── domain_data/
│       └── payments_igaming.py         #   Справочник доменных терминов для payments/iGaming
│
├── PSP_providers_glossary_builder/     # Provider pipeline (PSP supply)
│   ├── provider_classifier.py          #   Runtime: classify_provider_message
│   ├── provider_prompt.py              #   Runtime: промпты (Stage1, extractor, judge)
│   ├── provider_decision.py            #   Runtime: ProviderDecision dataclass + config
│   ├── __init__.py
│   └── data/
│       └── provider_glossary.db        #   SQLite: глоссарий провайдерских терминов
│
├── Traffic_glossary_builder/           # Traffic pipeline (buyers + sellers)
│   ├── traffic_classifier.py           #   Runtime: classify_traffic_message
│   ├── traffic_prompt.py               #   Runtime: промпты
│   ├── traffic_decision.py             #   Runtime: TrafficDecision dataclass + config
│   └── data/
│       └── traffic_glossary.db         #   SQLite: глоссарий трафик-терминов
│
├── Crypto_cards_glossary_builder/      # Crypto Cards pipeline
│   ├── crypto_cards_classifier.py      #   Runtime: classify_crypto_cards_message
│   ├── crypto_cards_prompt.py          #   Runtime: промпты (Stage1, Stage3, Stage4)
│   ├── crypto_cards_decision.py        #   Runtime: CryptoCardsDecision dataclass + config
│   ├── intent_filter_crypto_cards.py   #   Runtime: keyword prefilter
│   └── __init__.py
│
├── Casino_platforms_classifier/        # Cascade: casino platform seekers
│   └── casino_classifier.py            #   Runtime: run_casino_classifier
│
├── IBAN_classifier/                    # Cascade: IBAN/banking seekers
│   └── iban_lead_classifier.py         #   Runtime: run_iban_lead_classifier
│
├── Crypto_buyers_classifier/           # Cascade: crypto B2B buyers (keyword-based)
│   ├── crypto_buyer_classifier.py      #   Runtime: run_crypto_buyer_classifier
│   └── vertical/                       #   Конфиги: keywords.txt, anti_keywords.txt, промпты
│
├── glossary_builder_crypto_buyers/     # Cascade: crypto buyers (glossary-based, замена предыдущего)
│   ├── crypto_buyer_classifier.py      #   Runtime: classify_crypto_buyer_single_message
│   ├── lead_extraction.py              #   Форк glossary_builder/lead_extraction.py
│   ├── lead_prompt_versions.py         #   Версии промптов
│   ├── ... (остальные файлы — форк glossary_builder/)
│   └── domain_data/
│       └── crypto_otc_buyers.py        #   Справочник доменных терминов
│
├── Deduper/                            # Дедупликация сообщений
│   ├── deduplication.py                #   Runtime: is_duplicate (time-based + text-hash)
│   └── test_deduplication.py           #   Тесты
│
├── intent_filter.py                    # Buyer keyword prefilter (glossary + intent signals)
├── intent_filter_psp_providers.py      # Provider keyword prefilter
├── intent_filter_traffic.py            # Traffic keyword prefilter
├── intent_filter_crypto_buyers.py      # Crypto buyers keyword prefilter
│
├── payments_igaming_providers.py       # Справочник PSP-провайдеров (domain data)
├── traffic_domain_data.py              # Справочник трафик-терминов (domain data)
├── crypto_cards_domain_data.py         # Справочник крипто-карт терминов (domain data)
│
├── run.py                              # CLI entry: python run.py → glossary_builder.cli
├── run_crypto_buyers.py                # CLI entry: crypto buyers glossary
├── run_stage1b.py                      # Batch: Stage 1b phrase extraction (buyer)
├── run_stage1b_crypto_buyers.py        # Batch: Stage 1b phrase extraction (crypto buyers)
├── run_provider_glossary.py            # Batch: сборка provider glossary
├── run_provider_inference.py           # Batch: inference для providers
├── run_traffic_glossary.py             # Batch: сборка traffic glossary
├── run_traffic_inference.py            # Batch: inference для traffic
├── build_provider_rag.py              # Batch: сборка RAG-индекса для providers
└── retry_failed_leads.py              # Утилита: перезапуск ошибочных сообщений
```

## Границы проекта

**Входит в проект:**
- Приём сообщений через webhook (`POST /classify`)
- Классификация по 7 сигналам (4 параллельных + 3 каскадных)
- Запись результатов в PostgreSQL
- Оффлайн-инструменты для сборки глоссариев

**НЕ входит в проект (внешние сервисы):**
- Парсинг Telegram-групп — отдельный сервис, отправляющий JSON на `/classify`
- Доставка результатов клиенту — отдельный проект, читающий из `client_ready_leads`
- PostgreSQL и ChromaDB — внешние БД, не поднимаются из этого репозитория
- OpenRouter / OpenAI / Anthropic — внешние LLM API

## Дупликация кода

`glossary_builder_crypto_buyers/` — почти полный форк `glossary_builder/`.
Файлы `candidates.py`, `fusion.py`, `gazetteer.py`, `lemma.py`, `phrases.py`,
`pipeline.py`, `rag.py`, `research.py`, `self_definitions.py`, `storage.py`,
`validator.py` продублированы. Отличаются домен-специфичные файлы:
`lead_prompt_versions.py`, `domain_data/`, `config.py`.

Buyer-пайплайн runtime-классификатор (`classify_crypto_buyer_single_message`)
из `glossary_builder_crypto_buyers/` заменил старый из
`Crypto_buyers_classifier/` — оба всё ещё в репозитории, webhook использует
новый.
