# БД и персистенция

## Обзор

Проект использует три типа хранилищ:
- **PostgreSQL** — runtime БД для результатов классификации и дедупликации
- **SQLite** — offline-built глоссарии терминов для keyword-фильтров
- **ChromaDB** — RAG-индекс для few-shot примеров (только buyer pipeline)

Полный дамп PostgreSQL-схемы: [pg_schema.sql](pg_schema.sql).

---

## PostgreSQL

### `classified_messages_dirty`

Основная таблица результатов классификации. Используется Buyer, Provider,
Traffic пайплайнами и дедупликатором.

```sql
CREATE TABLE classified_messages_dirty (
    id                        BIGINT NOT NULL DEFAULT nextval('classified_messages_dirty_id_seq'),
    group_id                  BIGINT,
    message_id                BIGINT,
    "timestamp"               TIMESTAMPTZ,
    username                  TEXT,
    text                      TEXT,

    -- Classification result
    is_lead                   BOOLEAN,
    confidence                DOUBLE PRECISION,
    lead_type                 TEXT,           -- 'seeking_psp', 'psp_provider', 'traffic_provider', 'seeking_traffic', 'crypto_buyer', etc.
    intent                    TEXT,           -- 'buying', 'comparing', 'researching', 'complaining', 'other'
    interest_level            TEXT,           -- 'high', 'medium', 'low'

    -- Extracted fields (buyer/provider/traffic)
    vertical                  JSONB,
    geo                       JSONB,
    payment_methods_mentioned JSONB,
    company                   TEXT,
    "position"                TEXT,

    -- Traffic-specific
    traffic_source            JSONB,
    pricing_model             JSONB,

    -- Crypto buyer-specific
    asset                     JSONB,
    network                   JSONB,
    pay_asset                 JSONB,
    pay_method                JSONB,
    amount                    DOUBLE PRECISION,
    min                       DOUBLE PRECISION,
    max                       DOUBLE PRECISION,

    -- Evidence
    evidence_quote            TEXT,
    rationale                 TEXT,
    glossary_terms_seen       JSONB,

    -- Judge stage
    verdict                   TEXT,           -- 'REAL_LEAD', 'MISTAKE', 'REAL_PROVIDER', 'TRAFFIC_PROVIDER', 'TRAFFIC_LEAD', 'REVIEW'
    judge_reason              TEXT,

    -- Pipeline metadata
    pipeline                  TEXT,           -- 'buyer', 'provider', 'traffic', 'dedup'
    elapsed_ms                DOUBLE PRECISION,
    db_write_status           TEXT NOT NULL DEFAULT 'ok',
    db_write_error            TEXT,
    classified_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Manual review (используется внешним проектом доставки)
    manual_approval           TEXT,           -- 'pending', 'True', NULL

    -- Traffic delivery tracking (используется внешним проектом)
    traffic_notified_at       TIMESTAMPTZ,    -- когда строка была отправлена в traffic feed
    traffic_sent_at           TIMESTAMPTZ,    -- когда фактически доставлена

    PRIMARY KEY (id),
    CONSTRAINT uq_group_message_leadtype UNIQUE (group_id, message_id, lead_type)
);
```

**Ключевые индексы:**
```sql
-- Traffic delivery workflow
CREATE INDEX ix_cmd_traffic_pending          ON classified_messages_dirty (verdict, id) WHERE traffic_notified_at IS NULL;
CREATE INDEX ix_cmd_traffic_approved_unsent  ON classified_messages_dirty (verdict, id) WHERE manual_approval = 'True' AND traffic_sent_at IS NULL;
CREATE INDEX ix_cmd_traffic_in_review        ON classified_messages_dirty (verdict) WHERE manual_approval = 'pending';
CREATE INDEX ix_cmd_traffic_sent             ON classified_messages_dirty (verdict, traffic_sent_at) WHERE traffic_sent_at IS NOT NULL;
```

**Триггеры (NOTIFY → внешний проект доставки):**
- `real_lead_on_insert/update` → `notify_real_lead()` — buyer PSP leads (verdict=REAL_LEAD, исключая provider/casino/iban/crypto/card типы)
- `psp_provider_lead_on_insert/update` → `notify_psp_provider_lead()` — provider leads
- `crypto_lead_on_insert/update` → `notify_crypto_lead()` — crypto buyer leads
- `traffic_lead_on_insert/update` → `notify_traffic_lead()` — traffic buyer leads (seeking_traffic)
- `traffic_provider_on_insert/update` → `notify_traffic_provider_lead()` — traffic provider leads

---

### `client_ready_leads`

Финальная таблица «готовых лидов» — подхватывается внешним проектом для
доставки клиенту. Используется Crypto Cards и Cascade (Crypto Buyer, Casino, IBAN).

```sql
CREATE TABLE client_ready_leads (
    id                        BIGINT NOT NULL DEFAULT nextval('client_ready_leads_id_seq'),
    source_lead_id            BIGINT NOT NULL,     -- FK-like → classified_messages_dirty.id
    username                  TEXT,
    text                      TEXT,
    msg_timestamp             TIMESTAMPTZ,
    lead_type                 TEXT,                 -- 'card_seeker', 'card_provider', 'seeking_casino_platform', 'seeking_iban', 'psp_provider'
    vertical                  JSONB,
    geo                       JSONB,
    payment_methods_mentioned JSONB,
    company                   TEXT,
    "position"                TEXT,
    notes                     TEXT,                 -- "[confidence: high | warmth] \"quote\" — Stage4 notes"
    approved_by               TEXT,                 -- автоматический approver ID
    current_provider          TEXT,                 -- IBAN: текущий провайдер клиента
    platform                  JSONB,                -- Casino: запрашиваемая платформа
    traffic_source            JSONB,
    pricing_model             JSONB,
    created_at                TIMESTAMPTZ DEFAULT NOW(),

    -- Delivery tracking (используется внешним проектом)
    psp_notified_at           TIMESTAMPTZ,
    psp_approval_sent_at      TIMESTAMPTZ,
    psp_approved              BOOLEAN,
    psp_approved_by           TEXT,
    psp_decided_at            TIMESTAMPTZ,
    casino_notified_at        TIMESTAMPTZ,
    iban_notified_at          TIMESTAMPTZ,
    card_notified_at          TIMESTAMPTZ,

    PRIMARY KEY (id),
    CONSTRAINT client_ready_leads_source_lead_id_key UNIQUE (source_lead_id)
);
```

**Триггеры (NOTIFY → внешний проект доставки):**
- `psp_lead_on_insert/update` → `notify_psp_lead()` — lead_type='psp_provider'
- `casino_lead_on_insert/update` → `notify_casino_lead()` — lead_type='seeking_casino_platform'
- `iban_lead_on_insert/update` → `notify_iban_lead()` — lead_type='seeking_iban'
- `card_lead_on_insert/update` → `notify_card_lead()` — lead_type IN ('card_seeker', 'card_provider')

---

### `dedup_log`

Дедупликация сообщений (pre-classification). Одна запись на username.

```sql
CREATE TABLE dedup_log (
    username    TEXT NOT NULL PRIMARY KEY,
    text_hash   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

### `messages`

Все сообщения из Telegram-групп, записываемые парсером. Источник данных
для классификатора. Таблица принадлежит парсеру, но доступна из той же
PostgreSQL — может использоваться для подтягивания reply-контекста и
истории пользователя.

```sql
CREATE TABLE messages (
    id              SERIAL PRIMARY KEY,
    message_id      BIGINT,
    export_date     TIMESTAMP,
    reply_to_msg_id BIGINT,                -- ID сообщения, на которое отвечает (reply)
    top_msg_id      BIGINT,                -- ID верхнего сообщения в треде
    text            TEXT,
    date            TIMESTAMP,
    media_path      TEXT,
    media_type      TEXT,
    group_id        BIGINT,
    user_id         BIGINT,
    topic_id        BIGINT,
    is_post         BOOLEAN
);
```

---

### `users`

Telegram-пользователи с био. Таблица парсера, доступна из той же PostgreSQL.
Поле `bio` может использоваться как дополнительный контекст для классификации.

```sql
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    user_id         BIGINT,
    username        TEXT,
    first_name      TEXT,
    last_name       TEXT,
    phone           TEXT,
    message_count   INTEGER,
    bio             TEXT,                   -- Telegram bio пользователя
    bio_fetched_at  TIMESTAMPTZ
);
```

---

### `banned_usernames`

Забаненные юзернеймы. Триггер `banned_usernames_notify` рассылает NOTIFY
при INSERT/DELETE → webhook обновляет in-memory кеш в реальном времени.

```sql
CREATE TABLE banned_usernames (
    username    TEXT NOT NULL PRIMARY KEY,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

### Geo-таблицы

Используются [geo_pipeline.py](../glossary_builder/geo_pipeline.py) для
нормализации и расширения гео-кодов.

```sql
-- Raw value → canonical ISO code. Пополняется LLM-резолвером автоматически.
CREATE TABLE geo_normalization_map (
    raw_value   TEXT NOT NULL PRIMARY KEY,    -- e.g. 'казахстан', 'kazakhstan', 'kz'
    normalized  TEXT NOT NULL,                -- e.g. 'KZ'
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- Region → child codes. Статический справочник.
CREATE TABLE geo_expansion_map (
    parent      TEXT NOT NULL,                -- e.g. 'EU', 'CIS', 'LATAM'
    child       TEXT NOT NULL,                -- e.g. 'DE', 'FR', 'IT'
    created_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (parent, child)
);

-- Лог неразрешённых гео-значений (для ручного review).
CREATE TABLE geo_unknown_values (
    raw_value   TEXT NOT NULL,
    message_id  INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (raw_value, message_id)
);
```

---

### Payment-таблицы

Используются geo_pipeline (Stage 5) для инференса гео из платёжных методов.

```sql
-- Справочник платёжных методов с иерархией и локальной привязкой к гео.
CREATE TABLE payment_options (
    token                TEXT NOT NULL PRIMARY KEY,    -- e.g. 'visa', 'monobank', 'pix'
    display_name         TEXT NOT NULL,
    category             TEXT,                         -- e.g. 'card', 'bank_transfer', 'crypto'
    aliases              TEXT[] NOT NULL DEFAULT '{}',  -- альтернативные написания
    sort_order           INTEGER DEFAULT 0,
    parent_token         TEXT REFERENCES payment_options(token) ON DELETE SET NULL,
    is_local_method      BOOLEAN NOT NULL DEFAULT FALSE,
    local_method_country TEXT[] NOT NULL DEFAULT '{}',  -- e.g. '{UA}' для monobank
    CONSTRAINT payment_options_parent_not_self CHECK (parent_token IS NULL OR parent_token <> token)
);
CREATE INDEX ix_payment_options_local  ON payment_options (is_local_method);
CREATE INDEX ix_payment_options_parent ON payment_options (parent_token);

-- Маппинг payment method → geo code (дополнительный к payment_options.local_method_country).
CREATE TABLE payment_geo_map (
    payment_method TEXT NOT NULL,
    geo_code       TEXT NOT NULL,
    created_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (payment_method, geo_code)
);

-- Лог неизвестных платёжных методов (для ручного review).
CREATE TABLE payment_unknown_values (
    raw_value   TEXT,
    message_id  BIGINT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_payment_unknown UNIQUE (raw_value, message_id)
);
```

---

## Какой пайплайн куда пишет

| Пайплайн | Таблица | pipeline | lead_type | Условие записи |
|----------|---------|----------|-----------|----------------|
| Buyer | `classified_messages_dirty` | `buyer` | `seeking_psp` и др. | Всегда (и лиды, и не-лиды) |
| Provider | `classified_messages_dirty` | `provider` | `psp_provider` | Только `is_provider=True` и `verdict != MISTAKE` |
| Traffic | `classified_messages_dirty` | `traffic` | `traffic_provider` / `seeking_traffic` | Только `verdict ∈ {TRAFFIC_PROVIDER, TRAFFIC_LEAD}` |
| Dedup | `classified_messages_dirty` | `dedup` | `none` | Дубликаты (is_lead=false) |
| Crypto Cards | `client_ready_leads` | — | `card_seeker` / `card_provider` | Только `verdict ∈ {REAL_CARD_SEEKER, REAL_CARD_PROVIDER}` |
| Crypto Buyer | `classified_messages_dirty` | — | `crypto_buyer` | `verdict = REAL_LEAD` |
| Casino | `client_ready_leads` | — | `seeking_casino_platform` | Только при положительном verdict |
| IBAN | `client_ready_leads` | — | `seeking_iban` | Только при положительном verdict |

## Cross-user text deduplication

Все persist-функции (кроме buyer) проверяют, существует ли уже запись
с тем же `lead_type` и тем же текстом (по `MD5(LOWER(TRIM(text)))`).
Если да — помечают как `[multiacc duplicate of id=N]` и не создают новую запись.

Buyer-пайплайн **не выполняет** cross-user dedup — он пишет все результаты,
включая не-лиды (они нужны для каскада).

## NOTIFY: связь с внешним проектом доставки

Все trigger-функции (`notify_real_lead()`, `notify_psp_provider_lead()` и т.д.)
отсутствуют в этом репозитории — они живут в PostgreSQL и вызываются
триггерами при INSERT/UPDATE. Внешний проект доставки слушает эти NOTIFY-каналы
и отправляет лиды клиентам.

---

## Geo pipeline

**Файл:** [glossary_builder/geo_pipeline.py](../glossary_builder/geo_pipeline.py)

Запускается внутри persist-функций (buyer, provider, traffic) после classification,
перед записью в БД. Заменяет `result["geo"]` обработанным списком.

```
raw_geo_list (от LLM extractor)
  → Stage 2: pre_normalize (lowercase, strip)
  → Stage 3: normalize_geo (raw → canonical via geo_normalization_map)
  → Stage 3.5: LLM resolve (для неизвестных, пополняет geo_normalization_map)
  → Stage 4: expand_geo (region → child codes via geo_expansion_map)
  → Stage 5: apply_payment_geo (payment_options.aliases → local_method_country)
  → Stage 6: log_unknown_payment_methods (→ payment_unknown_values)
```

---

## SQLite: глоссарии

### `intent_glossary` (есть в каждой glossary DB)

Базовая таблица терминов, создаётся `run_stage1b.py` и аналогами.

```sql
CREATE TABLE intent_glossary (
    term                TEXT PRIMARY KEY,
    kind                TEXT NOT NULL,       -- 'russian_phrase', 'ukrainian_phrase', 'mixed_phrase', 'phrase'
    language            TEXT,                -- 'ru', 'uk', 'mixed', ''
    frequency           INTEGER NOT NULL,
    surface_forms       TEXT,                -- JSON array
    example_message_ids TEXT,                -- JSON array
    examples            TEXT,                -- JSON array of {message_id, date, username, text}
    source_stage        TEXT NOT NULL,        -- 'stage_1b'
    created_at          TEXT NOT NULL
);
CREATE INDEX idx_intent_frequency ON intent_glossary(frequency);
```

### `provider_glossary_enriched`

Enriched-версия provider-терминов. Создаётся `run_provider_inference.py`.

```sql
CREATE TABLE provider_glossary_enriched (
    term         TEXT PRIMARY KEY,
    relevant     INTEGER NOT NULL,   -- 0 или 1
    senses       TEXT,               -- JSON array of {definition, context_clues, example_message_ids}
    confidence   REAL,               -- 0.0 .. 1.0
    reasoning    TEXT,               -- LLM reasoning
    surface_forms TEXT,              -- JSON array (из intent_glossary)
    frequency    INTEGER,
    kind         TEXT,
    source_stage TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_enriched_relevant   ON provider_glossary_enriched(relevant);
CREATE INDEX idx_enriched_confidence ON provider_glossary_enriched(confidence);
```

### `traffic_glossary_enriched`

Enriched-версия traffic-терминов. Создаётся `run_traffic_inference.py`.

```sql
CREATE TABLE traffic_glossary_enriched (
    term            TEXT PRIMARY KEY,
    surface_forms   TEXT,       -- JSON array
    frequency       INTEGER,
    relevant        INTEGER,    -- 0 или 1 (LLM decision)
    senses          TEXT,       -- JSON array of {definition, context_clues}
    confidence      REAL,       -- 0.0 .. 1.0
    reasoning       TEXT,
    enriched_at     TEXT NOT NULL
);
CREATE INDEX idx_enriched_relevant ON traffic_glossary_enriched(relevant);
```

### `crypto_cards_glossary_enriched`

Enriched-версия crypto cards терминов. Скрипт создания (`run_crypto_cards_inference.py`)
отсутствует в репозитории — таблица создана вручную или из другого источника.
Схема аналогична `traffic_glossary_enriched`.

### `terms` + `senses` (полный glossary store, только buyer offline pipeline)

Создаётся [glossary_builder/storage.py](../glossary_builder/storage.py) при полном
запуске `python run.py build`.

```sql
CREATE TABLE terms (
    term            TEXT PRIMARY KEY,
    relevant        INTEGER NOT NULL,    -- 0 или 1
    confidence      REAL NOT NULL,       -- 0.0 .. 1.0
    source_stage    TEXT NOT NULL,
    reasoning       TEXT,
    critique        TEXT,                -- adversarial validation результат
    critique_severity TEXT,
    surface_forms   TEXT,                -- JSON array
    frequency       INTEGER,
    updated_at      TEXT NOT NULL
);
CREATE INDEX idx_terms_relevant   ON terms(relevant);
CREATE INDEX idx_terms_confidence ON terms(confidence);

CREATE TABLE senses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    term            TEXT NOT NULL REFERENCES terms(term) ON DELETE CASCADE,
    definition      TEXT NOT NULL,
    context_clues   TEXT,                -- JSON array
    example_message_ids TEXT             -- JSON array
);
CREATE INDEX idx_senses_term ON senses(term);
```

---

## ChromaDB: RAG-индекс

**Путь:** `data/rag/chroma`

Используется только buyer-пайплайном для few-shot retrieval. Хранит
размеченные примеры (text + label LEAD/NOT_LEAD). При классификации
извлекает k ближайших примеров каждого класса и включает их в промпт.

Строится через `build_provider_rag.py`.
Embeddings: `perplexity/pplx-embed-v1-0.6b` через OpenRouter.

Если при старте индекс пуст — сервер не запускается (`RuntimeError`).

---

## Env-переменные (БД-related)

```bash
POSTGRES_DSN=postgresql://user:password@host:5432/dbname
TABLE_PREFIX=                # опциональный префикс для таблиц (e.g. "staging_")
OPENAI_API_KEY=sk-...       # для RAG embeddings (ChromaDB)
```
