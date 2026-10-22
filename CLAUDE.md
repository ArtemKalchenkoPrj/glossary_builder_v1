# Glossary Builder v1 — Project Map

## What this project is

FastAPI service that classifies Telegram messages from iGaming/payments groups
into business signals (leads, providers, traffic, crypto cards). Receives JSON
via `POST /classify`, runs parallel classification pipelines, writes results to
PostgreSQL.

**Not in scope:** Telegram parsing (external), client delivery (separate project).

## Detailed documentation

- [docs/01-project-structure.md](docs/01-project-structure.md) — Directory map, runtime vs offline code
- [docs/02-request-lifecycle.md](docs/02-request-lifecycle.md) — Full request path from POST /classify to DB write
- [docs/03-keyword-filters.md](docs/03-keyword-filters.md) — Stage 1: keyword/intent filters per pipeline
- [docs/04-llm-classification.md](docs/04-llm-classification.md) — Stages 2-4: LLM prefilter, extractor, judge
- [docs/05-database-schema.md](docs/05-database-schema.md) — Stage 5: PostgreSQL schema, geo pipeline, persistence
- [docs/06-glossary-building.md](docs/06-glossary-building.md) — Offline tooling: glossary construction pipeline

## Stack

- Python 3.11+, FastAPI, asyncpg (PostgreSQL), SQLite (glossaries), ChromaDB (RAG)
- LLMs via OpenRouter (httpx/openai SDK), default model: `gpt-4.1-mini`
- Entry point: `uvicorn webhook:app --host 0.0.0.0 --port 8000 --reload`

## Request lifecycle (`POST /classify`)

```
Message JSON
  → Deduplication (Deduper/deduplication.py — time + text hash)
  → Banned usernames check (banned_usernames_cache.py)
  → Scam detector (scam_detector.py — keyword + LLM)
  → 4 parallel pipelines (asyncio.gather + to_thread):
      1. Buyer pipeline
      2. Provider pipeline
      3. Traffic pipeline
      4. Crypto Cards pipeline
  → Persist results to PostgreSQL
```

## Pipelines

### 1. Buyer (PSP demand) — `glossary_builder/`

Classifier: `glossary_builder/single_classifier.py`
Keyword filter: `intent_filter.py` (SQLite `glossary.db` + hardcoded intent signals)
Extractor: `glossary_builder/lead_extraction.py` (glossary-injected RAG, few-shot)
Judge: built into `single_classifier.py` (REAL_LEAD / MISTAKE)
Config: `glossary_builder/lead_prompt_versions.py` (v5-01-short), judge v2
DB target: `classified_messages_dirty` (pipeline="buyer")

**Cascade** — if buyer says NOT a lead, runs sequentially:
- Crypto Buyer → `glossary_builder_crypto_buyers/crypto_buyer_classifier.py`
- Casino → `Casino_platforms_classifier/casino_classifier.py`
- IBAN → `IBAN_classifier/iban_lead_classifier.py`
Each writes to `client_ready_leads` if positive.

### 2. Provider (PSP supply) — `PSP_providers_glossary_builder/`

Classifier: `PSP_providers_glossary_builder/provider_classifier.py`
Keyword filter: `intent_filter_psp_providers.py`
Stage1: micro-LLM YES/NO
Extractor: structured fields (company, position, geo, methods, vertical)
Judge: REAL_PROVIDER / MISTAKE
DB target: `classified_messages_dirty` (lead_type="psp_provider")

### 3. Traffic (buyers + sellers) — `Traffic_glossary_builder/`

Classifier: `Traffic_glossary_builder/traffic_classifier.py`
Keyword filter: `intent_filter_traffic.py`
Stage1: micro-LLM YES/NO
Extractor: structured fields + traffic_source, pricing_model
Judge: TRAFFIC_PROVIDER / TRAFFIC_LEAD / MISTAKE
DB target: `classified_messages_dirty` (pipeline="traffic")

### 4. Crypto Cards — `Crypto_cards_glossary_builder/`

Classifier: `Crypto_cards_glossary_builder/crypto_cards_classifier.py`
Keyword filter: `intent_filter_crypto_cards.py` (via `Crypto_cards_glossary_builder/intent_filter_crypto_cards.py`)
Stage1: micro-LLM YES/NO
Stage3: role-play LLM extraction
Stage4: judge (REAL_CARD_SEEKER / REAL_CARD_PROVIDER / MISTAKE)
DB target: `client_ready_leads` (bypasses classified_messages_dirty)

## Classification stages (general pattern)

1. **Keyword filter** — SQLite glossary terms + intent signals, substring match
2. **Stage1 micro-LLM** — binary YES/NO prefilter (not all pipelines have this)
3. **Extractor LLM** — structured field extraction (geo, methods, vertical, etc.)
4. **Judge LLM** — independent verification (REAL_LEAD / MISTAKE)
5. **DB write** — upsert with cross-user text dedup (MD5)

## Post-classification: Geo pipeline

`glossary_builder/geo_pipeline.py` — runs after extraction, before DB write:
normalize → LLM resolve unknowns → expand regions → infer geo from payment methods

## Database

- **PostgreSQL**: `classified_messages_dirty` (main), `client_ready_leads` (crypto cards + cascade), `dedup_log`
- **SQLite**: per-pipeline glossary DBs (`glossary.db`, `provider_glossary.db`, `traffic_glossary.db`, `crypto_cards_glossary.db`)
- **ChromaDB**: RAG index at `data/rag/chroma` (buyer pipeline only)

## Env vars (required)

- `POSTGRES_DSN` — PostgreSQL connection string
- `OPENAI_API_KEY` — for buyer pipeline LLM + RAG embeddings
- `PSP_PROVIDERS_OPENROUTER_API_KEY` — provider pipeline
- `TRAFFIC_OPENROUTER_API_KEY` — traffic pipeline
- `CRYPTO_CARDS_OPENROUTER_API_KEY` — crypto cards pipeline

## Offline tooling (not part of runtime classification)

- `glossary_builder/` and `glossary_builder_crypto_buyers/` — CLI tools for building glossaries from raw chat data (pipeline.py, candidates.py, fusion.py, etc.)
- `run_*.py` scripts — batch processing, glossary building, evaluation
- `build_provider_rag.py` — builds RAG index for provider pipeline
