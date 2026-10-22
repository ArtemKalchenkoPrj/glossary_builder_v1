# Этап 1: Keyword-фильтры

## Назначение

Keyword-фильтр — первый этап классификации, выполняется локально и бесплатно
(без LLM-вызовов). Цель — быстро отсечь заведомо нерелевантные сообщения
до вызова LLM. Каждый пайплайн имеет свой фильтр с собственным набором
терминов и сигналов.

## Общий принцип

Фильтр проверяет наличие двух типов совпадений в тексте:

1. **Domain terms** — термины из SQLite-глоссария (построенного offline tooling).
   Каждый термин имеет основную форму (`term`) и набор поверхностных форм
   (`surface_forms`). Загружаются один раз и кешируются через `@lru_cache`.

2. **Intent signals** — захардкоженный список фраз-сигналов, указывающих
   на намерение (покупки, продажи, поиска). Мультиязычный: русский, украинский,
   английский.

Матчинг — lowercase substring containment. Намеренно морфологически-толерантный:
«ищу», «ищут», «ищется» все совпадают по подстроке «ищ».

## Логика по пайплайнам

| Пайплайн | Файл | Логика | Anti-signals | SQLite DB |
|----------|------|--------|-------------|-----------|
| Buyer | [intent_filter.py](../intent_filter.py) | **OR** (term OR signal) | Нет | `data/output/glossary.db` → `intent_glossary` |
| Provider | [intent_filter_psp_providers.py](../intent_filter_psp_providers.py) | **AND** (term AND signal) | Да (ANTI_SIGNALS) | `PSP_providers_glossary_builder/data/provider_glossary.db` |
| Traffic | [intent_filter_traffic.py](../intent_filter_traffic.py) | **OR** (term OR signal) | Да (STOP_WORDS) | `Traffic_glossary_builder/data/traffic_glossary.db` |
| Crypto Cards | [intent_filter_crypto_cards.py](../Crypto_cards_glossary_builder/intent_filter_crypto_cards.py) | **OR** (term OR signal) | Да (STOP_WORDS) | `Crypto_cards_classifier/data/crypto_cards_glossary.db` |

### Buyer (intent_filter.py)

```
pre_filter(text) = has_domain_term(text) OR has_intent_signal(text)
```

- OR-логика: recall-ориентированная. Пропускает к LLM любое сообщение с доменным
  термином ИЛИ buyer-intent сигналом.
- Intent signals: ~60 фраз на RU/UK/EN — buyer-сигналы («ищу», «нужен»,
  «looking for», «шукаю»).
- Без anti-signals.
- Глоссарий: таблица `intent_glossary` в `glossary.db`, строится через `run_stage1b.py`.

### Provider (intent_filter_psp_providers.py)

```
pre_filter(text) = NOT has_anti_signal(text) AND has_domain_term(text) AND has_seller_signal(text)
```

- **AND-логика**: precision-ориентированная. Требует ОБОИХ — доменный термин
  и seller-сигнал. Только термин = buyer-обсуждение, только сигнал = нет PSP-контекста.
- Anti-signals (~30): SIM-дроп, кардинг, дропперская терминология, WL software, медиабаинг.
  Проверяются ПЕРВЫМИ — мгновенный reject.
- Seller signals (~80): «работаю по», «покрываю», «процессим», «апрув», «we offer»,
  «approval rate» — сигналы, что автор ПРЕДЛАГАЕТ услуги.
- Дополнительно: `GENERIC_TERMS_EXCLUDE` — термины, которые исключаются из
  glossary-матча как слишком общие («igaming», «gambling», «casino»).

### Traffic (intent_filter_traffic.py)

```
pre_filter(text) = NOT has_stop_word(text) AND (has_domain_term(text) OR has_intent_signal(text))
```

- OR-логика с предварительной проверкой stop words.
- Stop words: P2P-обмен, кредитный дроп, вакансии, мошенничество.
- Intent signals: охватывают обе стороны — продавцы трафика («продаю трафик»,
  «есть трафик», «selling traffic») и покупатели («ищу трафик», «нужен трафик»,
  «buying traffic»).
- Доменные термины загружаются из `traffic_glossary_enriched` (where `relevant=1`),
  с fallback на `intent_glossary`. Есть порог `GLOSSARY_CONFIDENCE_THRESHOLD = 0.70`.

### Crypto Cards (intent_filter_crypto_cards.py)

```
pre_filter(text) = NOT has_stop_word(text) AND (has_domain_term(text) OR has_intent_signal(text))
```

- OR-логика с предварительной проверкой stop words.
- Stop words: P2P-обменники, flash/fake crypto, PSP/gateway, скам/recovery, вакансии.
- Intent signals: **regex-based** (в отличие от всех остальных фильтров!).
  Паттерны вроде `\bкрипто[\s\-]?карт`, `\bcrypto[\s\-]?card`.
- Доменные термины из `crypto_cards_glossary_enriched`, аналогично Traffic.

### Cascade: Crypto Buyer (Crypto_buyers_classifier/)

Отличается от остальных фильтров:

```
filter(text) = has_keyword(text) AND NOT has_anti_keyword(text)
```

- Keywords и anti-keywords загружаются из файлов (`vertical/keywords.txt`,
  `vertical/anti_keywords.txt`), а не из SQLite.
- Каждая строка — regex-паттерн.
- Anti-keywords — **hard block**: если матч найден, сообщение отклоняется
  безусловно, даже если keyword тоже матчит.
- Нет отдельного файла `intent_filter_*.py` — фильтр встроен в
  `crypto_buyer_classifier.py`.

### Cascade: Casino, IBAN

У Casino и IBAN классификаторов keyword-фильтры встроены прямо в
`casino_classifier.py` и `iban_lead_classifier.py` соответственно.
Реализация аналогична: захардкоженные списки ключевых слов, substring match.

## Структура SQLite-глоссариев

Все глоссарии хранятся в SQLite и имеют таблицу `intent_glossary`:

```sql
CREATE TABLE intent_glossary (
    term TEXT,           -- каноническая форма термина
    surface_forms TEXT   -- JSON array дополнительных форм
);
```

Некоторые пайплайны (Traffic, Crypto Cards) используют расширенную таблицу
`*_glossary_enriched` с дополнительными полями (relevance score, confidence)
и фильтрацией по `relevant = 1`.

Глоссарии строятся offline-инструментами (`run_stage1b.py`, `run_provider_glossary.py`,
`run_traffic_glossary.py`) — см. [docs/06-glossary-building.md](06-glossary-building.md).

## Тестирование

Buyer, Traffic и Crypto Cards фильтры имеют встроенный CLI test harness
(`if __name__ == "__main__"`), который прогоняет N случайных сообщений
из .xlsx корпуса и показывает pass/drop с причинами.
