"""Cheap lexical pre-filter for the Crypto Cards vertical.

Keeps a message only if it contains at least one domain term (from the
glossary) OR at least one intent signal (hardcoded list).

OR logic is recall-oriented: a message is sent to the downstream LLM if it
shows EITHER a domain term OR a crypto card cue. Precision is delegated to
the LLM — this is the right division of labour for a coarse pre-gate.

Two signal sources:
  1. Domain terms — loaded from ``crypto_cards_glossary_enriched`` (relevant=1)
     or fallback to ``intent_glossary`` in the SQLite DB produced by
     run_crypto_cards_glossary.py + run_crypto_cards_inference.py.
  2. INTENT_SIGNALS — hardcoded RU / EN phrases covering crypto card intent.

Stop words are checked FIRST — they block unconditionally before any signal
check. This prevents P2P exchangers and PSP providers from reaching the LLM.

``pre_filter(text, db_path)`` → True iff the message should proceed to LLM.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = str(
    _PROJECT_ROOT / "Crypto_cards_classifier" / "data" / "crypto_cards_glossary.db"
)

GLOSSARY_CONFIDENCE_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Stop words — checked BEFORE intent signals, block unconditionally
# ---------------------------------------------------------------------------

STOP_WORDS: list[str] = [
    # P2P exchangers offering crypto-to-card withdrawal (not card issuers)
    "покупка usdt через сбп",
    "вывод на карту переводом",
    "вывод на карту от биржи",
    "вывод на карту от официальной биржи",
    "p2p обмен",
    "трейдер p2p",
    "p2p трейдер",
    "обменяю usdt",
    "обменяю btc",

    # Flash / fake crypto scam tools
    "flash usdt",
    "flash btc",
    "flash coins",
    "fake btc",
    "fake usdt",

    # PSP / payment gateway (not cards)
    "payment gateway",
    "платёжный шлюз",
    "платежный шлюз",
    "эквайринг",
    "acquiring solution",

    # Scam / recovery / databases
    "chargeback",
    "recovery leads",
    "database leads",
    "recovery chargeback",
    "returned victims",

    # Job postings
    "ищем менеджера",
    "ищем разработчика",
    "вакансия",
    "job opening",
    "we are hiring",
    "hiring now",
    "open position",
]

# ---------------------------------------------------------------------------
# Intent signals — crypto card buy / sell / seek intent (RU / EN)
# ---------------------------------------------------------------------------

INTENT_SIGNALS: list[str] = [
# Блок 1: Прямое упоминание криптокарты
    r"\bкрипто[\s\-]?карт",
    r"\bкрипто[\s\-]?кошел",
    r"\bкриптокарт",
    r"\bcrypto[\s\-]?card",
    r"\bcrypto[\s\-]?wallet",
    r"\bвиртуальн\w*\s+крипт\w*\s+карт",
    r"\bcard\s+crypto",
    r"\bcrypto.*visa\b",
    r"\bcrypto.*mastercard\b",
    r"\bdebit\s+crypto",
    r"\bcrypto\s+debit",
    r"\bprepaid\s+crypto",
    r"\bcrypto\s+prepaid",

    # Блок 2: Выпуск и эмиссия карт
    r"\bвыпуск\w*\s+карт",
    r"\bэмисси\w*\s+карт",
    r"\bcard[\s\-]?issu",
    r"\bcard[\s\-]?program\b",
    r"\bissuing[\s\-]?bank\b",
    r"\bпрограмм\w*\s+карт",
    r"\bwhite[\s\-]?label\s+card",
    r"\bcard\s+white[\s\-]?label",
    r"\bсубэмисси",
    r"\bsub[\s\-]?issu",
    r"\bэмитент\s+карт",
    r"\bcard\s+emit",

    # Блок 3: Поиск провайдера криптокарт
    r"\bпровайдер\w*\s+крипт",
    r"\bкрипт\w+\s+провайдер",
    r"\bcard\s+provider",
    r"\bcrypto\s+card\s+provider",
    r"\bнужн\w+\s+крипт\w+\s+карт",
    r"\bнужн\w+\s+карт\w+\s+крипт",
    r"\bищ\w+\s+крипт\w+\s+карт",
    r"\bищ\w+\s+провайдер\w+\s+карт",
    r"\blooking\s+for\s+crypto\s+card",
    r"\bneed\w*\s+crypto\s+card",
    r"\bneed\w*\s+card\s+crypto",

    # Блок 4: Конкретные провайдеры
    r"\bbinance\s+card",
    r"\bbybit\s+card",
    r"\bwirex\b",
    r"\bcrypto\.com\s+card",
    r"\bkucoin\s+card",
    r"\bnexo\s+card",
    r"\bcoinbase\s+card",
    r"\bunlimit\b",
    r"\badvance\.cash\b",
    r"\bpaybis\b",
    r"\bbitrefill\b",
    r"\bchange\.org\b",

    # Блок 5: Выплаты через крипто-карту
    r"\bвыплат\w+\s+\w*карт",
    r"\bвыплат\w+\s+крипт",
    r"\bвывод\w*\s+на\s+карт",
    r"\bвывод\w*\s+крипт\w+\s+карт",
    r"\bpayouts?\s+(?:via|through|with|in|on)\s+(?:crypto|card)",
    r"\baffiliate\s+payout",
    r"\bplayer\s+payout",
    r"\bигрок\w+\s+выплат",
    r"\bвыплат\w+\s+игрок",
    r"\boutpay\b",

    # Блок 6: Ресейл и дистрибуция карт
    r"\bресейл\w*\s+карт",
    r"\bresell\s+card",
    r"\bдистрибьют\w+\s+карт",
    r"\bdistribut\w+\s+card",
    r"\bпартнерк\w+\s+карт",
    r"\bкарт\w+\s+партнерк",
    r"\bcard\s+resell",
    r"\bcard\s+distribut",
    r"\bcard\s+reseller",

    # Блок 7: Открытие крипто-аккаунта с картой
    r"\bоткр\w+\s+крипт\w+\s+счет",
    r"\bоткр\w+\s+крипт\w+\s+кошел",
    r"\bopen\s+crypto\s+account",
    r"\bopen\s+crypto\s+wallet",
    r"\bcrypto\s+account\s+open",
    r"\bcrypto\s+wallet\s+card",
    r"\bкошел\w+\s+с\s+карт",

    # Все падежи/числа для "криптокарта"
    "криптокарту",    # вин.ед  (ищу криптокарту)
    "криптокарты",    # род.ед / им.мн  (нет криптокарты / нужны криптокарты)
    "криптокарт",     # род.мн  (поставщика криптокарт)
    "криптокарте",    # дат/пред.ед  (в криптокарте)

    # Крипто-карта (с дефисом)
    "крипто-карту", "крипто-карты", "крипто-карт",

    # Крипто карта (через пробел)
    "крипто карту", "крипто карты", "крипто карт",

    # Без верификации / без KYC — русский вариант
    "без верификации", "без kyc", "без кyc",
    "без лишнего kyc", "без лишней верификации",
    "без проверки", "без идентификации",

    # ── Явные продуктовые термины (уже были, оставляем) ──────────────────────
    "crypto card", "crypto cards",
    "криптокарта", "крипто-карта", "крипто карта",
    "virtual card", "physical card", "prepaid card",
    "виртуальная карта", "физическая карта",
    "card issuing", "card issuer", "card program",
    "выпуск карт", "эмиссия карт",
    "white label card", "white-label card",
    "bin sponsor", "multiple bins",
    "no kyc", "nokyc", "no-kyc",
    "anonymous card", "анонимная карта",

    # ── Поиск эмитента / партнёра (S2 — warm seeker) ─────────────────────────
    "нужен эмитент", "ищем эмитента", "нужен поставщик карт",
    "ищем партнёра по картам", "нужен партнёр по картам",
    "card issuer needed", "looking for card issuer",
    "need card issuer", "need card provider",
    "looking for card provider", "looking for card partner",

    # ── Объёмы и дистрибуция (S2 — есть оборот, нужно решение) ──────────────
    "объёмы по картам", "объёмы на карты",
    "есть объёмы", "большие объёмы",  # только в контексте карт/крипто
    "ресейл карт", "готовы ресейлить карты", "resell cards", "card reseller",
    "дистрибуция карт", "card distribution", "card distributor",

    # ── Дропы (proxy card holders, специфический B2B slang) ──────────────────
    "дропы", "дроп карты", "дроп база", "нужны дропы",
    "drop cards", "card drops",

    # ── Вертикали-потребители криптокарт (косвенный, но сильный сигнал) ──────
    "карты для медиабаинга", "медиабаинг карты",
    "карты для игаминга", "карты igaming", "igaming cards",
    "карты для арбитражников", "affiliate cards",
    "карты под трафик", "карты для фарма",
    "зп карты", "зарплатные карты", "salary cards", "payroll cards",

    # ── Крипто-загрузка карты (S2/S3 — ищут пополнение через крипту) ─────────
    "пополнение с usdt", "пополнение usdt",
    "карта с usdt", "карта под крипту",
    "funding usdt", "funding usdc",
    "funded from wallet", "loaded from wallet",
    "load with usdt", "load with btc",
    "spend crypto", "spend usdt", "card to spend",

    # ── Карточное решение (S2 — бизнес-язык без явного "ищу") ────────────────
    "карточное решение", "card solution",
    "карты для бизнеса под крипту",  # с крипто-маркером
    "корпоративные криптокарты", "corporate crypto cards",

    # ── API / технический провайдер (P3 — слабый сигнал) ─────────────────────
    "api delivery", "api карт", "card api",
    "issuing countries", "bin countries",

    # ── EN — explicit product terms ───────────────────────────────────────────
    "crypto card",
    "crypto cards",
    "reloadable crypto card",
    "reloadable card",
    "virtual card",
    "physical card",
    "prepaid card",
    "card issuing",
    "card issuer",
    "card program",
    "white label card",
    "white-label card",
    "multiple bins",
    "multiple bin",
    "bin sponsor",
    "bin sponsorship",
    "issuing countries",
    "api delivery",
    "no kyc",
    "nokyc",
    "no-kyc",
    "anonymous card",
    "funding usdt",
    "funding usdc",
    "funded from",
    "loaded from wallet",
    "load with usdt",
    "load with btc",
    "spend crypto",
    "spend usdt",
    "card to spend",
    "card loaded",

    # ── EN — seeker signals ───────────────────────────────────────────────────
    "looking for crypto card",
    "looking for card provider",
    "looking for card issuer",
    "need crypto card",
    "need a card",
    "need virtual card",
    "want to resell crypto cards",
    "resell crypto cards",
    "crypto card solution",
    "card solution",

    # ── RU — явные упоминания карты ──────────────────────────────────────────
    "криптокарта",
    "крипто-карта",
    "крипто карта",
    "виртуальная карта",
    "физическая карта",
    "зп карты",
    "зп-карты",
    "зарплатные карты",
    "выпуск карт",
    "эмиссия карт",
    "карта без kyc",
    "карта без кyc",
    "анонимная карта",
    "загрузка с кошелька",
    "пополнение с кошелька",
    "пополнение с usdt",
    "пополнение usdt",
    "карта с usdt",
    "карта под крипту",
    "пластик под крипту",
    "карты для игаминга",
    "карты igaming",
    "nokyc криптокарты",
    "nokyc карты",
    "карта без верификации",
    "без верификации карта",

    # ── RU — seeker signals ───────────────────────────────────────────────────
    "ищу криптокарту",
    "ищу крипто карту",
    "нужна криптокарта",
    "нужны крипто карты",
    "нужны провайдеры крипто карт",
    "ищем поставщика карт",
    "ищу поставщика карт",
    "готовы ресейлить",
    "ресейлить крипто карты",
    "ищу карту под крипту",
    "нужна карта для usdt",
    "нужна карта для крипты",
    "интересует партнёрка по картам",
    "партнерка по картам",
    "поставщик крипто карт",
    "провайдер крипто карт",
]

# ---------------------------------------------------------------------------
# Glossary loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_glossary_terms(db_path: str = DEFAULT_DB) -> frozenset[str]:
    """Load relevant crypto card domain terms from the SQLite DB, lower-cased.

    Prefers ``crypto_cards_glossary_enriched`` (relevant=1, confidence >= threshold).
    Falls back to ``intent_glossary`` if enriched table is absent or empty.
    Cached per db_path so repeated pre_filter() calls don't re-query SQLite.
    """
    terms: set[str] = set()
    path = Path(db_path)
    if not path.exists():
        return frozenset()

    conn = sqlite3.connect(db_path)
    try:
        # ── Source 1: enriched table ──────────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT term, surface_forms FROM crypto_cards_glossary_enriched "
                "WHERE relevant = 1 AND confidence >= ?",
                (GLOSSARY_CONFIDENCE_THRESHOLD,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        for term, surface_json in rows:
            if term:
                terms.add(term.strip().lower())
            try:
                for sf in json.loads(surface_json or "[]"):
                    if sf:
                        terms.add(str(sf).strip().lower())
            except (json.JSONDecodeError, TypeError):
                pass

        if terms:
            terms.discard("")
            return frozenset(terms)

        # ── Source 2: raw intent_glossary fallback ────────────────────────────
        try:
            rows = conn.execute(
                "SELECT term, surface_forms FROM intent_glossary"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        for term, surface_json in rows:
            if term:
                terms.add(term.strip().lower())
            try:
                for sf in json.loads(surface_json or "[]"):
                    if sf:
                        terms.add(str(sf).strip().lower())
            except (json.JSONDecodeError, TypeError):
                pass

    finally:
        conn.close()

    terms.discard("")
    return frozenset(terms)


# ---------------------------------------------------------------------------
# Matching helpers (for debug / explanation)
# ---------------------------------------------------------------------------

def matched_terms(text: str, db_path: str = DEFAULT_DB) -> list[str]:
    """Return the glossary terms found in ``text`` (lower-case substring match)."""
    if not text:
        return []
    low = text.lower()
    return [t for t in load_glossary_terms(db_path) if t in low]


def matched_signals(text: str) -> list[str]:
    """Return the intent signals found in ``text``."""
    if not text:
        return []
    low = text.lower()
    return [s for s in INTENT_SIGNALS if s in low]


def matched_stop_words(text: str) -> list[str]:
    """Return the stop words found in ``text`` (for debugging)."""
    if not text:
        return []
    low = text.lower()
    return [s for s in STOP_WORDS if s in low]


# ---------------------------------------------------------------------------
# Main pre-filter
# ---------------------------------------------------------------------------

def pre_filter(text: str, db_path: str = DEFAULT_DB) -> bool:
    """True iff ``text`` passes the crypto cards pre-filter.

    Logic:
      1. Stop words — block unconditionally (checked first).
      2. Glossary term OR intent signal → pass (OR logic).
    """
    if not text:
        return False
    low = text.lower()

    # Stop words — drop immediately, no further checks
    if any(s in low for s in STOP_WORDS):
        return False

    has_term   = any(t in low for t in load_glossary_terms(db_path))
    has_signal = any(s in low for s in INTENT_SIGNALS)
    return has_term or has_signal


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    terms = load_glossary_terms(db)
    print(f"Glossary terms loaded: {len(terms)} surface strings from {db}")
    print(f"Intent signals: {len(INTENT_SIGNALS)}")
    print(f"Stop words: {len(STOP_WORDS)}")
    print()

    samples = [
        # True positives
        "Reloadable Crypto Cards, loaded from wallet, 100% anonymous, no KYC",
        "LEVEL nokyc криптокарты для медиабаинга, iGaming VIP, зп карты",
        "looking for anonymous crypto card solution with no KYC, pls DM",
        "нужны провайдеры крипто карт. Готовы ресейлить крипто карты",
        "Hey! Virtual cards, Funding: USDT, USDC. Delivery: API. White Label available!",
        # False positives (should be blocked)
        "виртуальная карта, покупка USDT через СБП, вывод на карту от официальной биржи",
        "обменяю USDT на UAH, работаю как P2P трейдер",
        "нужен payment gateway для приёма платежей",
        "flash usdt, работаю с офшорными схемами",
        # Edge cases
        "multiple BINs and issuing countries",
        "virtual card, USDT funding, delivery via API",
    ]

    for msg in samples:
        result = pre_filter(msg, db)
        flag = "✅ PASS" if result else "⛔ DROP"
        sw = matched_stop_words(msg)
        t  = matched_terms(msg, db)
        s  = matched_signals(msg)
        print(f"{flag}  {msg[:70]!r}")
        if sw:
            print(f"       stop_words={sw}")
        else:
            print(f"       terms={t[:3] or '—'}  signals={s[:3] or '—'}")
        print()
