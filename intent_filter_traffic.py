"""Cheap lexical pre-filter for the Traffic vertical.

Keeps a message only if it contains at least one domain term (from the
glossary) OR at least one intent signal (hardcoded regex list).

OR logic is recall-oriented: a message is sent to the downstream LLM if it
shows EITHER a domain term OR an affiliate-traffic cue. Precision is
delegated to the LLM — this is the right division of labour for a coarse
pre-gate.

Two signal sources:
  1. Domain terms — loaded from ``traffic_glossary_enriched`` (relevant=1)
     or fallback to ``intent_glossary`` in the SQLite DB produced by
     run_traffic_glossary.py + run_traffic_inference.py.
  2. INTENT_SIGNALS — hardcoded RU / UK / EN phrases (buy/sell intent).

``pre_filter(text, db_path)`` → True iff the message should proceed to LLM.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = str(_PROJECT_ROOT / "Traffic_classifier" / "data" / "traffic_glossary.db")

GLOSSARY_CONFIDENCE_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Intent signals — buy / sell traffic (RU / UK / EN)
# ---------------------------------------------------------------------------

INTENT_SIGNALS: list[str] = [
    # ── Продавець шукає покупця (RU) ─────────────────────────────────────────
    "ищу оффер", "нужен оффер", "ищу партнерку", "ищу партнёрку",
    "ищу рекламодателя", "ищу рекла", "нужна партнерка",
    "под мой трафик", "под мій трафік",
    "ищу оффер под", "ищу партнёрку под",

    # ── Продавець є трафік (RU) ───────────────────────────────────────────────
    "продаю трафик", "продам трафик", "продаю трафiк", "продам трафiк",
    "есть трафик", "є трафiк", "имею трафик",
    "сливаю трафик", "слив трафика", "слить трафик",
    "отливаю трафик", "отлив трафика",
    "гоню трафик", "залью трафик", "заливаю трафик",
    "лью трафик", "льем трафик",

    # ── Продавець є ліди (RU/UK) ──────────────────────────────────────────────
    "продаю лиды", "продам лиды", "продаю ліди",
    "есть лиды", "є ліди",
    "лидген", "лидогенерация", "лідген", "лідогенерація",
    "генерирую лиды", "генерую ліди",

    # ── Покупець шукає трафік (RU/UK) ─────────────────────────────────────────
    "покупаю трафик", "куплю трафик",
    "нужен трафик", "нужны лиды",
    "ищу трафик", "ищем трафик",
    "принимаю трафик", "интересует трафик", "купим трафик",
    "ищу вебмастер", "ищем вебмастер", "нужен вебмастер", "нужны вебмастер",
    "ищу арбитражник", "нужен арбитражник", "ищем арбитражник",
    "работаем с вебмастерами", "сотрудничаем с вебмастерами",

    # ── Роли (самоідентифікація) ──────────────────────────────────────────────
    "арбитражник", "арбитражники", "вебмастер",
    "медиабаер", "медиабайер", "медіабаєр",

    # ── Модели оплаты в торговому контексті ──────────────────────────────────
    "cpa оффер", "cpa offer", "cpa сетк", "cpa network",
    "ревшар", "rev share", "revshare",
    "ftd", "first time deposit",

    # ── EN seller / buyer signals ─────────────────────────────────────────────
    "selling traffic", "have traffic", "traffic for sale",
    "traffic seller", "traffic buyer", "buying traffic",
    "need traffic", "looking for traffic", "seeking traffic",
    "have leads", "selling leads", "lead generation",
    "looking for offer", "seeking offer",
    "looking for affiliate", "seeking advertiser",
    "looking for webmaster", "seeking webmaster",
    "performance marketing", "affiliate network", "affiliate program",
    "media buyer",

    # ── Ніша + трафік (RU) ────────────────────────────────────────────────────
    "гемблинг трафик", "трафик на гемблинг",
    "трафик на казино", "казино трафик",
    "трафик на нутру", "нутра трафик",
    "трафик на крипт", "крипт",
    "трафик на форекс", "форекс трафик",
    "трафик на дейтинг", "дейтинг трафик",
    "трафик на адалт", "адалт трафик",
    "трафик на беттинг", "беттинг трафик",

    # ── Ніша + трафік (EN) ────────────────────────────────────────────────────
    "gambling traffic", "casino traffic",
    "nutra traffic", "crypto traffic",
    "forex traffic", "dating traffic",
    "adult traffic", "betting traffic", "igaming traffic",
]


STOP_WORDS: list[str] = [
    # P2P платёжные схемы
    "страховой депозит",
    "рапира",
    "трансгран",
    "геймбет",
    "гембет",
    "белые треугольники",
    "апелляции: до",
    "rapira net",
    "pdf чеки",
    "пдф чеки",
    "мобком",
    "мобильная коммерция",
    "бт-трафик",
    "bt трафик",
    "метчинг",
    "мэтчинг",
    "pay in",
    "pay out",
    "payin",
    "payout",
    "с2с",
    "c2c",
    "нспк",
    "набор трейдеров",
    "набор тимлидов",
    "площадка для трейдеров",
    "подключаю трейдеров",
    "поставлю трейдеров",
    "flash btc",
    "flash usdt",
    "flash coins",
    "fake btc",
    # Recovery / database scam
    "chargeback",
    "recovery leads",
    "database leads",
    "recovery chargeback",
    # PSP под видом трафика
    "платежное решение",
    "платёжное решение",
]

# ---------------------------------------------------------------------------
# Glossary loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_glossary_terms(db_path: str = DEFAULT_DB) -> frozenset[str]:
    """Load relevant traffic domain terms from the SQLite DB, lower-cased.

    Prefers ``traffic_glossary_enriched`` (relevant=1, confidence >= threshold).
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
                "SELECT term, surface_forms FROM traffic_glossary_enriched "
                "WHERE relevant = 1 AND confidence >= ?",
                (GLOSSARY_CONFIDENCE_THRESHOLD,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        import json
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


# ---------------------------------------------------------------------------
# Main pre-filter
# ---------------------------------------------------------------------------

def pre_filter(text: str, db_path: str = DEFAULT_DB) -> bool:
    """True iff ``text`` passes the traffic pre-filter.

    OR logic: keep if ≥1 glossary domain term OR ≥1 intent signal.
    """
    if not text:
        return False
    low = text.lower()

    # Stop words — drop immediately before any other check
    if any(s in low for s in STOP_WORDS):
        return False

    has_term = any(t in low for t in load_glossary_terms(db_path))
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
    print()

    samples = [
        "есть трафик на gambling, ищу оффер под FB",
        "ищу вебмастеров под крипту, работаем по CPA",
        "платёжный трафик вырос на 20%",
        "DDoS атака на сервер, смотрим трафик",
        "куплю FB аккаунты, есть хорошие фармы",
        "арбитражник, лью на нутру, нужен оффер",
        "ищем media buyer в команду, удалёнка",
        "have gambling traffic, looking for CPA offer",
    ]

    for msg in samples:
        result = pre_filter(msg, db)
        flag = "✅ PASS" if result else "⛔ DROP"
        t = matched_terms(msg, db)
        s = matched_signals(msg)
        print(f"{flag}  text={msg[:60]!r}")
        print(f"       terms={t or '—'}  signals={s[:3] or '—'}")
        print()
