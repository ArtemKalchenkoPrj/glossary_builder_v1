"""Buyer pipeline anti-filter: reject messages with media-buying / traffic
arbitrage / scam terminology ONLY when no PSP-domain term is present.

Double-gate logic:
  1. Message contains a stopword (media buying, traffic tools, scam, etc.)
  2. Message does NOT contain any PSP glossary term

If both conditions are met → block. If a PSP term is present alongside a
stopword, the message passes through (e.g. "фарм-партнёрка ищет процессинг"
has stopword "фарм" but also PSP term "процессинг" → passes).

    from buyer_anti_filter import load_stopwords, anti_filter

    stopwords = load_stopwords("data/buyer_stopwords.json")
    if anti_filter(text, stopwords, glossary_db="data/output/glossary.db"):
        # message blocked
        ...
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DEFAULT_STOPWORDS_PATH = "data/buyer_stopwords.json"
DEFAULT_GLOSSARY_DB = "data/output/glossary.db"


@lru_cache(maxsize=4)
def load_stopwords(path: str = DEFAULT_STOPWORDS_PATH) -> tuple[str, ...]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stopwords file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return tuple(entry["term"].lower() for entry in data["stopwords"])


def anti_filter(
    text: str,
    stopwords: tuple[str, ...] | None = None,
    glossary_terms: frozenset[str] | None = None,
    glossary_db: str | None = None,
) -> bool:
    """Return True if the message should be **blocked**.

    Blocks when: has stopword AND no PSP glossary term.
    Passes when: no stopword, or stopword + PSP term present.
    """
    if not text:
        return False
    if stopwords is None:
        stopwords = load_stopwords()

    low = text.lower()

    hit_stopwords = [sw for sw in stopwords if sw in low]
    if not hit_stopwords:
        return False

    if glossary_terms is None and glossary_db is not None:
        from intent_filter import load_glossary_terms
        glossary_terms = load_glossary_terms(glossary_db)

    if glossary_terms:
        has_psp_term = any(t in low for t in glossary_terms)
        if has_psp_term:
            return False

    return True


def matched_stopwords(text: str, stopwords: tuple[str, ...] | None = None) -> list[str]:
    """Return the stopwords found in text (for debugging / explanation)."""
    if not text:
        return []
    if stopwords is None:
        stopwords = load_stopwords()
    low = text.lower()
    return [sw for sw in stopwords if sw in low]
