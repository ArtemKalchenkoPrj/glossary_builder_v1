"""Cheap pre-filter: keep a message only if it mentions BOTH a domain term
AND an intent signal.

Two sources are combined:

  1. Domain terms — the ``intent_glossary`` table built by ``run_stage1b.py``
     (Stage 1b phrase extraction). Each row's ``term`` + ``surface_forms``
     contribute surface strings to match.
  2. Intent signals — a hardcoded, multilingual list of buyer-intent cues
     (``ищу``, ``нужен``, ``у кого есть``, ``looking for``, ``шукаю`` …).

``pre_filter(text)`` returns True iff the text contains at least one domain
term AND at least one intent signal. Matching is lower-cased substring
containment — deliberately morphology-tolerant (Russian/Ukrainian inflection)
to match the convention already used in ``phrases.py``. This is a coarse,
zero-cost gate meant to run BEFORE any LLM lead-extraction call, not a
classifier on its own.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path


DEFAULT_DB = "data/output/glossary.db"


# Hardcoded buyer-intent signals. Lower-case; matched as substrings so
# inflected/attached forms ("ищу", "ищут", "ищется") are caught. Multi-word
# signals are matched as-is. Russian / Ukrainian / English.
INTENT_SIGNALS: list[str] = [
    # --- Russian ---
    "ищу", "ищем", "ищет", "ищите", "поиске", "в поиске", "в поисках",
    "нужен", "нужна", "нужно", "нужны", "надо",
    "требуется", "требуются", "подскажите", "посоветуйте",
    "у кого есть", "кто может", "кто даст", "кто подскажет",
    "кто работает с", "ищется", "интересует", "интересуют",
    "хочу подключить", "ищу провайдера", "+1 в поиске",
    # --- Russian near-miss variants (recovered from confirmed-lead audit) ---
    "у кого-то есть", "у кого-нибудь есть", "у вас есть", "если у вас есть",
    "есть у кого", "в кого есть", "у кого",
    "кто-то знает", "кто знает", "кто-то нашел", "кто нашел",
    "кто дает", "кто даёт", "кто процессит", "кто располагает", "располагает",
    "кто продает", "кто продаёт", "кто-то продает", "кто-то продаёт",
    "поделитесь", "постучитесь", "отпишите", "скиньте",
    # --- Ukrainian ---
    "шукаю", "шукаємо", "шукає", "потрібен", "потрібна", "потрібно",
    "потрібні", "у кого є", "хто має", "цікавить", "порадьте",
    # --- English ---
    "looking for", "we need", "i need", "anyone have", "who has",
    "anyone got", "in search of", "need a", "need an", "searching for",
    "recommend a", "any recommendations", "who can provide",
]


@lru_cache(maxsize=8)
def load_glossary_terms(db_path: str = DEFAULT_DB) -> frozenset[str]:
    """Load every surface string from the intent_glossary table, lower-cased.

    Cached per db_path so repeated pre_filter() calls don't re-query SQLite.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Glossary DB not found: {path}. Run run_stage1b.py first."
        )
    terms: set[str] = set()
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT term, surface_forms FROM intent_glossary"
        ).fetchall()
    finally:
        conn.close()
    for term, surface_json in rows:
        if term:
            terms.add(term.strip().lower())
        for sf in json.loads(surface_json or "[]"):
            if sf:
                terms.add(str(sf).strip().lower())
    terms.discard("")
    return frozenset(terms)


def matched_terms(text: str, db_path: str = DEFAULT_DB) -> list[str]:
    """Return the glossary terms found in text (for debugging / explanation)."""
    if not text:
        return []
    low = text.lower()
    return [t for t in load_glossary_terms(db_path) if t in low]


def matched_signals(text: str) -> list[str]:
    """Return the intent signals found in text."""
    if not text:
        return []
    low = text.lower()
    return [s for s in INTENT_SIGNALS if s in low]


def pre_filter(text: str, db_path: str = DEFAULT_DB) -> bool:
    """True iff text has >=1 domain term OR >=1 intent signal.

    OR logic is recall-oriented: a message is kept for downstream LLM
    classification if it shows EITHER a domain term OR a buyer-intent cue.
    On the confirmed-lead audit this lifts recall from ~44% (AND) to ~85%,
    at the cost of a higher pass-rate (more, cheaper LLM calls). Precision
    is delegated to the downstream LLM, which is the right division of
    labour for a coarse pre-gate.
    """
    if not text:
        return False
    low = text.lower()
    if any(t in low for t in load_glossary_terms(db_path)):
        return True
    return any(s in low for s in INTENT_SIGNALS)


# --------------------------------------------------------------------------
# Test harness: run on N random messages and print pass/drop with reasons.
# --------------------------------------------------------------------------
def _test(input_path: str, sheet: str, n: int, seed: int, db_path: str) -> None:
    import random
    from glossary_builder.loader import load_messages

    messages = load_messages(input_path, sheet=sheet)
    with_text = [m for m in messages if m.text and m.text.strip()]
    rng = random.Random(seed)
    sample = rng.sample(with_text, min(n, len(with_text)))

    terms = load_glossary_terms(db_path)
    print(f"Glossary terms loaded: {len(terms)} surface strings from {db_path}")
    print(f"Intent signals: {len(INTENT_SIGNALS)}")
    print(f"Sampling {len(sample)} random messages (seed={seed}) "
          f"from sheet {sheet!r}.\n")

    passed = 0
    for i, m in enumerate(sample, 1):
        keep = pre_filter(m.text, db_path)
        terms_hit = matched_terms(m.text, db_path)
        sigs_hit = matched_signals(m.text)
        passed += int(keep)
        flag = "✅ PASS" if keep else "⛔ DROP"
        snippet = " ".join(m.text.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"[{i:>2}] {flag}  id={m.message_id}  @{m.username or '—'}")
        print(f"     text: {snippet}")
        print(f"     terms={terms_hit or '—'}  signals={sigs_hit or '—'}")
        print()

    print(f"Summary: {passed}/{len(sample)} passed the pre-filter "
          f"({len(sample) - passed} dropped).")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--input", "-i", "input_path",
                  default="data/input/PSP_Judjes_3_months.xlsx",
                  type=click.Path(exists=True))
    @click.option("--sheet", default="PSP Judjes")
    @click.option("-n", "n", type=int, default=20, help="Number of random messages.")
    @click.option("--seed", type=int, default=42)
    @click.option("--db", "db_path", default=DEFAULT_DB, type=click.Path())
    def cli(input_path, sheet, n, seed, db_path):
        """Test the pre-filter on N random messages."""
        _test(input_path, sheet, n, seed, db_path)

    cli()
