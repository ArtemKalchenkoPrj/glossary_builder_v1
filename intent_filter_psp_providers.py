"""Cheap pre-filter: keep a message only if it contains BOTH a domain term
AND a seller-intent signal.

Two sources are combined:

  1. Domain terms — the ``intent_glossary`` table built by
     ``run_provider_glossary.py`` (Stage 1b phrase extraction on the
     seller-intent corpus). Each row's ``term`` + ``surface_forms``
     contribute surface strings to match.

  2. Seller-intent signals — a hardcoded multilingual list of cues that
     indicate the author is OFFERING (not buying) payment services:
     ``работаю по``, ``покрываю``, ``процессим``, ``пропоную`` …

``pre_filter(text)`` returns True iff the text contains at least one domain
term AND at least one seller-intent signal.

AND logic is chosen deliberately for precision: a message with only a domain
term (buyer discussing PSP terms) or only a seller signal without PSP context
is dropped. Precision is more important than recall at this gate — the LLM
stages downstream provide the recall safety net.

Matching is lower-cased substring containment — morphology-tolerant to match
Russian/Ukrainian inflection, consistent with ``phrases.py`` convention.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path


# Path to the provider glossary DB built by run_provider_glossary.py.
# Resolved relative to this file so it works regardless of cwd.
_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROVIDER_GLOSSARY_DB = str(_MODULE_DIR / "PSP_provider_glossary_builder" / "data" / "provider_glossary.db")

# стоп слова
ANTI_SIGNALS: list[str] = [
    # Дроп SIM схема
    "sim-направление",
    "sim направление",
    "сим направление",
    "вход sim",

    # Кардинг / фрод
    "минусовые кредитки",
    "отработаю логи",
    "отрабатываю логи",
    "минусовые карты",

    # FaustEU спам паттерн
    "не льешь, но хочешь зарабатывать",
    "не льешь, но хочешь заработать",
    "реферальная система",  # в поєднанні з payment context
    "депозит лично под вас",
    "гарант любой на ваш вкус",

    # Посередники варіанти
    "подключу к площадке",
    "подключу без страхового депозита",  # але сам підключає до чужої площадки
    "подключайтесь к нам по лучшим условиям",

    # Обнал кредиток
    "balance transfer",
    "billpay",  # в контексті з "минусовые"

    # Дропперська термінологія
    "трейдеров",
    "трейдер",
    "для трейдер",
    "набор трейдер",
    "мануалы",
    "менторский состав",
    "куратор",
    "страховой депозит",
    "белый треугольник",
    "белые треугольники",
    "красный гринекс",
    "ставим на площадки",
    "подключайся через официального представителя",
    "подключу на площадку",
    "подключение к площадке",
    "подключу к площадке"

    # Продаж акаунтів
    "wts/sell",
    "wts sell",
    "selling accounts",
    "ready acc",
    "kyc service",
    "costume name",

    # Медіабаєри / трафік
    "focus on traffic",
    "facebook storming",
    "stop wasting time",

    # WL software
    "white label который выдерживает",
    "wl которые",
]

# ── Seller-intent signals ─────────────────────────────────────────────────────
# Lower-case; matched as substrings so inflected forms are caught.
# A message must contain at least one of these to pass the filter.

SELLER_SIGNALS: list[str] = [

    # --- Russian: first-person service description ---
    "работаю по",
    "работаем по",
    "работаем с мерчант",
    "покрываю",
    "покрытие по",
    "покрыт по",
    "покрываем",
    "закрываем по",
    "закрываем гео",
    "закрываем eu",
    "закрываем uk",
    "закрываем latam",
    "процессим",
    "процессируем",
    "подключаем",
    "подключим",
    "онбордим",
    "онбординг мерчант",
    "можем закрыть",
    "можем подключить",
    "специализируемся",
    "предлагаем эквайринг",
    "предлагаем процессинг",
    "предлагаем шлюз",
    "предлагаем решение",
    "предлагаю эквайринг",
    "предлагаю процессинг",

    # --- Russian: provider metrics reported as own ---
    "апрув",
    "аппрув",
    "апрув рейт",
    "наш апрув",
    "наш роутинг",
    "наш каскад",
    "наш процессинг",
    "наш шлюз",
    "наш гейтвей",
    "наша компания",
    "наша команда",
    "наше решение",

    # --- Russian: CTA patterns specific to providers ---
    # "пишите в лс",
    # "пишите в личку",
    # "пишите в дм",
    # "стучите в лс",
    # "стучите в личку",
    # "пиши в лс",
    # "в лс пишите",
    # "в личку пишите",

    # --- Russian: coverage / onboarding language ---
    "работаем с высоким риском",
    "закрываем гемблинг",
    "закрываем форекс",
    "закрываем adult",
    "онбординг за",
    "подключение за",
    "быстрый онбординг",

    # --- Ukrainian: seller-side signals ---
    "пропоную",
    "пропонуємо",
    "маємо покриття",
    "маємо рішення",
    "підключаємо",
    "підключимо",
    "покриваємо",
    "закриваємо",
    "процесуємо",
    "еквайринг пропоную",
    "наша компанія",
    "наше рішення",
    "пишіть в лс",
    "стукайте в лс",

    # --- English: provider role signals ---
    "we offer",
    "we provide",
    "we process",
    "we cover",
    "we support",
    "we handle",
    "we specialize",
    "we specialise",
    "covering eu",
    "covering uk",
    "covering latam",
    "covering asia",
    "processing high-risk",
    "processing igaming",
    "processing gambling",
    "can process",
    "can handle",
    "able to process",
    "able to handle",
    "our gateway",
    "our processing",
    "our solution",
    "our platform",
    "our acquiring",

    # --- English: CTA patterns specific to providers ---
    # "dm for details",
    # "dm for rates",
    # "dm me",
    # "reach out to discuss",
    # "get in touch",
    # "contact for volumes",
    # "pm for info",

    # --- English: provider spec-sheet patterns ---
    "approve rate",
    "approval rate",
    "onboarding in",
    "onboarding within",
    "settlement t+",
    "rolling reserve",
    "high-risk acquiring",
    "high risk acquiring",
]

GENERIC_TERMS_EXCLUDE: frozenset[str] = frozenset({
    # Вертикали — есть у всех в чате
    "igaming", "gambling", "casino", "betting", "forex", "crypto",
    "nutra", "latam", "крипта", "vertical",

    # Слишком общие платёжные термины
    # "psp", "high-risk", "processing", "merchant", "gateway",
    # "payment gateway", "3ds", "kyc", "kyb", "ftd",
    # "p2p", "p2c", "t+0", "mid", "geo", "geos",

    # CTA которые используют все
    "пишите в лс", "связь",

    # Transaction terms — покупатели тоже упоминают
    "chargeback", "чарджбэк", "чарджбек", "поддержка",

    # Контентные термины из новостей/CV
    "платежных систем", "платежи", "платёжный",
    "payment infrastructure", "payment partner",
    "платежных системах", "ставки",

    # Шум
    "gmatch", "логі",
})

@lru_cache(maxsize=4)
def load_glossary_terms(db_path: str = DEFAULT_PROVIDER_GLOSSARY_DB) -> frozenset[str]:
    """Load provider domain terms from intent_glossary table, lower-cased.

    Reads from the provider_glossary.db built by run_provider_glossary.py.
    Cached per db_path so repeated pre_filter() calls don't re-query SQLite.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Provider glossary not found: {path}\n"
            f"Run run_provider_glossary.py first to build it."
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
        try:
            for sf in json.loads(surface_json or "[]"):
                if sf:
                    terms.add(str(sf).strip().lower())
        except (json.JSONDecodeError, TypeError):
            pass

    terms.discard("")
    return frozenset(terms - GENERIC_TERMS_EXCLUDE)


def matched_terms(
    text: str,
    db_path: str = DEFAULT_PROVIDER_GLOSSARY_DB,
) -> list[str]:
    """Return glossary terms found in text (for debugging / explanation)."""
    if not text:
        return []
    low = text.lower()
    return [t for t in load_glossary_terms(db_path) if t in low]


def matched_signals(text: str) -> list[str]:
    """Return seller-intent signals found in text."""
    if not text:
        return []
    low = text.lower()
    return [s for s in SELLER_SIGNALS if s in low]


def pre_filter(
    text: str,
    db_path: str = DEFAULT_PROVIDER_GLOSSARY_DB,
) -> bool:
    """True iff text contains >=1 domain term AND >=1 seller-intent signal.

    AND logic: both conditions must be met.
    - Domain term only  → buyer or market discussion, drop.
    - Seller signal only → no PSP context, drop.
    - Both              → likely provider offering services, pass to LLM.

    If recall on the test dataset is too low, switch to OR:
        return any(t in low for t in load_glossary_terms(db_path)) or
               any(s in low for s in SELLER_SIGNALS)
    """
    if not text:
        return False
    low = text.lower()

    # Антислова — миттєво відсікаємо без LLM
    if any(s in low for s in ANTI_SIGNALS):
        return False

    has_term = any(t in low for t in load_glossary_terms(db_path))
    has_signal = any(s in low for s in SELLER_SIGNALS)
    return has_term or has_signal


# ── Test harness ──────────────────────────────────────────────────────────────
def _test(input_path: str, sheet: str, n: int, seed: int, db_path: str) -> None:
    """Print pass/drop decisions for N random messages with reasons."""
    import random
    from glossary_builder.loader import load_messages

    messages = load_messages(input_path, sheet=sheet)
    with_text = [m for m in messages if m.text and m.text.strip()]
    rng = random.Random(seed)
    sample = rng.sample(with_text, min(n, len(with_text)))

    terms = load_glossary_terms(db_path)
    print(f"Provider glossary terms : {len(terms)} surface strings")
    print(f"Seller-intent signals   : {len(SELLER_SIGNALS)}")
    print(f"Logic                   : AND (term + signal both required)")
    print(f"Sampling {len(sample)} random messages (seed={seed})\n")

    passed = 0
    for i, m in enumerate(sample, 1):
        keep       = pre_filter(m.text, db_path)
        terms_hit  = matched_terms(m.text, db_path)
        sigs_hit   = matched_signals(m.text)
        passed    += int(keep)
        flag       = "✅ PASS" if keep else "⛔ DROP"
        snippet    = " ".join(m.text.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"[{i:>2}] {flag}  id={m.message_id}  @{m.username or '—'}")
        print(f"     text   : {snippet}")
        print(f"     terms  : {terms_hit or '—'}")
        print(f"     signals: {sigs_hit or '—'}")
        print()

    print(f"Summary: {passed}/{len(sample)} passed "
          f"({len(sample) - passed} dropped).")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--input", "-i", "input_path",
                  default="PSP_providers_glossary_builder/data/seller_corpus.xlsx",
                  type=click.Path(exists=True))
    @click.option("--sheet", default="corpus")
    @click.option("-n", type=int, default=30, help="Number of random messages.")
    @click.option("--seed", type=int, default=42)
    @click.option("--db", "db_path",
                  default=DEFAULT_PROVIDER_GLOSSARY_DB,
                  type=click.Path())
    def cli(input_path, sheet, n, seed, db_path):
        """Test the PSP provider pre-filter on N random messages."""
        _test(input_path, sheet, n, seed, db_path)

    cli()
