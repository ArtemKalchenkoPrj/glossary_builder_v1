"""Stage 1d — PSP / company brand mining via contextual patterns.

A common failure mode of frequency-based extractors: ~2/3 of brand names
appearing in a typical chat never make it into the candidate pool because
they only appear once or twice — well below Stage 1's frequency threshold.

This stage exploits the *structural* contexts where brand names appear:
  1. Profile signatures / intros:
        "Кристина / CEO / Betarion, IGaming"
        "Меня зовут Алексей, я PM в команде FrontsUp"
  2. Review-request questions:
        "Кто-то работал с XoraPay?"
        "представители PSP Paylift?"
        "опыт работы с ixopay?"
  3. Blacklist / reputation announcements:
        "Добавил в ЧС — RED Alert PSP libersave.com"
        "платежка Paydex - скам"

We extract candidate tokens from these contexts with regex, then a single
LLM call confirms whether each candidate is actually a brand name (vs noise).
We deliberately keep the LLM call lightweight: it sees only the candidate
token + a few example messages, not the whole context.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .candidates import Candidate
from .llm import LLMClient
from .loader import Message


# Patterns are intentionally narrow. We accept false negatives over false
# positives at this stage — the LLM validator picks up real brands the
# regex missed.
#
# Each pattern captures a group named "brand" containing the brand token.
# We require the token starts with an uppercase letter (or is a known
# all-caps acronym pattern) and has at least 4 chars.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "хто працював з X", "досвід роботи з X", "знайомий з X"
    ("worked_with_uk",
     re.compile(r"(?:працював[ала]?|досвід(?:\s+роботи)?|стикався|знайомий)\s+з\s+(?P<brand>[A-Z][\w\-\.]{3,30})\b", re.IGNORECASE)),

    # "представники X", "колеги з X", "команда X"
    ("representatives_uk",
     re.compile(r"(?:представники|колеги\s+з|команд[аи]?\s+|співробітники\s+)\s*(?P<brand>[A-Z][\w\-\.]{3,30})\b")),

    # "платіжка X" / "провайдер X" / "компанія X" — українські форми
    ("titled_provider_uk",
     re.compile(r"(?:платіжка|провайдер[а-яіїє]*|компанi[а-яіїє]+|оператор[а-яіїє]*)\s+(?P<brand>[A-Z][\w\-\.]{3,30})\b", re.IGNORECASE)),

    # "кто работал с X", "опыт работы с X", "знаком с X", "сталкивался с X"
    ("worked_with_ru",
     re.compile(r"(?:работал[аи]?|опыт(?:\s+работы)?|сталкивал(?:ся|ась)|знаком[аы]?|знаете(?:\s+про)?)\s+с\s+(?P<brand>[A-Z][\w\-\.]{3,30})\b", re.IGNORECASE)),

    # "представители X", "коллеги из X"
    ("representatives_ru",
     re.compile(r"(?:представители|коллеги\s+из|команд[аы]?\s+|сотрудники\s+)\s*(?P<brand>[A-Z][\w\-\.]{3,30})\b")),

    # English: "anyone worked with X", "experience with X"
    ("worked_with_en",
     re.compile(r"(?:worked|experience|familiar|deal[ts]?)\s+with\s+(?P<brand>[A-Z][\w\-\.]{3,30})\b", re.IGNORECASE)),

    # "X - скам / X скам / X is a scam"
    ("scam_callout",
     re.compile(r"\b(?P<brand>[A-Z][\w\-\.]{3,30})\b\s*[-–—]?\s*(?:скам|scam|fraud|шахрайство|кидалово)",
                re.IGNORECASE)),

    # "платежка X" / "PSP X" / "провайдер X" / "компания X" — only when followed by capitalised token
    ("titled_provider",
     re.compile(r"(?:платежка|psp|провайдер[а-я]*|компани[а-я]+|оператор[а-я]*)\s+(?P<brand>[A-Z][\w\-\.]{3,30})\b", re.IGNORECASE)),

    # Profile signature line — "Name / Role / Brand" or "Name, Role, Brand"
    # We capture the LAST capitalised standalone token after a comma/slash on a
    # multi-line message. This one is noisy by design; the LLM will filter.
    ("intro_signature",
     re.compile(r"(?:[/,]|\bв\s+команд[іе]\s+)\s*(?P<brand>[A-Z][a-zA-Z][\w\-\.]{2,28})\b\s*[\n,)/!.]")),

    # Domain-name reference: "X.com / X.uk / X.io / X.app" — useful for
    # blacklist posts that quote URLs.
    ("domain",
     re.compile(r"\b(?P<brand>[a-zA-Z][\w\-]{2,28})\.(?:com|io|app|uk|net|co|gg|finance|pay|fi)\b")),
]


# Common false positives we never want from these patterns — generic words
# that capitalise frequently in chat. Hard skip before sending to the LLM.
_HARD_SKIP = {
    # Країни українською
    "Україна", "Росія", "Білорусь", "Німеччина", "Франція", "Іспанія",
    "Польща", "Туреччина", "Китай", "Індія",
    # Дні тижня українською
    "Понеділок", "Вівторок", "Середа", "Четвер", "Пятниця", "Субота", "Неділя",
    # Місяці українською
    "Січень", "Лютий", "Березень", "Квітень", "Травень", "Червень",
    "Липень", "Серпень", "Вересень", "Жовтень", "Листопад", "Грудень",
    # Job titles українською
    "Керівник", "Менеджер", "Директор", "Власник", "Партнер",
    # Вітання
    "Привіт", "Дякую", "Hello", "Thanks",

    "Apple", "Google", "Visa", "Mastercard", "Maestro", "American", "Discover",
    "Telegram", "WhatsApp", "Facebook", "LinkedIn", "YouTube", "Twitter",
    "TRC-20", "ERC-20", "BEP-20",
    "Russia", "Ukraine", "Belarus", "Germany", "Spain", "France", "Italy",
    "Europe", "European", "USA", "Canada", "China", "India", "Japan", "Korea",
    "Brazil", "Mexico", "Turkey", "Latam", "MENA", "CIS", "Africa", "Asia",
    "Latin",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Head", "Chief", "Senior", "Junior", "Lead", "Account", "Business",
    "Payment", "Payments", "Manager", "Director", "Officer", "Development",
    "Sales", "Operations", "Product", "Owner",
    "EUR", "USD", "GBP", "INR", "KZT", "BRL", "MXN", "AZN",
    "WhatsApp",
    "Hello", "Привет", "Спасибо",
}

_HARD_SKIP_LOWER = {s.lower() for s in _HARD_SKIP}

def mine_brands(messages: list[Message]) -> dict[str, dict]:
    """Pass 1 — regex-only candidate brand mining (no LLM).

    Returns a dict: { brand_token: { freq, contexts, example_messages } }.
    """
    # canonical_map: lowercase -> canonical display form (most frequent casing)
    canonical_map: dict[str, str] = {}
    casing_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    out: dict[str, dict] = defaultdict(lambda: {
        "frequency": 0,
        "contexts": defaultdict(int),
        "example_messages": [],
        "seen_msg_ids": set(),
        "surface_forms": set(),
    })

    for msg in messages:
        if not msg.text:
            continue
        for label, pat in _PATTERNS:
            for m in pat.finditer(msg.text):
                token = m.group("brand").strip(".,;:'\" \t-")
                if not token or len(token) < 4:
                    continue
                if token in _HARD_SKIP:
                    continue
                if token.lower() in _HARD_SKIP_LOWER:
                    continue

                token_key = token.lower()  # ключ для дедуплікації
                casing_counter[token_key][token] += 1

                entry = out[token_key]
                entry["frequency"] += 1
                entry["contexts"][label] += 1
                entry["surface_forms"].add(token)
                if msg.message_id not in entry["seen_msg_ids"]:
                    entry["seen_msg_ids"].add(msg.message_id)
                    if len(entry["example_messages"]) < 12:
                        entry["example_messages"].append({
                            "message_id": msg.message_id,
                            "date": msg.date.isoformat() if msg.date else None,
                            "username": msg.username,
                            "text": msg.text,
                            "matched_pattern": label,
                        })

    # Визначаємо canonical форму — та що зустрічалась найчастіше
    final: dict[str, dict] = {}
    for token_key, v in out.items():
        canonical = max(casing_counter[token_key], key=casing_counter[token_key].get)
        final[canonical] = {
            "frequency": v["frequency"],
            "contexts": dict(v["contexts"]),
            "example_messages": v["example_messages"],
            "surface_forms": list(v["surface_forms"]),
        }
    return final


# ---------- LLM confirmation pass ----------

_CONFIRM_SYSTEM = """\
You filter regex-extracted candidate tokens to keep only actual *brand names*
of payment service providers, payment rails, platforms, or companies
operating in the payments / iGaming / high-risk processing space.

KEEP a candidate if it is a proper-noun brand of:
  - a PSP / acquirer / payment gateway / payment orchestrator
  - an iGaming platform / casino / sportsbook / aggregator
  - a payment method / wallet / network NOT already in our known list
  - a software vendor / consultancy / law firm serving this industry

REJECT a candidate if it is:
  - a common English, Russian, or Ukrainian word capitalised by accident
  - a job title fragment
  - a person's name (use the example messages to judge)
  - a country / region / language
  - already obvious infrastructure (Apple, Google, Telegram, WhatsApp)
  - too generic to be a brand ("Service", "Platform" alone)
  - a domain TLD or URL fragment

Output strict JSON only.
"""


_CONFIRM_USER_TEMPLATE = """\
Candidate brand tokens with example messages where each appeared (regex-mined
from profile signatures, review-requests, and blacklist callouts):

{candidates}

For each candidate, decide: is this an actual brand name a payments/iGaming
industry agent would want in its glossary? Return JSON:

{{
  "verdicts": [
    {{ "token": "<token>", "is_brand": <bool>, "reason": "<<=12 words>" }}
  ]
}}
"""


def _format_candidate_for_confirm(token: str, info: dict) -> str:
    lines = [f"Token: {token}  (freq={info['frequency']}, contexts={info['contexts']})"]
    for ex in info["example_messages"][:4]:
        snippet = (ex.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        lines.append(f"  [{ex['matched_pattern']}] {snippet}")
    return "\n".join(lines)


def confirm_brands(
    raw_candidates: dict[str, dict],
    llm: LLMClient,
    batch_size: int = 15,
) -> list[Candidate]:
    """Pass 2 — ask the LLM which of the regex-mined candidates are real brands.

    Returns Candidate objects for confirmed brands. The LLM call is cheap —
    one batched call per 15 candidates.
    """
    tokens = list(raw_candidates.keys())
    if not tokens:
        return []

    llm.set_stage("stage1d_brand_mining")
    kept: list[Candidate] = []

    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i + batch_size]
        formatted = "\n\n".join(
            _format_candidate_for_confirm(t, raw_candidates[t]) for t in batch_tokens
        )
        try:
            data, _ = llm.complete_json(
                _CONFIRM_SYSTEM,
                _CONFIRM_USER_TEMPLATE.format(candidates=formatted),
                max_tokens=1500,
            )
        except Exception:
            continue
        verdict_by_token: dict[str, bool] = {}
        for v in (data.get("verdicts") or []):
            if not isinstance(v, dict):
                continue
            tok = str(v.get("token", "")).strip()
            verdict_by_token[tok] = bool(v.get("is_brand", False))
        for t in batch_tokens:
            if not verdict_by_token.get(t):
                continue
            info = raw_candidates[t]
            kept.append(Candidate(
                term=t,
                kind="brand_candidate",
                frequency=info["frequency"],
                example_messages=[
                    {k: v for k, v in ex.items() if k != "matched_pattern"}
                    for ex in info["example_messages"]
                ],
                surface_forms=[t],
            ))

    kept.sort(key=lambda c: c.frequency, reverse=True)
    return kept
