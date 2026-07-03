"""IBAN lead classifier — identifies B2B buyers looking for banking accounts
(IBAN / SEPA / SWIFT) in high-risk industries.

Target persona: companies in iGaming, VASP/crypto, Adult, IT/marketing that
need a corporate settlement account and are either actively searching or
expressing pain with their current provider (Wise, Revolut, etc.).

NOT the target: PSP/acquirer buyers, C2B merchants, forex companies,
affiliates, or anyone looking for card-processing / APM solutions.

Pipeline (three LLM stages):

    1. Keyword filter (local, free)         — fast pre-screen
    2. LLM verification (VERIFY_MODEL)      — binary: is this an IBAN buyer?
       -> if is_buyer != true: stop, nothing persisted
    3. LLM field extraction (EXTRACT_MODEL) — structured lead data
       -> vertical, currencies, current_provider, company, notes

Context: if the message has a username, up to 10 of that author's previous
messages are pulled from client_ready_leads (most recent first) and included
in both LLM calls for richer signal. Only the ORIGINAL message is persisted.

Entry point: run_iban_lead_classifier(...), called from webhook._run() after
the primary result has been persisted (so source_lead_id is available).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

import asyncpg
import httpx

logger = logging.getLogger(__name__)
load_dotenv()

# ---------------------------------------------------------------------------
# Config — models and API key come from .env, same file the webhook reads.
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MAX_CONTEXT_MESSAGES = 10
_MAX_CONTEXT_CHARS = 3000

_TABLE_PREFIX = os.getenv("TABLE_PREFIX") or ""


def _api_key() -> str:
    key = os.environ.get("IBAN_LEADS_OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "IBAN_LEADS_OPENROUTER_API_KEY not set in .env — "
            "iban_lead_classifier cannot call OpenRouter without it."
        )
    return key


def _verify_model() -> str:
    return os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")


def _extract_model() -> str:
    return os.environ.get("JUDGE_MODEL", "google/gemini-3.1-flash-lite")


# ---------------------------------------------------------------------------
# Stage 1 — keyword filter (IBAN buyer signals, RU + EN)
# ---------------------------------------------------------------------------

_BUYER_KEYWORDS = [
    # --- Фінтех інфраструктура / вертикалі без прямого запиту ---
    # (перенесено з iban_classifier.py — CLI-версія ловила ці сигнали,
    #  вебхук-версія їх не мала)
    r"\bEMI\b",
    r"payment\s+institution",
    r"финансовая\s+инфраструктур\w+",
    r"banking\s+(?:solution|infrastructure|setup|partner|rails)",
    r"банковское\s+решение",
    r"банківське\s+рішення",
    r"финансовое\s+решение",
    r"merchant\s+(?:bank\s+)?account",
    r"merchant\s+banking",
    r"открытие\s+(?:корпоративного\s+)?счёт\w+",
    r"открытие\s+(?:корпоративного\s+)?счет\w+",
    r"відкриття\s+рахунк\w+",
    r"открыть\s+компани\w+",
    r"регистрация\s+компани\w+",
    r"company\s+(?:formation|registration|setup|incorporation)",
    r"open\s+(?:a\s+)?(?:company|entity)",
    r"\bVASP\b",
    r"igaming\s+(?:operator|project|platform|company|license)",
    r"crypto\s+(?:exchange|gateway|company|license|startup)",
    r"gambling\s+(?:license|operator|company)",
    r"betting\s+(?:company|operator|platform)",
    r"\blicens\w+\b.{0,30}\b(?:crypto|igaming|gambling|payment|emi)\b",
    r"\b(?:crypto|igaming|gambling|payment|emi)\b.{0,30}\blicens\w+\b",
    r"лицензи\w+.{0,30}(?:крипто|игейминг|платёжн)",
    r"(?:malta|кипр|cyprus|estoni\w+|latvia|lithu\w+)\b.{0,50}\b(?:company|account|license|бизнес|компани)",
    r"\bCFO\b",
    r"\btreasur\w+\b",
    r"финансовый\s+директор",
    r"фінтех\s+ліцензі\w+",
    r"fintech\s+licens\w+",
    r"нужна\s+платёж\w+",
    r"нужно\s+(?:финансовое|банковское)\s+решение",
    r"ищем\s+(?:финансовое|банковское)\s+решение",

    # --- Прямой спрос на счёт / IBAN ---
    r"\bIBAN\b",
    r"расчётн\w+\s+счёт",
    r"расчетн\w+\s+счет",
    r"открыть\s+счёт",
    r"открыть\s+счет",
    r"нужен\s+счёт",
    r"нужен\s+счет",
    r"ищем\s+счёт",
    r"ищем\s+счет",
    r"банковский\s+счёт",
    r"банковский\s+счет",
    r"корпоративный\s+счёт",
    r"корпоративный\s+счет",
    r"нужен\s+банк\b",
    r"нужны\s+реквизиты",
    r"нужен\s+провайдер\s+для\s+расчёт",
    r"нужен\s+провайдер\s+для\s+расчет",
    r"мультивалютный\s+счёт",
    r"мультивалютный\s+счет",
    r"business\s+account",
    r"corporate\s+account",
    r"settlement\s+account",
    r"bank\s+account",
    r"open\s+(?:a\s+)?(?:business\s+|corporate\s+)?account",
    r"need\s+(?:a\s+)?(?:bank|banking|iban)",
    r"looking\s+for\s+(?:a\s+)?bank",
    r"banking\s+(?:provider|solution|partner)",
    # --- SEPA / SWIFT / реквизиты ---
    r"\bSEPA\b",
    r"\bSWIFT\b",
    r"sepa\s+счёт",
    r"sepa\s+счет",
    r"uk\s+счёт",
    r"uk\s+счет",
    r"gbp\s+счёт",
    r"gbp\s+счет",
    r"принимать\s+sepa",
    r"получать\s+swift",
    r"receive\s+(?:via\s+)?sepa",
    r"receive\s+(?:via\s+)?swift",
    r"wire\s+transfer",
    r"payment\s+rails",
    # --- Крипто on/off-ramp ---
    r"\bon[\s-]?ramp\b",
    r"\boff[\s-]?ramp\b",
    r"крипто\s+в\s+фиат",
    r"вывод\s+usdt",
    r"вывод\s+крипт\w*",
    r"зачисление\s+с\s+биржи",
    r"crypto\s+to\s+fiat",
    r"usdt\s+(?:to|в)\s+(?:eur|gbp|usd)",
    # --- Боль с текущим провайдером ---
    r"wise\s+заблокировал",
    r"revolut\s+закрыл",
    r"заблокировали\s+счёт",
    r"заблокировали\s+счет",
    r"заморозили\s+счёт",
    r"заморозили\s+счет",
    r"замораживают\s+транзакц\w*",
    r"блокируют\s+счёт",
    r"блокируют\s+счет",
    r"альтернатива\s+revolut",
    r"альтернатива\s+wise",
    r"wise\s+(?:blocked|banned|закрыл)",
    r"revolut\s+(?:closed|banned|закрыл)",
    r"account\s+(?:blocked|frozen|terminated|closed)",
    r"alternative\s+to\s+(?:revolut|wise)",
]

_COMPILED_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in _BUYER_KEYWORDS]


def _keyword_hits(text: str) -> int:
    return sum(1 for pattern in _COMPILED_KEYWORDS if pattern.search(text))


# ---------------------------------------------------------------------------
# Stage 1b — стоп-слова (жёсткий reject ДО LLM, экономит токены)
# ---------------------------------------------------------------------------
# Перенесено з iban_classifier.py — вебхук-версія раніше не мала цього шару
# і покладалась виключно на LLM-промпт для відсіву дроп-схем/РФ/Індії/F2F.

_STOP_KEYWORDS = [
    # Дроп-схемы
    r"\bWTB\b",
    r"куплю\s+(?:расчетн|счет|ип|ооо)",
    r"покупаю\s+(?:счет|аккаунт)",
    r"продаю\s+(?:счет|аккаунт|iban)",
    r"набираем\s+анкет",
    r"нужны\s+дроп",
    r"ищу\s+дроп",
    # Россия / санкции
    r"\bRUB\b",
    r"\bруб\b",
    r"залью\s+рф",
    r"поток\s+рф",
    r"реквизиты\s+рф",
    r"\bqiwi\b",
    # F2F / наличные
    r"\bF2F\b",
    r"face[\s-]?to[\s-]?face",
    r"личная\s+встреч",
    r"наличными",
    r"\bcash\b",
    # Индия / INR
    r"\bINR\b",
    r"\bUPI\b",
    r"indian\s+(?:bank|account)",
    r"\bindus\b",
    r"\bbandhan\b",
    r"\bIDBI\b",
    # Базы данных IBAN
    r"iban\s+database",
    r"база\s+(?:iban|счетов|реквизит)",
]

_COMPILED_STOP = [re.compile(p, re.IGNORECASE) for p in _STOP_KEYWORDS]


def _stop_hits(text: str) -> list[str]:
    return [p.pattern for p in _COMPILED_STOP if p.search(text)]


# ---------------------------------------------------------------------------
# Stage 2 — LLM verification: is this an IBAN/banking account buyer?
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = """You are a fintech market analyst. Your task is to determine
whether the author of a message is a B2B company LOOKING FOR a banking account
(IBAN / SEPA / SWIFT / corporate account) for their business settlements.

IMPORTANT: when in doubt — is_buyer: true with confidence: low.
Better to pass an irrelevant lead than to lose a real client.

Criteria for IBAN buyer (is_buyer: true):
- Directly states they need a bank account, IBAN, or settlement account
- Expresses pain with a current banking provider (Wise blocked, Revolut closed, etc.)
- Looking for crypto on/off-ramp (converting USDT/crypto to fiat via a bank account)
- Works in a high-risk vertical: iGaming, crypto/VASP, Adult, IT/digital goods
- Asking about SEPA, SWIFT, GBP/EUR/multi-currency accounts for B2B purposes

NOT an IBAN buyer (is_buyer: false):
- Looking for a PSP, payment processor, or card acquiring
- Wants to accept payments FROM end-users / players (C2B flow)
- Looking for payment methods, APMs, e-wallets, or checkout solutions
- A banking provider pitching their own services
- Forex company, affiliate network
- Looking for an anonymous account or account without KYC

Reply ONLY in valid JSON (no markdown):
{
  "is_buyer": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence"
}"""


# ---------------------------------------------------------------------------
# Stage 3 — LLM field extraction: structured lead data
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """You are a fintech market analyst. You receive one or more
messages from the same author from Telegram chats.

Your tasks:
1. Issue a final verdict — is the author a B2B buyer looking for a banking account (IBAN)?
2. Extract structured information about the lead

FIRST — check if the author is a PROVIDER:
If the author OFFERS, PROVIDES, or SELLS banking/IBAN/payment solutions to others
("we offer", "we provide", "our solutions", "contact us for setup", "#selling",
"предлагаем", "предоставляем") → ALWAYS "not_lead", regardless of anything else.

ONLY if author is NOT a provider, apply the criteria below:

IBAN buyer (judged_as: "buyer"):
- Directly states they need a bank account, IBAN, or settlement account
- Expressing pain with current banking provider (blocked, frozen, closed)
- Looking for crypto on/off-ramp via corporate bank account
- Operates in high-risk vertical: iGaming, crypto/VASP, Adult, IT/digital goods

NOT an IBAN buyer (judged_as: "not_lead"):
- Looking for PSP, card acquiring, or payment methods
- Wants to accept payments from end-users (C2B)
- A banking provider pitching their services
- Forex company or affiliate network

Unclear (judged_as: "unclear"):
- Some signals present but no clear confirmation of IBAN need

Return ONLY valid JSON (no markdown, no explanations):
{
  "judged_as":        "buyer" | "not_lead" | "unclear",
  "company":          "company name or null",
  "vertical":         ["iGaming" | "crypto/VASP" | "Adult" | "IT/marketing" | "digital goods" | "other"],
  "geo":               ["countries or regions mentioned in text, as written, do not normalize"],
  "currencies":       ["list of needed currencies: EUR, GBP, USD, USDT, etc."],
  "current_provider": "provider they are leaving (Wise, Revolut, etc.) or null",
  "position":         "author job title or null",
  "notes":            "anything important that didn't fit above, or null"
}

Rules:
- Unknown fields → null, do not guess
- Array fields always as arrays, even with a single element
- Company name as written in the text, do not translate
- geo — as written in the text, do not normalize or translate
- notes — max one sentence"""


# ---------------------------------------------------------------------------
# OpenRouter call helper
# ---------------------------------------------------------------------------

async def _call_openrouter(model: str, system: str, user: str, max_tokens: int) -> dict:
    """POST one chat-completion request to OpenRouter, return parsed JSON.

    On any failure (network, non-2xx, bad JSON) returns a dict with an
    "_error" key so callers can branch on it without a try/except at every
    call site.
    """
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_api_key()}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/psp-detector",
                },
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw_clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw_clean)

    except Exception as exc:  # noqa: BLE001 — network, HTTP, JSON — all fold to one error shape
        logger.error("second_pass: OpenRouter call failed (model=%s): %s", model, exc)
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Context — pull this author's recent messages from client_ready_leads
# ---------------------------------------------------------------------------

async def _load_author_context(
    username: Optional[str],
    conn: asyncpg.Connection,
) -> list[str]:
    """Fetch up to _MAX_CONTEXT_MESSAGES previous texts by the same author.

    Returns an empty list if username is None or no prior messages exist —
    callers then fall back to processing the single current message alone.
    """
    if not username:
        return []

    rows = await conn.fetch(
        f"""
        SELECT text FROM {_TABLE_PREFIX}client_ready_leads
        WHERE username = $1 AND text IS NOT NULL
        ORDER BY created_at DESC
        LIMIT $2
        """,
        username,
        _MAX_CONTEXT_MESSAGES,
    )
    return [row["text"] for row in rows if row["text"]]


def _build_combined_text(current_text: str, context_texts: list[str]) -> str:
    """Combine current message + historical context into one block, capped
    at _MAX_CONTEXT_CHARS — same truncation strategy as judge_messages.py.
    """
    all_texts = [current_text] + context_texts
    combined = "\n\n---\n\n".join(all_texts)
    if len(combined) > _MAX_CONTEXT_CHARS:
        combined = combined[:_MAX_CONTEXT_CHARS] + "...[обрізано]"
    return combined


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

_INSERT_CLIENT_READY_SQL = f"""
INSERT INTO {_TABLE_PREFIX}client_ready_leads (
    source_lead_id, username, text, msg_timestamp,
    lead_type, vertical, geo, payment_methods_mentioned, approved_by,
    company, position, notes, current_provider
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7, $8, $9,
    $10, $11, $12, $13
)
RETURNING id;
"""


async def _persist_iban_lead(
    *,
    source_lead_id: Optional[int],
    username: Optional[str],
    original_text: str,
    timestamp: Optional[datetime],
    extracted: dict,
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> None:
    """Write a confirmed IBAN-buyer lead to client_ready_leads.

    Only the ORIGINAL message goes into `text` — context messages used for
    the LLM calls are never persisted here.

    If dry_run=True, nothing is written to the DB — the row that WOULD have
    been inserted is printed to the console instead, for safe end-to-end
    testing against the real LLM pipeline without touching the database.
    """
    vertical = extracted.get("vertical") or []
    currencies = extracted.get("currencies") or []
    geo = extracted.get("geo") or []

    if dry_run:
        print("\n" + "=" * 70)
        print("[iban_classifier DRY RUN] row that WOULD be written to client_ready_leads:")
        print("=" * 70)
        print(f"  source_lead_id   : {source_lead_id}")
        print(f"  username         : {username}")
        print(f"  text             : {original_text!r}")
        print(f"  msg_timestamp    : {timestamp}")
        print(f"  lead_type        : seeking_iban")
        print(f"  vertical         : {vertical}")
        print(f"  geo              : {geo}")
        print(f"  currencies       : {currencies}")
        print(f"  current_provider : {extracted.get('current_provider')}")
        print(f"  -- LLM-extracted fields --")
        print(f"  company          : {extracted.get('company')}")
        print(f"  position         : {extracted.get('position')}")
        print(f"  notes            : {extracted.get('notes')}")
        print("=" * 70 + "\n")
        return

    async with conn_pool.acquire() as conn:
        row = await conn.fetchrow(
            _INSERT_CLIENT_READY_SQL,
            source_lead_id,
            username,
            original_text,
            timestamp,
            "seeking_iban",                                         # lead_type
            json.dumps(vertical, ensure_ascii=False),              # vertical
            json.dumps(geo, ensure_ascii=False),
            json.dumps(currencies, ensure_ascii=False),            # payment_methods_mentioned (currencies here)
            'huyochok',                                                   # approved_by — set later by reviewer
            extracted.get("company"),
            extracted.get("position"),
            extracted.get("notes"),
            extracted.get("current_provider"),
        )

    print(
        f"iban_classifier: persisted IBAN lead id={row['id'] if row else None} "
        f"username={username} company={extracted.get('company')!r} "
        f"vertical={vertical} current_provider={extracted.get('current_provider')!r}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_iban_lead_classifier(
    *,
    text: str,
    source_lead_id: Optional[int],
    message_id: Optional[int],
    username: Optional[str],
    timestamp: Optional[datetime],
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> bool:
    """Run the IBAN lead classifier on an incoming message.

    Can be used as the primary classifier or as a second-pass on messages
    rejected by another pipeline.

    Returns True if the message was classified as an IBAN buyer AND actually
    persisted to client_ready_leads (i.e. dry_run=False and the insert
    succeeded). Returns False at every early-exit point (empty text,
    stop-word, no keyword hits, LLM says not a buyer, LLM error) and also
    when dry_run=True, since nothing was actually written in that case.

    This mirrors the run_psp_provider_classifier / run_casino_classifier
    convention so callers can chain second-pass classifiers, e.g.:

        if not psp_classified and not casino_classified:
            iban_classified = await run_iban_lead_classifier(...)
    """
    text = (text or "").strip()
    if not text:
        print("[iban_classifier] step 0: empty text — exit")
        return False
    print("[iban_classifier] step 0: start, text =", text[:80])

    # --- Step 1: context -------------------------------------------------------
    print("[iban_classifier] step 1: loading author context...")
    context_texts: list[str] = []
    try:
        async with conn_pool.acquire() as conn:
            context_texts = await _load_author_context(username, conn)
    except Exception as exc:
        print(f"[iban_classifier] step 1: ERROR loading context: {exc!r}")
        raise
    print("[iban_classifier] step 1: context loaded, count =", len(context_texts))
    combined_text = _build_combined_text(text, context_texts)

    # --- Step 1.5: stop-words — instant reject, no LLM call --------------------
    stops = _stop_hits(combined_text)
    if stops:
        print("[iban_classifier] step 1.5: stop hit —", stops[0], "— exit")
        logger.debug(
            "iban_classifier: stop-word reject (message_id=%s): %s", message_id, stops[0]
        )
        return False

    # --- Step 2: keyword filter ------------------------------------------------
    hits = _keyword_hits(combined_text)
    print("[iban_classifier] step 2: keyword hits =", hits)
    if hits == 0:
        print("[iban_classifier] step 2: 0 hits — exit")
        logger.debug("iban_classifier: no keyword hits, skipping message_id=%s", message_id)
        return False

    # --- Step 3: LLM verification ----------------------------------------------
    print("[iban_classifier] step 3: calling verify LLM, model =", _verify_model())
    author_line = username or "anon"
    verify_user_msg = f"[Author]: {author_line}\n[Message]: {combined_text}"

    verify_result = await _call_openrouter(
        model=_verify_model(),
        system=_VERIFY_SYSTEM_PROMPT,
        user=verify_user_msg,
        max_tokens=200,
    )
    print("[iban_classifier] step 3: verify result =", verify_result)

    if verify_result.get("_error"):
        print("[iban_classifier] step 3: call failed — exit")
        logger.warning(
            "iban_classifier: verification call failed for message_id=%s, skipping", message_id
        )
        return False

    if verify_result.get("is_buyer") is not True:
        print("[iban_classifier] step 3: is_buyer not True — exit")
        logger.debug(
            "iban_classifier: not an IBAN buyer (message_id=%s): %s",
            message_id,
            verify_result.get("reason"),
        )
        return False

    # --- Step 4: LLM field extraction ------------------------------------------
    print("[iban_classifier] step 4: calling extract LLM, model =", _extract_model())
    extract_user_msg = f"[Author]: {author_line}\n\n[Message]:\n{combined_text}"

    extract_result = await _call_openrouter(
        model=_extract_model(),
        system=_EXTRACT_SYSTEM_PROMPT,
        user=extract_user_msg,
        max_tokens=500,
    )
    print("[iban_classifier] step 4: extract result =", extract_result)

    if extract_result.get("_error"):
        print("[iban_classifier] step 4: call failed — exit")
        logger.warning(
            "iban_classifier: extraction call failed for message_id=%s, skipping", message_id
        )
        return False

    if extract_result.get("judged_as") != "buyer":
        print("[iban_classifier] step 4: judged_as != buyer — exit")
        logger.debug(
            "iban_classifier: extraction stage downgraded verdict (message_id=%s): %s",
            message_id,
            extract_result.get("judged_as"),
        )
        return False

    # --- Step 5: persist -------------------------------------------------------
    print("[iban_classifier] step 5: persisting lead...")
    await _persist_iban_lead(
        source_lead_id=source_lead_id,
        username=username,
        original_text=text,
        timestamp=timestamp,
        extracted=extract_result,
        conn_pool=conn_pool,
        dry_run=dry_run,
    )
    print("[iban_classifier] step 5: done")

    # dry_run means nothing was actually written — report False so callers
    # (e.g. a classifier cascade) still try the next fallback classifier.
    return not dry_run