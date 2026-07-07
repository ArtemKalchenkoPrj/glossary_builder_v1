"""Second-pass classifier — a safety net for messages marked is_lead=False
by the primary pipeline.

Adapted from the standalone scripts psp_detector.py + judge_messages.py.
Instead of looking for "buyers" (PSP leads, the primary pipeline's job),
this pass looks for the OPPOSITE persona: PSP PROVIDERS pitching their own
payment processing services. The primary pipeline's pre-filter / stage1 /
full prompt are all tuned for buyer-intent, so provider pitches often slip
through as is_lead=False — this is where we catch them.

Pipeline (two LLM stages, same shape as the original scripts):

    1. Keyword filter (local, free)        — psp_detector.py logic
    2. LLM verification (GLOSSARY_MODEL)   — psp_detector.py logic
       -> if is_psp != true: stop, nothing persisted
    3. LLM field extraction (JUDGE_MODEL)  — judge_messages.py logic
       -> company, geo, methods, position, notes

Context: if the message has a username, up to 10 of that author's previous
messages are pulled from client_ready_leads (most recent first) and used
as extra context for both LLM calls — exactly like judge_messages.py
grouped by user_id, except the group is fetched from the DB instead of
from an in-memory CSV. Only the ORIGINAL message (not the context ones)
is ever written back to client_ready_leads.text.

Entry point: run_psp_provider_classifier(...), called from webhook._run() right after
the primary result has been persisted (so source_lead_id is available).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — models and API key come from .env, same file the webhook reads.
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MAX_CONTEXT_MESSAGES = 10
_MAX_CONTEXT_CHARS = 3000

_TABLE_PREFIX = os.getenv("TABLE_PREFIX") or ""

def _api_key() -> str:
    key = os.environ.get("PSP_PROVIDERS_OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "PSP_PROVIDERS_OPENROUTER_API_KEY not set in .env — "
            "psp_provider_classifier cannot call OpenRouter without it."
        )
    return key


def _verify_model() -> str:
    return os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")


def _extract_model() -> str:
    return os.environ.get("JUDGE_MODEL", "google/gemini-3.1-flash-lite")


# ---------------------------------------------------------------------------
# Stage 1 — keyword filter (verbatim logic from psp_detector.py)
# ---------------------------------------------------------------------------

_PROVIDER_KEYWORDS = [
    r"платежн\w* провайдер",
    r"payment provider",
    r"\bPSP\b",
    r"процессинг\w*",
    r"эквайр\w*",
    r"acquire[rd]?",
    r"работаю по\b",
    r"покрыт\w+ по\b",
    r"cover\w*\s+(by|in)\b",
    r"работаем с\b",
    r"\bапрув\w*\b",
    r"\bapprove\s*rate\b",
    r"\bроутинг\b",
    r"\brouting\b",
    r"\bкаскад\w*\b",
    r"\bchargeback\b",
    r"\bчарджбек\b",
    r"\bмерчант\b",
    r"\bmerchant\b",
    r"\bМСС\b",
    r"\bMCC\b",
    r"\bвыплат\w*\b.*транзакц\w*",
    r"транзакц\w*.*\bвыплат\w*\b",
    r"подключ\w+\s+(к|вас|вам|магазин|мерчант)",
    r"интегрируем\w*",
    r"можем\s+подключить",
    r"предлагаем\s+(шлюз|эквайринг|процессинг)",
]

_COMPILED_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in _PROVIDER_KEYWORDS]


def _keyword_hits(text: str) -> int:
    return sum(1 for pattern in _COMPILED_KEYWORDS if pattern.search(text))


# ---------------------------------------------------------------------------
# Stage 2 — LLM verification (verbatim prompt from psp_detector.py)
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = """Ти — аналітик фінтех-ринку. Твоє завдання — визначити,
чи є автор повідомлення платіжним провайдером (PSP / acquirer / payment gateway),
який ПРОПОНУЄ свої послуги.

Критерії PSP-провайдера:
- Прямо заявляє, що є провайдером / процесором / еквайєром
- Пропонує підключення, інтеграцію або обробку платежів
- Згадує покриття по країнах у контексті своїх послуг
- Вживає технічну термінологію (approve rate, routing, MCC, chargeback, cascade) від свого імені

НЕ є PSP-провайдером:
- Клієнт, який шукає провайдера
- Людина, яка просто обговорює ринок чи ставить питання
- Розробник, який налаштовує платежі для свого проекту

Відповідай ТІЛЬКИ у форматі JSON (без markdown):
{
  "is_psp": true або false,
  "confidence": "high" | "medium" | "low",
  "reason": "одне речення"
}"""


# ---------------------------------------------------------------------------
# Stage 3 — LLM field extraction (verbatim prompt from judge_messages.py)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """Ти — аналітик фінтех-ринку. Тобі надають одне або кілька повідомлень від одного автора з Telegram-чатів.

Твої завдання:
1. Винести фінальний вердикт — чи є автор PSP-провайдером який ПРОПОНУЄ свої послуги
2. Витягти структуровану інформацію про компанію

PSP-провайдер (judged_as: "provider"):
- Прямо заявляє що є провайдером / процесором / еквайєром
- Пропонує підключення, інтеграцію або обробку платежів
- Згадує покриття по країнах у контексті своїх послуг

НЕ є PSP-провайдером (judged_as: "not_psp"):
- Клієнт який шукає провайдера
- Людина яка обговорює ринок або ставить питання
- Розробник який налаштовує платежі для свого проекту

Недостатньо інформації (judged_as: "unclear"):
- Є ознаки але немає чіткого підтвердження

Поверни ТІЛЬКИ валідний JSON (без markdown, без пояснень):
{
  "judged_as": "provider" | "not_psp" | "unclear",
  "company":   "назва компанії або null",
  "geo":       ["список країн або регіонів покриття"],
  "methods":   ["список платіжних методів: SWIFT, SEPA, карти, крипто тощо"],
  "position":  "посада автора або null",
  "notes":     "важливе що не вписалось у поля вище, або null"
}

Правила:
- Якщо поле невідоме — null, не вигадуй
- geo і methods — завжди масиви, навіть якщо один елемент
- Назву компанії пиши як у тексті, не перекладай
- notes — коротко, одне речення максимум"""


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
        logger.error("psp_provider_classifier: OpenRouter call failed (model=%s): %s", model, exc)
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
    company, position, notes
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7, $8, $9,
    $10, $11, $12
)
RETURNING id;
"""


async def _persist_provider_lead(
    *,
    source_lead_id: Optional[int],
    username: Optional[str],
    original_text: str,
    timestamp: Optional[datetime],
    extracted: dict,
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> None:
    """Write a confirmed PSP-provider lead to client_ready_leads.

    Only the ORIGINAL message goes into `text` — context messages used for
    the LLM calls are never persisted here (they already exist as their
    own rows from earlier second-pass runs).

    If dry_run=True, nothing is written to the DB — the row that WOULD have
    been inserted is printed to the console instead, for safe end-to-end
    testing against the real LLM pipeline without touching the database.
    """
    geo = extracted.get("geo") or []
    methods = extracted.get("methods") or []

    if dry_run:
        print("\n" + "=" * 70)
        print("[psp_provider_classifier DRY RUN] рядок, який мав би записатись у client_ready_leads:")
        print("=" * 70)
        print(f"  source_lead_id            : {source_lead_id}")
        print(f"  username                  : {username}")
        print(f"  text                      : {original_text!r}")
        print(f"  msg_timestamp             : {timestamp}")
        print(f"  lead_type                 : psp_provider")
        print(f"  vertical                  : []")
        print(f"  geo                       : {geo}")
        print(f"  payment_methods_mentioned : {methods}")
        print(f"  approved_by               : None")
        print(f"  -- додаткові поля з LLM-екстракції (ще не в БД) --")
        print(f"  company                   : {extracted.get('company')}")
        print(f"  position                  : {extracted.get('position')}")
        print(f"  notes                     : {extracted.get('notes')}")
        print("=" * 70 + "\n")
        return

    async with conn_pool.acquire() as conn:
        row = await conn.fetchrow(
            _INSERT_CLIENT_READY_SQL,
            source_lead_id,
            username,
            original_text,
            timestamp,
            "psp_provider",                          # lead_type
            json.dumps([], ensure_ascii=False),       # vertical — not extracted here
            json.dumps(geo, ensure_ascii=False),
            json.dumps(methods, ensure_ascii=False),
            None,                                     # approved_by — set later by reviewer
            extracted.get("company"),
            extracted.get("position"),
            extracted.get("notes"),
        )

    print(f"psp_provider_classifier: persisted PSP provider lead id={row['id'] if row else None} username={username} company={extracted.get('company')!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_psp_provider_classifier(
    *,
    text: str,
    source_lead_id: Optional[int],
    message_id: Optional[int],
    username: Optional[str],
    timestamp: Optional[datetime],
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> bool:
    """Run the second-pass PSP-provider classifier on a message the primary
    pipeline rejected (is_lead=False).
    """
    text = (text or "").strip()
    if not text:
        print("[psp_provider_classifier] step 0: ПОРОЖНІЙ text — вихід одразу")
        return False
    print("[psp_provider_classifier] step 0: старт, text =", text[:50])

    # --- Step 1: keyword filter по оригінальному тексту ---------------------
    hits = _keyword_hits(text)
    print("[psp_provider_classifier] step 1: keyword hits =", hits)
    if hits == 0:
        print("[psp_provider_classifier] step 1: 0 збігів — вихід")
        logger.debug("psp_provider_classifier: no keyword hits, skipping message_id=%s", message_id)
        return False

    # --- Step 2: context (тільки якщо пройшов keyword-фільтр) ---------------
    print("[psp_provider_classifier] step 2: пробую acquire() з conn_pool...")
    context_texts: list[str] = []
    try:
        async with conn_pool.acquire() as conn:
            print("[psp_provider_classifier] step 2: conn отримано, виконую SELECT...")
            context_texts = await _load_author_context(username, conn)
    except Exception as exc:
        print(f"[psp_provider_classifier] step 2: ПОМИЛКА при читанні контексту: {exc!r}")
        raise
    print("[psp_provider_classifier] step 2: контекст завантажено, count =", len(context_texts))
    combined_text = _build_combined_text(text, context_texts)

    # --- Step 3: LLM verification --------------------------------------------
    print("[psp_provider_classifier] step 3: викликаю verify LLM, model =", _verify_model())
    author_line = username or "anon"
    verify_user_msg = f"[Автор]: {author_line}\n[Повідомлення]: {combined_text}"

    verify_result = await _call_openrouter(
        model=_verify_model(),
        system=_VERIFY_SYSTEM_PROMPT,
        user=verify_user_msg,
        max_tokens=200,
    )
    print("[psp_provider_classifier] step 3: результат verify =", verify_result)

    if verify_result.get("_error"):
        print("[psp_provider_classifier] step 3: помилка виклику — вихід")
        logger.warning(
            "psp_provider_classifier: verification call failed for message_id=%s, skipping", message_id
        )
        return False

    if verify_result.get("is_psp") is not True:
        print("[psp_provider_classifier] step 3: is_psp не True — вихід")
        logger.debug(
            "psp_provider_classifier: not a PSP provider (message_id=%s): %s",
            message_id,
            verify_result.get("reason"),
        )
        return False

    # --- Step 4: LLM field extraction ----------------------------------------
    print("[psp_provider_classifier] step 4: викликаю extract LLM, model =", _extract_model())
    extract_user_msg = f"[Автор]: {author_line}\n\n[Повідомлення]:\n{combined_text}"

    extract_result = await _call_openrouter(
        model=_extract_model(),
        system=_EXTRACT_SYSTEM_PROMPT,
        user=extract_user_msg,
        max_tokens=400,
    )
    print("[psp_provider_classifier] step 4: результат extract =", extract_result)

    if extract_result.get("_error"):
        print("[psp_provider_classifier] step 4: помилка виклику — вихід")
        logger.warning(
            "psp_provider_classifier: extraction call failed for message_id=%s, skipping", message_id
        )
        return False

    if extract_result.get("judged_as") != "provider":
        print("[psp_provider_classifier] step 4: judged_as != provider — вихід")
        logger.debug(
            "psp_provider_classifier: extraction stage downgraded verdict (message_id=%s): %s",
            message_id,
            extract_result.get("judged_as"),
        )
        return False

    # --- Step 5: persist -------------------------------------------------------
    print("[psp_provider_classifier] step 5: записую результат...")
    await _persist_provider_lead(
        source_lead_id=source_lead_id,
        username=username,
        original_text=text,
        timestamp=timestamp,
        extracted=extract_result,
        conn_pool=conn_pool,
        dry_run=dry_run,
    )
    print("[psp_provider_classifier] step 5: готово")
    return True