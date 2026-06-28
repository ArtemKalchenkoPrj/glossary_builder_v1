"""Casino platform seeker classifier — webhook-side pipeline.

Adapted from psp_provider_classifier.py.
Looks for people who want to LAUNCH their own casino or iGaming platform
(buyers), as opposed to PSP providers who pitch their services.

Pipeline (two LLM stages):

    1. Keyword filter (local, free)
    2. LLM verification (GLOSSARY_MODEL)  → is_lead true/false
       -> if is_lead != true: stop, nothing persisted
    3. LLM field extraction (JUDGE_MODEL) → verdict + geo, platform, vertical, notes
       -> if verdict != REAL_LEAD: stop, nothing persisted

No author-context logic (unlike psp_provider_classifier.py) —
each message is classified independently.

Entry point: run_casino_classifier(...), called from webhook._run().
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

import asyncpg
import httpx
from json_repair import repair_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _api_key() -> str:
    key = os.environ.get("PSP_PROVIDERS_OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "PSP_PROVIDERS_OPENROUTER_API_KEY not set in .env — "
            "casino_classifier cannot call OpenRouter without it."
        )
    return key


def _verify_model() -> str:
    return os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")


def _extract_model() -> str:
    return os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Stage 1 — keyword filter
# Broad by design: catches any casino/iGaming context.
# Stage 2 LLM then filters down to actual platform seekers.
# ---------------------------------------------------------------------------

_CASINO_KEYWORDS = [
    # Прямое название продукта
    r"\bcasino\b",
    r"\bказин[оуіа]\b",
    r"\bигровой\s+клуб\b",
    # iGaming / gambling
    r"\bigaming\b",
    r"\bi[-\s]gaming\b",
    r"\bgambling\b",
    r"\bгемблинг\w*\b",
    r"\bгамблинг\w*\b",
    # Беттинг
    r"\bбукмекер\w*\b",
    r"\bbookmaker\w*\b",
    r"\bbetting\b",
    r"\bбеттинг\w*\b",
    # White label / платформа в контексте запуска
    r"\bwhite.?label\b",
    r"\bплатформ\w+.{0,40}(казин|igaming|gambling|беттинг)\w*",
    r"\b(казин|igaming|gambling|беттинг)\w*.{0,40}платформ\w+",
    r"\bplatform\w*.{0,40}(casino|igaming|gambling|betting)\w*",
    r"\b(casino|igaming|gambling|betting)\w*.{0,40}platform\w*",
    # Фразы запуска
    r"\bоткрыт\w+\s+казин\w*\b",
    r"\bзапуст\w+\s+казин\w*\b",
    r"\bзапуст\w+\s+(казино|iGaming|gambling)\b",
    r"\blaunch\w*\s+casino\b",
    r"\bopen\w*\s+casino\b",
    r"\bрозгорнут\w+\s+казин\w*\b",
    r"\bсделать\s+казин\w*\b",
    r"\bсоздат\w+\s+казин\w*\b",
    # ПО / поставщики
    r"\bcasino\s+software\b",
    r"\bigaming\s+software\b",
    r"\bcasino\s+solution\w*\b",
    r"\bgame.?provider\w*\b",
    r"\bпровайдер.{0,10}игр\w*\b",
    r"\bпровайдер.{0,10}контент\w*\b",
    r"\bсофт.{0,15}казин\w*\b",
    r"\bказин\w*.{0,15}софт\b",
    # Лицензия в контексте бизнеса
    r"\bлиценз\w+.{0,20}(казин|igaming|gambling)\w*\b",
    r"\b(казин|igaming|gambling)\w*.{0,20}лиценз\w+\b",
    r"\bcasino\s+licen[sc]e\b",
]

_COMPILED_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in _CASINO_KEYWORDS]


def _keyword_hits(text: str) -> int:
    return sum(1 for pattern in _COMPILED_KEYWORDS if pattern.search(text))


# ---------------------------------------------------------------------------
# Stage 2 — LLM verification prompt
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = """\
Ты — аналитик рынка iGaming. Определи, ищет ли автор сообщения платформу или решение, \
чтобы САМОМУ запустить казино или iGaming-продукт.

Признаки потенциального заказчика (is_lead: true):
- Прямо заявляет о намерении открыть / запустить / развернуть казино или ставочный продукт
- Ищет white-label платформу для казино или беттинга
- Спрашивает про software / CMS / поставщика ПО для iGaming
- Ищет технического партнёра или вендора для запуска проекта
- Обсуждает получение лицензии с целью открыть собственный продукт
- Ищет game provider для своей будущей платформы

НЕ является потенциальным заказчиком (is_lead: false):
- Игрок, который ищет где поиграть или получить бонус
- Человек, который обсуждает рынок, регуляцию или читает новости
- Аффилиат или SEO-специалист без явного намерения запустить собственную платформу
- Провайдер / поставщик, который рекламирует свои услуги (он продаёт, а не покупает)
- Человек, который просто упоминает казино в бытовом контексте

Отвечай ТОЛЬКО в формате JSON (без markdown, без пояснений):
{
  "is_lead": true или false,
  "confidence": "high" | "medium" | "low",
  "reason": "одно предложение — краткое обоснование решения"
}"""


# ---------------------------------------------------------------------------
# Stage 3 — LLM field extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """\
Ты — старший аналитик рынка iGaming. Тебе дают сообщение из Telegram-чата.

Задачи:
1. Вынести финальный вердикт — является ли автор реальным потенциальным заказчиком \
платформы для казино / iGaming
2. Извлечь доступную мета-информацию

REAL_LEAD — автор явно ищет платформу / software / white-label или партнёра \
для ЗАПУСКА казино / iGaming
MISTAKE   — это не заказчик: игрок, провайдер, обсуждение рынка, реклама и т.п.

Верни ТОЛЬКО валидный JSON (без markdown, без пояснений):
{
  "verdict": "REAL_LEAD" | "MISTAKE",
  "geo": ["целевые страны или регионы автора — или []"],
  "platform": ["названия платформ / CMS / software, которые упоминаются — или []"],
  "vertical": ["casino" | "betting" | "poker" | "esports" | "crypto" | другое — или []"],
  "notes": "важный контекст одним предложением — или null"
}

Правила:
- Если поле неизвестно — [] или null, не придумывай
- geo, platform, vertical — массивы даже если один элемент
- Названия оставляй как в тексте, не переводи
- notes — максимум одно предложение"""


# ---------------------------------------------------------------------------
# OpenRouter call helper
# ---------------------------------------------------------------------------

async def _call_openrouter(model: str, system: str, user: str, max_tokens: int) -> dict:
    """POST one chat-completion request to OpenRouter, return parsed JSON.

    Uses json_repair to handle malformed JSON responses from the model.
    On any failure returns {"_error": ...} so callers can branch without try/except.
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
                },
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```json\s*|```", "", raw).strip()
            return json.loads(repair_json(clean))

    except Exception as exc:
        logger.error("casino_classifier: OpenRouter call failed (model=%s): %s", model, exc)
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO client_ready_leads (
    source_lead_id, username, text, msg_timestamp,
    lead_type, vertical, geo, platform, approved_by, notes
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7, $8, $9, $10
)
RETURNING id;
"""


async def _persist_casino_lead(
    *,
    source_lead_id: Optional[int],
    username: Optional[str],
    original_text: str,
    timestamp: Optional[datetime],
    extracted: dict,
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> None:
    """Write a confirmed casino platform seeker lead to client_ready_leads.

    If dry_run=True, prints the row that would have been inserted instead
    of touching the database.
    """
    geo      = extracted.get("geo") or []
    platform = extracted.get("platform") or []
    vertical = extracted.get("vertical") or []
    notes    = extracted.get("notes")

    if dry_run:
        print("\n" + "=" * 70)
        print("[casino_classifier DRY RUN] row that would be inserted:")
        print("=" * 70)
        print(f"  source_lead_id : {source_lead_id}")
        print(f"  username       : {username}")
        print(f"  text           : {original_text!r}")
        print(f"  msg_timestamp  : {timestamp}")
        print(f"  lead_type      : casino_platform_seeker")
        print(f"  vertical       : {vertical}")
        print(f"  geo            : {geo}")
        print(f"  platform       : {platform}")
        print(f"  approved_by    : None")
        print(f"  notes          : {notes}")
        print("=" * 70 + "\n")
        return

    async with conn_pool.acquire() as conn:
        row = await conn.fetchrow(
            _INSERT_SQL,
            source_lead_id,
            username,
            original_text,
            timestamp,
            "seeking_casino_platform",
            json.dumps(vertical, ensure_ascii=False),
            json.dumps(geo, ensure_ascii=False),
            json.dumps(platform, ensure_ascii=False),
            "huyochok",
            notes,
        )

    logger.info(
        "casino_classifier: persisted lead id=%s username=%s vertical=%s",
        row["id"] if row else None, username, vertical,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_casino_classifier(
    *,
    text: str,
    source_lead_id: Optional[int],
    message_id: Optional[int],
    username: Optional[str],
    timestamp: Optional[datetime],
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> None:
    """Run the casino platform seeker classifier on an incoming message."""

    text = (text or "").strip()
    if not text:
        logger.debug("casino_classifier: empty text, skipping message_id=%s", message_id)
        return
    logger.debug("casino_classifier: start, message_id=%s text=%.50r", message_id, text)

    # --- Step 1: keyword filter -----------------------------------------------
    hits = _keyword_hits(text)
    logger.debug("casino_classifier: keyword hits=%d message_id=%s", hits, message_id)
    if hits == 0:
        logger.debug("casino_classifier: no keyword hits, skipping message_id=%s", message_id)
        return

    # --- Step 2: LLM verification ---------------------------------------------
    logger.debug("casino_classifier: calling verify model=%s", _verify_model())

    verify_result = await _call_openrouter(
        model=_verify_model(),
        system=_VERIFY_SYSTEM_PROMPT,
        user=text,
        max_tokens=200,
    )
    logger.debug("casino_classifier: verify result=%s", verify_result)

    if verify_result.get("_error"):
        logger.warning(
            "casino_classifier: verification failed for message_id=%s: %s",
            message_id, verify_result["_error"],
        )
        return

    if verify_result.get("is_lead") is not True:
        logger.debug(
            "casino_classifier: not a lead (message_id=%s): %s",
            message_id, verify_result.get("reason"),
        )
        return

    # --- Step 3: LLM field extraction -----------------------------------------
    logger.debug("casino_classifier: calling extract model=%s", _extract_model())

    extract_result = await _call_openrouter(
        model=_extract_model(),
        system=_EXTRACT_SYSTEM_PROMPT,
        user=text,
        max_tokens=400,
    )
    logger.debug("casino_classifier: extract result=%s", extract_result)

    if extract_result.get("_error"):
        logger.warning(
            "casino_classifier: extraction failed for message_id=%s: %s",
            message_id, extract_result["_error"],
        )
        return

    if extract_result.get("verdict") != "REAL_LEAD":
        logger.debug(
            "casino_classifier: judge downgraded verdict (message_id=%s): %s",
            message_id, extract_result.get("verdict"),
        )
        return

    # --- Step 4: persist ------------------------------------------------------
    await _persist_casino_lead(
        source_lead_id=source_lead_id,
        username=username,
        original_text=text,
        timestamp=timestamp,
        extracted=extract_result,
        conn_pool=conn_pool,
        dry_run=dry_run,
    )
