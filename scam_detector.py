"""Scam self-declaration detector.

Cheap pre-filter that runs BEFORE any other classifier in the webhook pipeline.

Pipeline:
    1. Keyword filter (local, free)
       → no hit: returns False immediately, no LLM call
    2. LLM binary classification (openai/gpt-4o-mini via OpenRouter)
       → is_scam=0: returns False, message continues to main pipeline
       → is_scam=1: writes to classified_messages_dirty, returns True (stop)

Entry point: check_scam(...) — called from webhook._run() right after
duplicate check, before classify_single_message.

Returns True  → message is a scam self-declaration, pipeline should stop.
Returns False → message is clean, pipeline continues normally.
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

load_dotenv()
logger = logging.getLogger(__name__)

_TABLE_PREFIX = os.getenv("TABLE_PREFIX") or ""
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "openai/gpt-4o-mini"

# ---------------------------------------------------------------------------
# Keyword filter — LLM is called only if at least one keyword is found.
# ---------------------------------------------------------------------------

_SCAM_KEYWORDS: list[str] = ["скам", "scam", "fraud", "мошен"]


def _has_scam_keyword(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SCAM_KEYWORDS)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a lead-quality filter. Your only task is to detect whether the author \
of a message EXPLICITLY states that their OWN project, product, or business is a scam, \
fraudulent, unlicensed by their own admission, or otherwise self-declared illegal.

Output ONLY a JSON object — no markdown, no explanation:
{"is_scam": 0 or 1, "reason": "one short sentence"}

Rules:
- is_scam = 1  ONLY if the author explicitly describes THEIR OWN project as a scam / fraud.
- is_scam = 1  if "scam" / "скам" appears as part of a list of project characteristics \
with no clear subject — treat it as a self-description \
(e.g. "скам, без документов, нужен p2p" or "no license, scam, high risk").
- is_scam = 0  if the word refers to a third party, competitor, or provider \
(e.g. "they are a scam", "этот PSP — скам").
- is_scam = 0  if the word is explicitly negated \
(e.g. "not a scam", "не скам", "это точно не скам").
- is_scam = 0  if there is any doubt."""


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return key


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(text: str) -> dict:
    """Call OpenRouter and return parsed JSON response."""
    payload = {
        "model": _MODEL,
        "max_tokens": 60,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Message:\n{text}"},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
            return json.loads(clean)
    except Exception as exc:
        logger.error("[scam_detector] LLM call failed: %s", exc)
        # On error — let message through, don't block the pipeline.
        return {"is_scam": 0, "reason": f"llm_error: {exc}"}


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

_INSERT_SQL = f"""
INSERT INTO {_TABLE_PREFIX}classified_messages_dirty (
    group_id, message_id, timestamp, username, text,
    is_lead, confidence, lead_type, intent, interest_level,
    vertical, geo, payment_methods_mentioned,
    evidence_quote, rationale, glossary_terms_seen, elapsed_ms,
    verdict, judge_reason,
    db_write_status, db_write_error,
    classified_at
) VALUES (
    $1,  $2,  $3,  $4,  $5,
    $6,  $7,  $8,  $9,  $10,
    $11, $12, $13,
    $14, $15, $16, $17,
    $18, $19,
    $20, $21,
    NOW()
)
ON CONFLICT ON CONSTRAINT {_TABLE_PREFIX}uq_group_message_leadtype DO UPDATE SET
    is_lead        = EXCLUDED.is_lead,
    rationale      = EXCLUDED.rationale,
    db_write_status = EXCLUDED.db_write_status,
    classified_at  = NOW()
RETURNING id;
"""


async def _persist_scam(
    *,
    group_id: Optional[int],
    message_id: Optional[int],
    timestamp: Optional[datetime],
    username: Optional[str],
    text: str,
    reason: str,
    conn_pool: asyncpg.Pool,
) -> None:
    try:
        async with conn_pool.acquire() as conn:
            row = await conn.fetchrow(
                _INSERT_SQL,
                group_id,
                message_id,
                timestamp,
                username,
                text,
                False,                          # is_lead
                1.0,                            # confidence
                None,                           # lead_type
                None,                           # intent
                None,                           # interest_level
                json.dumps([]),                 # vertical
                json.dumps([]),                 # geo
                json.dumps([]),                 # payment_methods_mentioned
                None,                           # evidence_quote
                f"[scam offer detected: {reason}]",  # rationale
                json.dumps([]),                 # glossary_terms_seen
                None,                           # elapsed_ms
                "",                             # verdict
                None,                           # judge_reason
                "ok",                           # db_write_status
                None,                           # db_write_error
            )
            logger.info(
                "[scam_detector] persisted scam message_id=%s db_id=%s reason=%r",
                message_id, row["id"] if row else None, reason,
            )
    except Exception as exc:
        logger.error("[scam_detector] DB write failed message_id=%s: %s", message_id, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def check_scam(
    *,
    text: str,
    group_id: Optional[int],
    message_id: Optional[int],
    timestamp: Optional[datetime],
    username: Optional[str],
    conn_pool: asyncpg.Pool,
) -> bool:
    """Return True if the message is a scam self-declaration (pipeline should stop).

    False means the message is clean — continue to the main classifier.
    """
    if not _has_scam_keyword(text):
        return False

    logger.info(
        "[scam_detector] keyword hit, calling LLM message_id=%s", message_id
    )

    result = await _call_llm(text)
    is_scam: int = result.get("is_scam", 0)
    reason: str  = result.get("reason", "")

    logger.info(
        "[scam_detector] LLM result message_id=%s is_scam=%s reason=%r",
        message_id, is_scam, reason,
    )

    if is_scam:
        await _persist_scam(
            group_id=group_id,
            message_id=message_id,
            timestamp=timestamp,
            username=username,
            text=text,
            reason=reason,
            conn_pool=conn_pool,
        )
        return True

    return False
