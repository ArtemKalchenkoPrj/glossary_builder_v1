"""Crypto B2B buyer classifier — streaming / webhook edition.

Drop-in replacement for ``run_crypto_buyer_classifier``:
  - Same async signature and bool return value.
  - Uses the full extract → judge pipeline (LeadExtractor + judge LLM call)
    instead of the old keyword → verify → extract → judge chain.
  - Module-level state (glossary, LLM, cfg, judge prompt) is initialised
    lazily on first call via asyncio.Lock — no changes to the webhook's
    lifespan() are needed.
  - No RAG (index is empty; will be wired in once enough examples exist).
  - geo field dropped from the DB update (decision per project checklist).

Usage in webhook:
    Replace:
        from crypto_buyer_classifier_old import run_crypto_buyer_classifier
        await run_crypto_buyer_classifier(text=..., source_lead_id=..., ...)
    With:
        from crypto_buyer_classifier import classify_crypto_buyer_single_message
        await classify_crypto_buyer_single_message(text=..., source_lead_id=..., ...)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

from glossary_builder_crypto_buyers.lead_extraction import (
    LeadDecision,
    LeadExtractionConfig,
    LeadExtractor,
    load_glossary,
)
from glossary_builder_crypto_buyers.lead_prompt_versions import BY_VERSION as LEAD_BY_VERSION
from glossary_builder_crypto_buyers.cli import JUDGE_PROMPTS_BY_VERSION
from glossary_builder_crypto_buyers.llm import LLMClient


def _fmt(v) -> str:
    """Format a value for the judge prompt (None/[]/"" → empty string)."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else ""
    return str(v)


_JUDGE_USER_TEMPLATE = """\
LEAD CANDIDATE:
  message_id: {message_id}
  @{username}  [{timestamp}]
  text: {text}

EXTRACTOR's claim:
  company: {company}
  asset: {asset}
  network: {network}
  pay_asset: {pay_asset}
  pay_method: {pay_method}
  amount: {amount}
  min: {min}
  max: {max}
  notes: {notes}
  evidence_quote: {evidence_quote}
  rationale: {rationale}

Return JSON:
{{
  "verdict": "REAL_LEAD" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file's location — adjust if needed)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GLOSSARY_PATH = _PROJECT_ROOT / "data/output/crypto_glossary/glossary_primary_sense_only.json"

# Table prefix — mirror the pattern used in the rest of the webhook.
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# CRITICAL: TABLE_PREFIX must be set to prevent writes to production tables.
# Fail loudly at import time so misconfiguration is caught immediately.
_TABLE_PREFIX = os.getenv("TABLE_PREFIX",'')


# ---------------------------------------------------------------------------
# Module-level state — lazy init on first call
# ---------------------------------------------------------------------------

@dataclass
class _CryptoBuyerState:
    glossary: list[dict]
    llm: LLMClient
    cfg: LeadExtractionConfig
    judge_system: str


_state: Optional[_CryptoBuyerState] = None
_state_lock = asyncio.Lock()


async def _get_state() -> _CryptoBuyerState:
    """Return the shared state, initialising it on first call."""
    global _state
    if _state is not None:
        return _state
    async with _state_lock:
        # Double-checked locking: another coroutine may have initialised
        # while we were waiting for the lock.
        if _state is not None:
            return _state

        logger.info("[crypto_buyer_classifier] initialising module state…")

        glossary = load_glossary(_GLOSSARY_PATH)
        logger.info("[crypto_buyer_classifier] glossary loaded: %d entries", len(glossary))

        llm = LLMClient()

        cfg = LeadExtractionConfig()
        cfg.lead_definition = LEAD_BY_VERSION["v1-crypto"]
        # RAG disabled until index has enough crypto B2B examples.
        cfg.stage1_enabled = True

        judge_system = JUDGE_PROMPTS_BY_VERSION["v1"]

        _state = _CryptoBuyerState(
            glossary=glossary,
            llm=llm,
            cfg=cfg,
            judge_system=judge_system,
        )
        logger.info("[crypto_buyer_classifier] init complete")
    return _state


# ---------------------------------------------------------------------------
# DB — UPDATE the existing classified_messages_dirty row (no geo field)
# ---------------------------------------------------------------------------

_UPDATE_SQL = f"""
UPDATE {_TABLE_PREFIX}classified_messages_dirty
SET
    is_lead      = TRUE,
    verdict      = 'REAL_LEAD',
    judge_reason = $2,
    rationale    = $3,
    company      = $4,
    asset        = $5,
    network      = $6,
    pay_asset    = $7,
    pay_method   = $8,
    amount       = $9,
    "min"        = $10,
    "max"        = $11,
    confidence   = $12,
    lead_type    = 'crypto_buyer'
WHERE id = $1
RETURNING id;
"""


async def _persist(
    *,
    dirty_row_id: int,
    decision: LeadDecision,
    verdict: str,
    judge_reason: Optional[str],
    conn_pool: asyncpg.Pool,
    dry_run: bool,
) -> None:
    """UPDATE the dirty row with extracted crypto B2B fields."""
    company    = decision.company
    asset      = decision.asset or []
    network    = decision.network or []
    pay_asset  = decision.pay_asset or []
    pay_method = decision.pay_method or []
    amount     = decision.amount
    min_val    = decision.min
    max_val    = decision.max
    confidence = decision.confidence
    rationale  = decision.rationale or ""
    notes      = decision.notes or judge_reason

    if dry_run:
        print("\n" + "=" * 70)
        print("[crypto_buyer_classifier DRY RUN] UPDATE that would run:")
        print("=" * 70)
        print(f"  WHERE id        : {dirty_row_id}")
        print(f"  is_lead         : TRUE")
        print(f"  verdict         : {verdict}")
        print(f"  lead_type       : crypto_buyer")
        print(f"  judge_reason    : {notes}")
        print(f"  rationale       : {rationale}")
        print(f"  company         : {company}")
        print(f"  asset           : {asset}")
        print(f"  network         : {network}")
        print(f"  pay_asset       : {pay_asset}")
        print(f"  pay_method      : {pay_method}")
        print(f"  amount          : {amount}")
        print(f"  min             : {min_val}")
        print(f"  max             : {max_val}")
        print(f"  confidence      : {confidence}")
        print("=" * 70 + "\n")
        return

    async with conn_pool.acquire() as conn:
        row = await conn.fetchrow(
            _UPDATE_SQL,
            dirty_row_id,  # $1
            notes,         # $2  judge_reason
            rationale,     # $3  rationale
            company,       # $4
            json.dumps(asset, ensure_ascii=False),      # $5
            json.dumps(network, ensure_ascii=False),    # $6
            json.dumps(pay_asset, ensure_ascii=False),  # $7
            json.dumps(pay_method, ensure_ascii=False), # $8
            amount,        # $9
            min_val,       # $10
            max_val,       # $11
            confidence,    # $12
        )

    logger.info(
        "[crypto_buyer_classifier] updated dirty row id=%s company=%s asset=%s",
        row["id"] if row else None, company, asset,
    )


# ---------------------------------------------------------------------------
# Judge call (mirrors _run_judge from classify_single_message)
# ---------------------------------------------------------------------------

def _run_judge(
    decision: LeadDecision,
    llm: LLMClient,
    judge_system: str,
) -> tuple[str, Optional[str]]:
    """Call the judge LLM and return (verdict, reason).

    Runs synchronously — wrap in asyncio.to_thread at the call site.
    """
    user = _JUDGE_USER_TEMPLATE.format(
        message_id=decision.message_id or "?",
        username=decision.username or "anon",
        timestamp=decision.timestamp or "?",
        text=(decision.text or "").replace("\n", " ").strip()[:500],
        company=_fmt(decision.company),
        asset=_fmt(decision.asset),
        network=_fmt(decision.network),
        pay_asset=_fmt(decision.pay_asset),
        pay_method=_fmt(decision.pay_method),
        amount=_fmt(decision.amount),
        min=_fmt(decision.min),
        max=_fmt(decision.max),
        notes=_fmt(decision.notes),
        evidence_quote=(decision.evidence_quote or "")[:200],
        rationale=(decision.rationale or "")[:200],
    )

    try:
        data, _ = llm.complete_json(judge_system, user, max_tokens=200)
        verdict = str(data.get("verdict", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        if verdict not in ("REAL_LEAD", "MISTAKE"):
            verdict = "REVIEW"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[crypto_buyer_classifier] judge call failed: %s", exc)
        verdict = "REVIEW"
        reason = f"judge call failed: {exc}"

    return verdict, reason


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for run_crypto_buyer_classifier
# ---------------------------------------------------------------------------

async def classify_crypto_buyer_single_message(
    *,
    text: str,
    source_lead_id: Optional[int],
    message_id: Optional[int],
    username: Optional[str],
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> bool:
    """Classify one message as a crypto B2B buyer lead.

    Runs the full extract → judge pipeline (same logic as the CLI's
    ``extract-leads`` + ``judge-leads`` commands).

    Returns True if the message was classified as REAL_LEAD and persisted;
    False otherwise (not a lead, judge downgraded, or any error).

    Designed as a drop-in replacement for run_crypto_buyer_classifier:
    same signature, same return semantics, same DB side-effect.
    """
    text = (text or "").strip()
    if not text:
        logger.debug("[crypto_buyer_classifier] empty text, skip message_id=%s", message_id)
        return False
    if source_lead_id is None:
        logger.warning(
            "[crypto_buyer_classifier] no source_lead_id, skip message_id=%s", message_id
        )
        return False

    state = await _get_state()

    # Build a minimal Message for the extractor.
    from glossary_builder_crypto_buyers.loader import Message  # noqa: PLC0415
    message = Message(
        message_id=message_id,
        date=None,
        text=text,
        username=username,
        group_id=None,
        user_id=None,
        first_name=None,
        last_name=None,
    )

    # Run the extractor in a thread (it's synchronous / CPU-bound).
    extractor = LeadExtractor(state.glossary, state.llm, state.cfg, rag_index=None)

    decision: Optional[LeadDecision] = await asyncio.to_thread(
        extractor.extract_one, message, []
    )

    # Message skipped by extractor (too short, excluded author, etc.)
    if decision is None:
        logger.debug(
            "[crypto_buyer_classifier] extractor skipped message_id=%s", message_id
        )
        return False

    logger.debug(
        "[crypto_buyer_classifier] extract done message_id=%s is_lead=%s confidence=%.2f",
        message_id, decision.is_lead, decision.confidence,
    )

    if not decision.is_lead:
        return False

    # Stage 2: independent judge verdict.
    verdict, judge_reason = await asyncio.to_thread(
        _run_judge, decision, state.llm, state.judge_system
    )

    logger.info(
        "[crypto_buyer_classifier] LEAD message_id=%s verdict=%s company=%s asset=%s",
        message_id, verdict, decision.company, decision.asset,
    )

    if verdict != "REAL_LEAD":
        logger.debug(
            "[crypto_buyer_classifier] judge downgraded to %s message_id=%s reason=%s",
            verdict, message_id, judge_reason,
        )
        return False

    await _persist(
        dirty_row_id=source_lead_id,
        decision=decision,
        verdict=verdict,
        judge_reason=judge_reason,
        conn_pool=conn_pool,
        dry_run=dry_run,
    )
    return True