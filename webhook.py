"""FastAPI webhook for single-message lead classification.

Two parallel pipelines:
    1. Buyer pipeline  — classify_single_message → cascade (crypto, casino, iban)
    2. Provider pipeline — classify_provider_message (glossary-based)

Exposes two endpoints:
    POST /classify  — classify a message and persist the result to PostgreSQL
    GET  /health    — liveness check

Run locally:
    uvicorn webhook:app --host 0.0.0.0 --port 8000 --reload

Required .env variables:
    POSTGRES_DSN=postgresql://user:password@host:5432/dbname
    PSP_PROVIDERS_OPENROUTER_API_KEY=sk-or-...  (for provider pipeline)
    PROVIDER_STAGE1_MODEL=openai/gpt-4.1-mini   (optional)
    PROVIDER_JUDGE_MODEL=openai/gpt-4.1-mini    (optional)

Migration (run once):
    ALTER TABLE classified_messages_dirty ADD COLUMN IF NOT EXISTS company TEXT DEFAULT NULL;
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Casino_platforms_classifier.casino_classifier import run_casino_classifier
from glossary_builder_crypto_buyers.crypto_buyer_classifier import classify_crypto_buyer_single_message
from IBAN_classifier.iban_lead_classifier import run_iban_lead_classifier
from glossary_builder.cli import JUDGE_PROMPTS_BY_VERSION
from glossary_builder.lead_extraction import LeadExtractionConfig, load_glossary
from glossary_builder.lead_prompt_versions import BY_VERSION
from glossary_builder.llm import LLMClient
from glossary_builder.loader import Message
from glossary_builder.rag import RagIndex, RagConfig
from glossary_builder.single_classifier import classify_single_message
from glossary_builder.geo_pipeline import process_geo
from Deduper.deduplication import is_duplicate

# Provider pipeline imports
from PSP_providers_glossary_builder.provider_classifier import (
    classify_provider_message,
    load_provider_glossary,
    make_provider_llm_client,
)
from PSP_providers_glossary_builder.provider_decision import ProviderExtractionConfig

_TABLE_PREFIX = os.getenv('TABLE_PREFIX') or ""
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# ---------------------------------------------------------------------------
# SQL — buyer upsert (updated with company column)
# ---------------------------------------------------------------------------

_UPSERT_SQL = f"""
INSERT INTO {_TABLE_PREFIX}classified_messages_dirty (
    group_id, message_id, timestamp, username, text,
    is_lead, confidence, lead_type, intent, interest_level,
    vertical, geo, payment_methods_mentioned,
    evidence_quote, rationale, glossary_terms_seen, elapsed_ms,
    verdict, judge_reason, company, position,
    pipeline,
    db_write_status, db_write_error,
    classified_at
) VALUES (
    $1,  $2,  $3,  $4,  $5,
    $6,  $7,  $8,  $9,  $10,
    $11, $12, $13,
    $14, $15, $16, $17,
    $18, $19, $20, $21,
    $22,
    $23, $24,
    NOW()
)
ON CONFLICT ON CONSTRAINT {_TABLE_PREFIX}uq_group_message_leadtype DO UPDATE SET
    timestamp                 = EXCLUDED.timestamp,
    username                  = EXCLUDED.username,
    text                      = EXCLUDED.text,
    is_lead                   = EXCLUDED.is_lead,
    confidence                = EXCLUDED.confidence,
    lead_type                 = EXCLUDED.lead_type,
    intent                    = EXCLUDED.intent,
    interest_level            = EXCLUDED.interest_level,
    vertical                  = EXCLUDED.vertical,
    geo                       = EXCLUDED.geo,
    payment_methods_mentioned = EXCLUDED.payment_methods_mentioned,
    evidence_quote            = EXCLUDED.evidence_quote,
    rationale                 = EXCLUDED.rationale,
    glossary_terms_seen       = EXCLUDED.glossary_terms_seen,
    elapsed_ms                = EXCLUDED.elapsed_ms,
    verdict                   = EXCLUDED.verdict,
    judge_reason              = EXCLUDED.judge_reason,
    company                   = EXCLUDED.company,
    position                  = EXCLUDED.position,
    pipeline                  = EXCLUDED.pipeline,
    db_write_status           = EXCLUDED.db_write_status,
    db_write_error            = EXCLUDED.db_write_error,
    classified_at             = NOW()
RETURNING id;
"""


# ---------------------------------------------------------------------------
# App state — loaded once at startup, reused across all requests.
# ---------------------------------------------------------------------------

class _AppState:
    # Buyer pipeline
    glossary: list[dict]
    llm: LLMClient
    cfg: LeadExtractionConfig
    rag_index: RagIndex
    judge_system: str

    # Provider pipeline
    provider_glossary: list[dict]
    provider_llm: LLMClient
    provider_cfg: ProviderExtractionConfig

    # Shared
    pool: asyncpg.Pool

_state = _AppState()


def _log_startup_config() -> None:
    """Print a human-readable startup config summary."""
    from PSP_providers_glossary_builder.provider_prompt import (
        _PROVIDER_STAGE1_SYSTEM,
        _PROVIDER_JUDGE_SYSTEM,
        PROVIDER_DEFINITION,
    )

    sep = "=" * 65

    logger.info(sep)
    logger.info("  WEBHOOK STARTUP CONFIG")
    logger.info(sep)

    # ── Database ──────────────────────────────────────────────────────────
    logger.info("  DB prefix      : %r", _TABLE_PREFIX or "(none)")
    logger.info("  Constraint     : %suq_group_message_leadtype", _TABLE_PREFIX)

    # ── Buyer pipeline ────────────────────────────────────────────────────
    logger.info(sep)
    logger.info("  BUYER PIPELINE")
    logger.info("  LLM provider   : %s", _state.llm.provider)
    logger.info("  LLM model      : %s", _state.llm.model)
    logger.info("  Glossary       : %d entries", len(_state.glossary))
    logger.info("  RAG index      : %d examples", _state.rag_index.count())
    logger.info("  Pre-filter DB  : %s", _state.cfg.pre_filter_db_path)
    logger.info("  Lead definition: %s...", _state.cfg.lead_definition[:120].replace("\n", " "))

    # ── Provider pipeline ─────────────────────────────────────────────────
    logger.info(sep)
    logger.info("  PROVIDER PIPELINE")
    logger.info("  LLM provider   : %s", _state.provider_llm.provider)
    logger.info("  LLM model      : %s", _state.provider_llm.model)
    logger.info(
        "  Stage1 model   : %s",
        os.environ.get("PROVIDER_STAGE1_MODEL", "(same as provider LLM)"),
    )
    logger.info(
        "  Judge model    : %s",
        os.environ.get("PROVIDER_JUDGE_MODEL", "(same as provider LLM)"),
    )
    logger.info("  Glossary       : %d entries", len(_state.provider_glossary))
    logger.info("  Pre-filter     : %s", _state.provider_cfg.pre_filter_enabled)
    logger.info("  Pre-filter DB  : %s", _state.provider_cfg.pre_filter_db_path)
    logger.info("  Stage1         : %s", _state.provider_cfg.stage1_enabled)
    logger.info("  Judge          : %s", _state.provider_cfg.judge_enabled)

    # ── Prompt previews ───────────────────────────────────────────────────
    logger.info(sep)
    logger.info("  PROMPT PREVIEWS (first 200 chars)")
    logger.info(
        "  Stage1 system  : %s...",
        _PROVIDER_STAGE1_SYSTEM[:200].replace("\n", " "),
    )
    logger.info(
        "  Judge system   : %s...",
        _PROVIDER_JUDGE_SYSTEM[:200].replace("\n", " "),
    )
    logger.info(
        "  Provider def   : %s...",
        PROVIDER_DEFINITION[:200].replace("\n", " "),
    )

    logger.info(sep)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources on startup; release on shutdown."""
    project_root = Path(__file__).parent

    # ── Buyer pipeline resources ──────────────────────────────────────────
    _state.glossary = load_glossary(
        project_root / "data/output/glossary_primary_sense_only.json"
    )
    rag_cfg = RagConfig(
        db_path=str(project_root / "data/rag/chroma"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    _state.rag_index = RagIndex(rag_cfg)
    if _state.rag_index.count() == 0:
        raise RuntimeError("RAG index is empty — run build-rag first")
    logger.info("[startup] rag=enabled examples=%d", _state.rag_index.count())

    _state.llm = LLMClient()
    _state.cfg = LeadExtractionConfig()
    _state.cfg.lead_definition = BY_VERSION["v5-01-short"]
    logger.info("[startup] prompt_version=v5-01-short")
    _state.cfg.pre_filter_db_path = str(project_root / "data/output/glossary.db")
    _state.judge_system = JUDGE_PROMPTS_BY_VERSION["v2"]
    logger.info("[startup] judge_version=v2")

    # ── Provider pipeline resources ───────────────────────────────────────
    provider_db_path = str(
        project_root / "PSP_providers_glossary_builder/data/provider_glossary.db"
    )
    _state.provider_glossary = load_provider_glossary(provider_db_path)
    logger.info(
        "[startup] provider glossary loaded: %d terms from %s",
        len(_state.provider_glossary), provider_db_path,
    )

    try:
        _state.provider_llm = make_provider_llm_client()
        logger.info("[startup] provider LLM: OpenRouter (separate key)")
    except ValueError:
        _state.provider_llm = LLMClient()
        logger.warning("[startup] provider LLM: fallback to main LLMClient")

    _state.provider_cfg = ProviderExtractionConfig(
        pre_filter_db_path=provider_db_path,
    )
    logger.info(
        "[startup] provider pipeline ready: pre_filter=%s stage1=%s judge=%s",
        _state.provider_cfg.pre_filter_enabled,
        _state.provider_cfg.stage1_enabled,
        _state.provider_cfg.judge_enabled,
    )

    # ── PostgreSQL connection pool ────────────────────────────────────────
    dsn = os.environ["POSTGRES_DSN"]
    _state.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

    logger.info(
        "Startup complete — buyer glossary: %d, provider glossary: %d",
        len(_state.glossary), len(_state.provider_glossary),
    )

    _log_startup_config()

    yield

    await _state.pool.close()


app = FastAPI(
    title="Lead Classifier",
    description="Parallel buyer + provider classification via glossary pipelines.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ContextMessage(BaseModel):
    """One surrounding message passed as context."""

    message_id: Optional[int] = None
    date: Optional[datetime] = None
    text: Optional[str] = None
    username: Optional[str] = None

    def to_message(self) -> Message:
        return Message(
            message_id=self.message_id,
            date=self.date,
            text=self.text,
            username=self.username,
            group_id=None,
            user_id=None,
            first_name=None,
            last_name=None,
        )


class ClassifyRequest(BaseModel):
    """Body for POST /classify."""

    text: str = Field(..., description="Raw message text to classify.")
    group_id: Optional[int] = Field(None, description="Telegram group / chat identifier.")
    message_id: Optional[int] = Field(None, description="Upstream message identifier.")
    username: Optional[str] = Field(None, description="Sender handle.")
    timestamp: Optional[datetime] = Field(None, description="Message datetime (ISO 8601).")
    context: Optional[list[ContextMessage]] = Field(
        None,
        description=(
            "Optional surrounding messages (e.g. a few before/after). "
            "When omitted the classifier runs with no surrounding context."
        ),
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _jsonb(value) -> str:
    """Serialise list fields to JSON strings for asyncpg JSONB params."""
    return json.dumps(value, ensure_ascii=False)


def _parse_dt(value) -> Optional[datetime]:
    """Convert ISO string timestamp from classifier output back to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


async def _persist_buyer(group_id: Optional[int], result: dict) -> dict:
    """Write buyer classification result to PostgreSQL."""
    try:
        async with _state.pool.acquire() as conn:
            result["geo"] = await process_geo(
                message_id=result.get("message_id"),
                raw_geo_list=result.get("geo") or [],
                payment_methods=result.get("payment_methods_mentioned") or [],
                conn=conn,
            )
            row = await conn.fetchrow(
                _UPSERT_SQL,
                group_id,  # $1
                result.get("message_id"),  # $2
                _parse_dt(result.get("timestamp")),  # $3
                result.get("username"),  # $4
                result.get("text"),  # $5
                result.get("is_lead"),  # $6
                result.get("confidence"),  # $7
                result.get("lead_type") or "none",  # $8
                result.get("intent"),  # $9
                result.get("interest_level"),  # $10
                _jsonb(result.get("vertical") or []),  # $11
                _jsonb(result.get("geo") or []),  # $12
                _jsonb(result.get("payment_methods_mentioned") or []),  # $13
                result.get("evidence_quote"),  # $14
                result.get("rationale"),  # $15
                _jsonb(result.get("glossary_terms_seen") or []),  # $16
                result.get("elapsed_ms"),  # $17
                result.get("verdict"),  # $18
                result.get("judge_reason"),  # $19
                None,  # $20 company
                None,  # $21 position
                "buyer",  # $22 pipeline
                "ok",  # $23 db_write_status
                None,  # $24 db_write_error
            )
        result["db_write_status"] = "ok"
        result["db_write_error"] = None
        result["db_id"] = row["id"] if row else None

    except Exception as exc:
        error_msg = str(exc)
        logger.error("DB write failed for message_id=%s: %s", result.get("message_id"), error_msg)
        result["db_write_status"] = "failed"
        result["db_write_error"] = error_msg

    return result


async def _persist_provider(group_id: Optional[int], result: dict, source_lead_id: Optional[int] = None) -> dict:
    """Write provider classification result to PostgreSQL.

    Reuses the same classified_messages_dirty table with lead_type='psp_provider'.
    Only writes confirmed providers (is_provider=True, verdict != MISTAKE).
    """
    is_provider = result.get("is_provider")
    verdict = result.get("verdict")

    if not is_provider or verdict == "MISTAKE":
        result["db_write_status"] = "skipped"
        result["db_write_error"] = None
        result["db_id"] = None
        return result

    # Parse list fields from comma-separated strings back to lists
    def _to_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.strip():
            return [x.strip() for x in val.split(",") if x.strip()]
        return []

    try:
        async with _state.pool.acquire() as conn:
            geo_list = _to_list(result.get("geo"))
            methods_list = _to_list(result.get("methods"))

            geo_processed = await process_geo(
                message_id=result.get("message_id"),
                raw_geo_list=geo_list,
                payment_methods=methods_list,
                conn=conn,
            )

            row = await conn.fetchrow(
                _UPSERT_SQL,
                group_id,  # $1
                result.get("message_id"),  # $2
                _parse_dt(result.get("timestamp")),  # $3
                result.get("username"),  # $4
                result.get("text"),  # $5
                True,  # $6  is_lead
                result.get("confidence"),  # $7
                "psp_provider",  # $8  lead_type
                None,  # $9  intent
                None,  # $10 interest_level
                _jsonb(_to_list(result.get("vertical"))),  # $11
                _jsonb(geo_processed or geo_list),  # $12
                _jsonb(methods_list),  # $13
                result.get("evidence_quote"),  # $14
                result.get("rationale"),  # $15
                _jsonb(_to_list(result.get("glossary_terms_seen"))),  # $16
                result.get("elapsed_ms"),  # $17
                result.get("verdict"),  # $18
                result.get("judge_reason"),  # $19
                result.get("company"),  # $20 company
                result.get("position"),  # $21 position
                "provider",  # $22 pipeline
                "ok",  # $23 db_write_status
                None,  # $24 db_write_error
            )
        result["db_write_status"] = "ok"
        result["db_write_error"] = None
        result["db_id"] = row["id"] if row else None

    except Exception as exc:
        error_msg = str(exc)
        logger.error(
            "Provider DB write failed for message_id=%s: %s",
            result.get("message_id"), error_msg,
        )
        result["db_write_status"] = "failed"
        result["db_write_error"] = error_msg

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/classify")
async def classify(body: ClassifyRequest) -> dict:
    cfg = _state.cfg

    context: list[Message] | None = None
    if body.context is not None:
        context = [msg.to_message() for msg in body.context]

    async def _run():
        import time
        t_start = time.perf_counter()
        try:
            logger.info(
                "[classify] ━━━ START message_id=%s username=%r "
                "group_id=%s text_len=%d ━━━",
                body.message_id, body.username,
                body.group_id, len(body.text or ""),
            )
            logger.debug("[classify] text preview: %r",
                         (body.text or "")[:120].replace("\n", " "))

            # ── Dedup ─────────────────────────────────────────────────────
            async with _state.pool.acquire() as conn:
                is_dup = await is_duplicate(
                    body.username, body.text, body.timestamp, conn
                )
                if is_dup:
                    logger.info(
                        "[classify] DUPLICATE skipped message_id=%s username=%r",
                        body.message_id, body.username,
                    )
                    await conn.execute(
                        _UPSERT_SQL,
                        body.group_id, body.message_id, body.timestamp,  # $1, $2, $3
                        body.username, body.text,  # $4, $5
                        False, None, "none", None, None,  # $6, $7, $8, $9, $10
                        _jsonb([]), _jsonb([]), _jsonb([]),  # $11, $12, $13
                        None,  # $14 evidence_quote
                        f"[duplicate text='{(body.text or '').strip()[:50]}'"
                        f" user={body.username}]",  # $15 rationale
                        _jsonb([]), None, None, None,  # $16, $17, $18, $19
                        None,  # $20 company
                        None,  # $21 position
                        "dedup",  # $22 pipeline
                        "ok",  # $23 db_write_status
                        None,  # $24 db_write_error
                    )
                    return

            logger.info(
                "[classify] dedup=ok → launching parallel pipelines "
                "message_id=%s", body.message_id,
            )

            # ── Parallel pipelines ────────────────────────────────────────
            buyer_task = asyncio.to_thread(
                classify_single_message,
                text=body.text,
                glossary=_state.glossary,
                llm=_state.llm,
                cfg=cfg,
                message_id=body.message_id,
                username=body.username,
                timestamp=body.timestamp,
                context=context,
                rag_index=_state.rag_index,
                judge_system=_state.judge_system,
            )
            provider_task = asyncio.to_thread(
                classify_provider_message,
                text=body.text,
                glossary=_state.provider_glossary,
                llm=_state.provider_llm,
                cfg=_state.provider_cfg,
                message_id=body.message_id,
                username=body.username,
                timestamp=body.timestamp,
                context=context,
            )

            buyer_result, provider_result = await asyncio.gather(
                buyer_task, provider_task,
                return_exceptions=True,
            )

            # ── Buyer result ──────────────────────────────────────────────
            if isinstance(buyer_result, Exception):
                logger.error(
                    "[buyer] PIPELINE ERROR message_id=%s: %s",
                    body.message_id, buyer_result, exc_info=buyer_result,
                )
                buyer_result = None
            else:
                is_lead = buyer_result.get("is_lead")
                verdict = buyer_result.get("verdict")
                lead_type = buyer_result.get("lead_type")
                conf = buyer_result.get("confidence")
                elapsed = buyer_result.get("elapsed_ms", 0)
                logger.info(
                    "[buyer] done message_id=%s is_lead=%s lead_type=%s "
                    "verdict=%s confidence=%s elapsed=%.0fms",
                    body.message_id, is_lead, lead_type,
                    verdict, conf, elapsed or 0,
                )
                if is_lead and verdict != "MISTAKE":
                    logger.info(
                        "[buyer] LEAD ✓ message_id=%s vertical=%s geo=%s methods=%s",
                        body.message_id,
                        buyer_result.get("vertical"),
                        buyer_result.get("geo"),
                        buyer_result.get("payment_methods_mentioned"),
                    )
                else:
                    logger.info(
                        "[buyer] not a lead message_id=%s rationale=%s",
                        body.message_id,
                        (buyer_result.get("rationale") or "")[:100],
                    )

            # ── Buyer persist + cascade ───────────────────────────────────
            if buyer_result is not None:
                try:
                    buyer_result = await _persist_buyer(
                        body.group_id, buyer_result
                    )
                    db_ok = buyer_result.get("db_write_status") == "ok"
                    logger.info(
                        "[buyer] persisted message_id=%s db_ok=%s db_id=%s",
                        body.message_id, db_ok, buyer_result.get("db_id"),
                    )

                    is_lead = buyer_result.get("is_lead")
                    verdict = buyer_result.get("verdict")

                    if (not is_lead or verdict == "MISTAKE") and db_ok:
                        source = buyer_result.get("db_id")

                        # Crypto buyer
                        logger.info(
                            "[cascade] crypto_buyer start message_id=%s",
                            body.message_id,
                        )
                        crypto_ok = await classify_crypto_buyer_single_message(
                            text=body.text,
                            source_lead_id=source,
                            message_id=body.message_id,
                            username=body.username,
                            conn_pool=_state.pool,
                        )
                        logger.info(
                            "[cascade] crypto_buyer done message_id=%s classified=%s",
                            body.message_id, crypto_ok,
                        )

                        if not crypto_ok:
                            # Casino
                            logger.info(
                                "[cascade] casino start message_id=%s",
                                body.message_id,
                            )
                            casino_ok = await run_casino_classifier(
                                text=body.text,
                                source_lead_id=source,
                                message_id=body.message_id,
                                username=body.username,
                                timestamp=body.timestamp,
                                conn_pool=_state.pool,
                            )
                            logger.info(
                                "[cascade] casino done message_id=%s classified=%s",
                                body.message_id, casino_ok,
                            )

                            if not casino_ok:
                                # IBAN
                                logger.info(
                                    "[cascade] iban start message_id=%s",
                                    body.message_id,
                                )
                                iban_ok = await run_iban_lead_classifier(
                                    text=body.text,
                                    source_lead_id=source,
                                    message_id=body.message_id,
                                    username=body.username,
                                    timestamp=body.timestamp,
                                    conn_pool=_state.pool,
                                )
                                logger.info(
                                    "[cascade] iban done message_id=%s classified=%s",
                                    body.message_id, iban_ok,
                                )

                                if not iban_ok:
                                    logger.info(
                                        "[cascade] no classifier matched "
                                        "message_id=%s", body.message_id,
                                    )

                except Exception as exc:
                    logger.error(
                        "[buyer] persist/cascade FAILED message_id=%s: %s",
                        body.message_id, exc, exc_info=True,
                    )

            # ── Provider result ───────────────────────────────────────────
            if isinstance(provider_result, Exception):
                logger.error(
                    "[provider] PIPELINE ERROR message_id=%s: %s",
                    body.message_id, provider_result, exc_info=provider_result,
                )
                provider_result = None
            else:
                is_prov = provider_result.get("is_provider")
                verdict = provider_result.get("verdict")
                conf = provider_result.get("confidence")
                elapsed = provider_result.get("elapsed_ms", 0)
                rationale = provider_result.get("rationale", "")

                logger.info(
                    "[provider] done message_id=%s is_provider=%s "
                    "verdict=%s confidence=%s elapsed=%.0fms",
                    body.message_id, is_prov, verdict, conf, elapsed or 0,
                )

                if is_prov and verdict == "REAL_PROVIDER":
                    logger.info(
                        "[provider] PROVIDER ✓ message_id=%s "
                        "company=%r position=%r geo=%s vertical=%s methods=%s",
                        body.message_id,
                        provider_result.get("company"),
                        provider_result.get("position"),
                        provider_result.get("geo"),
                        provider_result.get("vertical"),
                        provider_result.get("methods"),
                    )
                    logger.info(
                        "[provider] evidence: %r",
                        provider_result.get("evidence_quote", "")[:120],
                    )
                    logger.info(
                        "[provider] judge_reason: %s",
                        provider_result.get("judge_reason", ""),
                    )
                elif is_prov and verdict == "MISTAKE":
                    logger.info(
                        "[provider] judge REJECTED message_id=%s "
                        "judge_reason=%s",
                        body.message_id,
                        provider_result.get("judge_reason", ""),
                    )
                else:
                    logger.info(
                        "[provider] not a provider message_id=%s "
                        "rationale=%s",
                        body.message_id,
                        (rationale or "")[:100],
                    )

            # ── Provider persist ──────────────────────────────────────────
            if provider_result is not None:
                try:
                    provider_result = await _persist_provider(
                        body.group_id, provider_result,
                        source_lead_id=(
                            buyer_result.get("db_id") if buyer_result else None
                        ),
                    )
                    logger.info(
                        "[provider] persisted message_id=%s "
                        "db_status=%s db_id=%s",
                        body.message_id,
                        provider_result.get("db_write_status"),
                        provider_result.get("db_id"),
                    )
                except Exception as exc:
                    logger.error(
                        "[provider] persist FAILED message_id=%s: %s",
                        body.message_id, exc, exc_info=True,
                    )

            t_total = (time.perf_counter() - t_start) * 1000
            logger.info("[classify] ━━━ DONE message_id=%s total=%.0fms ━━━", body.message_id, t_total)

        except Exception as exc:
            logger.error(
                "[classify] UNHANDLED ERROR message_id=%s: %s",
                body.message_id, exc, exc_info=True,
            )

    asyncio.create_task(_run())
    return {"status": "accepted", "message_id": body.message_id}

@app.get("/health")
async def health() -> dict:
    """Liveness check — also verifies DB connectivity."""
    try:
        async with _state.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status": "ok",
        "glossary_entries": len(_state.glossary),
        "provider_glossary_entries": len(_state.provider_glossary),
        "db": db_status,
    }