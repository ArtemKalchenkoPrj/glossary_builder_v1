"""FastAPI webhook for single-message lead classification.

Exposes two endpoints:

    POST /classify  — classify a message and persist the result to PostgreSQL
    GET  /health    — liveness check

Run locally:
    uvicorn glossary_builder.webhook:app --host 0.0.0.0 --port 8000 --reload

Or from the project root:
    python -m uvicorn glossary_builder.webhook:app --host 0.0.0.0 --port 8000

Required .env variables (in addition to the LLM key already used by the project):
    POSTGRES_DSN=postgresql://user:password@host:5432/dbname
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
from glossary_builder.cli import JUDGE_PROMPTS_BY_VERSION
from glossary_builder.lead_extraction import LeadExtractionConfig, load_glossary
from glossary_builder.lead_prompt_versions import BY_VERSION
from glossary_builder.llm import LLMClient
from glossary_builder.loader import Message
from glossary_builder.rag import RagIndex, RagConfig
from glossary_builder.single_classifier import classify_single_message
from glossary_builder.geo_pipeline import process_geo
from PSP_providers_classifier.psp_provider_classifier import run_psp_provider_classifier
from Deduper.deduplication import is_duplicate

_TABLE_PREFIX = os.getenv('TABLE_PREFIX') or ""
logger = logging.getLogger(__name__)

# Upsert: on (group_id, message_id) conflict — overwrite all classification
# fields with the fresh result so re-classification is idempotent.
_UPSERT_SQL = f"""
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
ON CONFLICT ON CONSTRAINT {_TABLE_PREFIX}uq_group_message DO UPDATE SET
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
    db_write_status           = EXCLUDED.db_write_status,
    db_write_error            = EXCLUDED.db_write_error,
    classified_at             = NOW()
RETURNING id;
"""


# ---------------------------------------------------------------------------
# App state — loaded once at startup, reused across all requests.
# ---------------------------------------------------------------------------

class _AppState:
    glossary: list[dict]
    llm: LLMClient
    cfg: LeadExtractionConfig
    pool: asyncpg.Pool
    rag_index: RagIndex
    judge_system: str

_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources on startup; release on shutdown."""
    project_root = Path(__file__).parent

    # Classifier resources.
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

    # PostgreSQL connection pool.
    dsn = os.environ["POSTGRES_DSN"]  # fail fast if not set
    _state.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

    logger.info("Startup complete — glossary: %d entries", len(_state.glossary))

    yield

    await _state.pool.close()


app = FastAPI(
    title="Lead Classifier",
    description="Single-message lead classification via extract → judge pipeline.",
    version="1.0.0",
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
# DB helper
# ---------------------------------------------------------------------------

async def _persist(group_id: Optional[int], result: dict) -> dict:
    """Write classification result to PostgreSQL.

    Returns the result dict with db_write_status / db_write_error fields
    added so the caller always gets a consistent response shape, even when
    the DB write fails (variant A error marking).
    """
    def _jsonb(value) -> str:
        """Serialise list fields to JSON strings for asyncpg JSONB params."""
        return json.dumps(value, ensure_ascii=False)

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
                group_id,
                result.get("message_id"),
                # asyncpg accepts datetime objects or None for TIMESTAMPTZ.
                _parse_dt(result.get("timestamp")),
                result.get("username"),
                result.get("text"),
                result.get("is_lead"),
                result.get("confidence"),
                result.get("lead_type"),
                result.get("intent"),
                result.get("interest_level"),
                _jsonb(result.get("vertical") or []),
                _jsonb(result.get("geo") or []),
                _jsonb(result.get("payment_methods_mentioned") or []),
                result.get("evidence_quote"),
                result.get("rationale"),
                _jsonb(result.get("glossary_terms_seen") or []),
                result.get("elapsed_ms"),
                result.get("verdict"),
                result.get("judge_reason"),
                "ok",   # db_write_status
                None,   # db_write_error
            )
        result["db_write_status"] = "ok"
        result["db_write_error"] = None
        result["db_id"] = row["id"] if row else None

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        logger.error("DB write failed for message_id=%s: %s", result.get("message_id"), error_msg)
        result["db_write_status"] = "failed"
        result["db_write_error"] = error_msg

    return result


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
        try:
            logger.info(
                "[classify] start message_id=%s username=%r text_len=%d",
                body.message_id, body.username, len(body.text or "")
            )

            async with _state.pool.acquire() as conn:
                if await is_duplicate(body.username, body.text, body.timestamp, conn):
                    logger.info(
                        "[classify] duplicate skipped message_id=%s username=%r",
                        body.message_id, body.username
                    )
                    await conn.execute(
                        _UPSERT_SQL,
                        body.group_id,
                        body.message_id,
                        body.timestamp,
                        body.username,
                        body.text,
                        False,
                        None, None, None, None,
                        json.dumps([]), json.dumps([]), json.dumps([]),
                        None,
                        f"[duplicate of text='{body.text.strip()[:50]}' of user={body.username}]",
                        json.dumps([]),
                        None, None, None,
                        "ok", None,
                    )
                    return

            logger.info("[classify] running extractor message_id=%s", body.message_id)
            result = await asyncio.to_thread(
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

            is_lead = result.get("is_lead")
            verdict = result.get("verdict")
            confidence = result.get("confidence")

            logger.info(
                "[classify] extractor done message_id=%s is_lead=%s verdict=%s confidence=%s",
                body.message_id, is_lead, verdict, confidence
            )

            if is_lead:
                logger.info(
                    "[classify] LEAD found message_id=%s vertical=%s geo=%s methods=%s",
                    body.message_id,
                    result.get("vertical"),
                    result.get("geo"),
                    result.get("payment_methods_mentioned"),
                )

            result = await _persist(body.group_id, result)
            db_ok = result.get("db_write_status") == "ok"

            logger.info(
                "[classify] persisted message_id=%s db_ok=%s db_id=%s",
                body.message_id, db_ok, result.get("db_id")
            )

            if (not is_lead or verdict == "MISTAKE") and db_ok:
                logger.info("[classify] running psp_provider_classifier message_id=%s", body.message_id)
                psp_classified = await run_psp_provider_classifier(
                    text=body.text,
                    source_lead_id=result.get("db_id"),
                    message_id=body.message_id,
                    username=body.username,
                    timestamp=body.timestamp,
                    conn_pool=_state.pool,
                )
                logger.info(
                    "[classify] psp_classifier done message_id=%s psp_classified=%s",
                    body.message_id, psp_classified
                )

                if not psp_classified:
                    logger.info("[classify] running casino_classifier message_id=%s", body.message_id)
                    await run_casino_classifier(
                        text=body.text,
                        source_lead_id=result.get("db_id"),
                        message_id=body.message_id,
                        username=body.username,
                        timestamp=body.timestamp,
                        conn_pool=_state.pool,
                    )
                    logger.info("[classify] casino_classifier done message_id=%s", body.message_id)

            logger.info("[classify] done message_id=%s", body.message_id)

        except Exception as exc:
            logger.error(
                "[classify] FAILED message_id=%s error=%s",
                body.message_id, exc, exc_info=True
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
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    return {
        "status": "ok",
        "glossary_entries": len(_state.glossary),
        "db": db_status,
    }