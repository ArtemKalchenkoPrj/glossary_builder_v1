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

from glossary_builder.lead_extraction import LeadExtractionConfig, load_glossary
from glossary_builder.lead_prompt_versions import BY_VERSION
from glossary_builder.llm import LLMClient
from glossary_builder.loader import Message
from glossary_builder.single_classifier import classify_single_message
from glossary_builder.geo_pipeline import process_geo
from PSP_providers_classifier.psp_provider_classifier import run_psp_provider_classifier

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — runs once on startup if the table does not exist yet.
#
# Primary key: serial id.
# Unique constraint: (group_id, message_id) — ON CONFLICT DO UPDATE.
# db_write_status / db_write_error: variant A marker for failed DB writes
#   (the classification result is always returned to the caller; only the
#   persistence status is reflected here).
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS classified_messages_dirty (
    -- Surrogate primary key.
    id                       BIGSERIAL PRIMARY KEY,

    -- Source identifiers. The pair (group_id, message_id) is unique;
    -- a re-classification of the same message will UPDATE the existing row.
    group_id                 BIGINT,
    message_id               BIGINT,

    -- Core message fields.
    timestamp                TIMESTAMPTZ,
    username                 TEXT,
    text                     TEXT,

    -- Extract-stage output.
    is_lead                  BOOLEAN,
    confidence               DOUBLE PRECISION,
    lead_type                TEXT,
    intent                   TEXT,
    interest_level           TEXT,
    vertical                 JSONB,
    geo                      JSONB,
    payment_methods_mentioned JSONB,
    evidence_quote           TEXT,
    rationale                TEXT,
    glossary_terms_seen      JSONB,
    elapsed_ms               DOUBLE PRECISION,

    -- Judge-stage output (NULL when is_lead=FALSE).
    verdict                  TEXT,
    judge_reason             TEXT,

    -- DB write metadata (variant A: status recorded in the row itself).
    db_write_status          TEXT    NOT NULL DEFAULT 'ok',
    db_write_error           TEXT,

    -- Housekeeping.
    classified_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_group_message UNIQUE (group_id, message_id)
);
"""

# Upsert: on (group_id, message_id) conflict — overwrite all classification
# fields with the fresh result so re-classification is idempotent.
_UPSERT_SQL = """
INSERT INTO classified_messages_dirty (
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
ON CONFLICT ON CONSTRAINT uq_group_message DO UPDATE SET
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


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources on startup; release on shutdown."""
    project_root = Path(__file__).parent

    # Classifier resources.
    _state.glossary = load_glossary(
        project_root / "data/output/glossary_primary_sense_only.json"
    )
    _state.llm = LLMClient()
    _state.cfg = LeadExtractionConfig()
    _state.cfg.lead_definition = BY_VERSION["v5-short"]
    _state.cfg.pre_filter_db_path = str(project_root / "data/output/glossary.db")

    # PostgreSQL connection pool.
    dsn = os.environ["POSTGRES_DSN"]  # fail fast if not set
    _state.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

    # Ensure the table exists (idempotent).
    async with _state.pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE_SQL)

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
    prompt_version: Optional[str] = Field(
        None,
        description=(
            "Lead-definition prompt version to use for this request. "
            f"Available: {list(BY_VERSION.keys())}. "
            "When omitted the server default (v5-short) is used."
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
    if body.prompt_version is not None:
        if body.prompt_version not in BY_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown prompt_version {body.prompt_version!r}. "
                    f"Available: {list(BY_VERSION.keys())}"
                ),
            )
        cfg = LeadExtractionConfig()
        cfg.lead_definition = BY_VERSION[body.prompt_version]
        cfg.pre_filter_db_path = _state.cfg.pre_filter_db_path
    else:
        cfg = _state.cfg

    context: list[Message] | None = None
    if body.context is not None:
        context = [msg.to_message() for msg in body.context]

    async def _run():
        try:
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
            )
            result = await _persist(body.group_id, result)

            is_lead = result.get("is_lead")
            verdict = result.get("verdict")

            if (not is_lead or verdict == "MISTAKE") and result.get("db_write_status") == "ok":
                print("running psp_provider_classifier")
                await run_psp_provider_classifier(
                    text=body.text,
                    source_lead_id=result.get("db_id"),
                    message_id=body.message_id,
                    username=body.username,
                    timestamp=body.timestamp,
                    conn_pool=_state.pool
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Background classify failed for message_id=%s: %s", body.message_id, exc)

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