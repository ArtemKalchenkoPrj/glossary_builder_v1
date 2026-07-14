"""Message deduplication logic.

Two independent checks before classification:

1. Time-based block — if the same user sent ANY message within the last
   3 minutes, the new message is treated as a duplicate. Prevents
   rapid-fire duplicates regardless of text.

2. Text-based block — if the same user sent the SAME text (by MD5 hash)
   within the last 24 hours, the new message is treated as a duplicate.

Race condition protection: Postgres advisory transaction lock scoped to
`username` serialises concurrent calls for the same user.

Table DDL (run once):

    CREATE TABLE dedup_log (
        username    TEXT PRIMARY KEY,
        text_hash   TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

Migration from old schema (if classified_as column exists):

    ALTER TABLE dedup_log DROP COLUMN classified_as;

Usage:

    if await is_duplicate(body.username, body.text, body.timestamp, conn):
        ...  # skip classification, write duplicate row, return
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import asyncpg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_HOLD_WINDOW = timedelta(minutes=3)
_TEXT_WINDOW = timedelta(hours=24)
_TABLE_PREFIX = os.getenv("TABLE_PREFIX") or ""

_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1))"

_SELECT_SQL = f"""
SELECT text_hash, created_at
FROM {_TABLE_PREFIX}dedup_log
WHERE username = $1
FOR UPDATE
"""

_INSERT_SQL = f"""
INSERT INTO {_TABLE_PREFIX}dedup_log (username, text_hash, created_at)
VALUES ($1, $2, $3)
"""

_UPDATE_SQL = f"""
UPDATE {_TABLE_PREFIX}dedup_log
SET text_hash = $2, created_at = $3
WHERE username = $1
"""


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def is_duplicate(
    username: Optional[str],
    text: str,
    timestamp: Optional[datetime],
    conn: asyncpg.Connection,
) -> bool:
    """Check whether this message should be skipped as a duplicate.

    Returns:
        True  — duplicate, caller should silently ignore.
        False — not a duplicate, caller should proceed to classification.
    """
    if not username:
        return False

    text_norm = text.strip().lower()
    if not text_norm:
        return False

    now = _aware(timestamp) if timestamp is not None else datetime.now(timezone.utc)
    new_hash = _hash(text)

    async with conn.transaction():
        await conn.execute(_LOCK_SQL, username)
        row = await conn.fetchrow(_SELECT_SQL, username)

        if row is None:
            await conn.execute(_INSERT_SQL, username, new_hash, now)
            return False

        created_at = _aware(row["created_at"])
        stored_hash = row["text_hash"]

        logger.info(
            "[dedup] username=%s now=%s created_at=%s diff=%s new_hash=%s stored_hash=%s",
            username, now, created_at, now - created_at, new_hash, stored_hash
        )

        if now - created_at < _HOLD_WINDOW:
            logger.info("[dedup] DUPLICATE: hold window")
            return True

        if stored_hash == new_hash and now - created_at < _TEXT_WINDOW:
            logger.info("[dedup] DUPLICATE: same text")
            return True

        await conn.execute(_UPDATE_SQL, username, new_hash, now)
        return False