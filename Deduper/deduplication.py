"""Message deduplication logic.

Uses a dedicated `dedup_log` table (one row per username) to atomically
detect duplicate messages before classification begins — solving the race
condition where two identical messages arrive within milliseconds of each
other and both pass the check before either is written to
classified_messages_dirty.

Table DDL (run once):

    CREATE TABLE dedup_log (
        username    TEXT PRIMARY KEY,
        text_hash   TEXT        NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

is_duplicate(username, text, timestamp, conn) -> bool

    Returns True  — duplicate, caller should silently ignore.
    Returns False — new message, caller should proceed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import asyncpg


_WINDOW = timedelta(hours=24)

_UPSERT_SQL = """
INSERT INTO dedup_log (username, text_hash, created_at)
VALUES ($1, $2, $3)
ON CONFLICT (username) DO UPDATE
    SET text_hash  = EXCLUDED.text_hash,
        created_at = EXCLUDED.created_at
    WHERE
        dedup_log.text_hash != EXCLUDED.text_hash
        OR dedup_log.created_at < EXCLUDED.created_at - INTERVAL '24 hours'
RETURNING text_hash, created_at
"""

_SELECT_SQL = """
SELECT text_hash, created_at
FROM dedup_log
WHERE username = $1
"""


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


async def is_duplicate(
    username: Optional[str],
    text: str,
    timestamp: Optional[datetime],
    conn: asyncpg.Connection,
) -> bool:
    """Check whether this message is a duplicate within the 24h sliding window.

    Atomically updates dedup_log so concurrent calls for the same username
    are serialised by the DB — no race condition possible.

    Args:
        username:  Sender handle. If None, deduplication is skipped (False).
        text:      Raw message text.
        timestamp: Message datetime. Falls back to utcnow() if None.
        conn:      Active asyncpg connection.

    Returns:
        True  — duplicate, caller should silently ignore.
        False — new message, caller should proceed.
    """
    if not username:
        return False

    text_norm = text.strip().lower()
    if not text_norm:
        return False

    now = timestamp if timestamp is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    new_hash = _hash(text)

    row = await conn.fetchrow(_UPSERT_SQL, username, new_hash, now)

    if row is not None:
        return False

    stored = await conn.fetchrow(_SELECT_SQL, username)
    if stored is None:
        return False

    stored_hash: str = stored["text_hash"]
    stored_time: datetime = stored["created_at"]

    if stored_hash != new_hash:
        return False

    if stored_time.tzinfo is None:
        stored_time = stored_time.replace(tzinfo=timezone.utc)

    delta = now - stored_time
    if delta > _WINDOW:
        return False

    return True