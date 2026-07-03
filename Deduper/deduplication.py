"""Message deduplication logic.

Deduplicates incoming Telegram messages before they reach classification.

Two independent problems are solved here:

1. Race condition — two messages from the same user arrive within
   milliseconds of each other and both would otherwise pass the check
   before either is written back. Solved with a Postgres advisory
   transaction lock (`pg_advisory_xact_lock`) scoped to `username`, so
   concurrent calls for the same user are fully serialised for the
   short duration of the check-and-capture step (not for the whole
   classification, which can take seconds).

2. Same message, different wording — a sender broadcasts the same
   message to several similar-topic chats, but not always with
   identical text (e.g. once in English, once in Russian). Exact text
   matching misses this. Instead, once a user has been classified as a
   `lead`, ANY message from them within the next 24h is treated as a
   duplicate, regardless of text. Only when the last classification was
   `not_lead` do we fall back to exact text matching (username + text
   hash + 24h window) — a `not_lead` message can plausibly be followed
   by a genuinely different, on-topic message later.

Table DDL (run once):

    CREATE TABLE dedup_log (
        username        TEXT PRIMARY KEY,
        text_hash       TEXT,
        classified_as   TEXT NOT NULL DEFAULT 'not_lead'
                            CHECK (classified_as IN ('in_progress', 'lead', 'not_lead')),
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

Migration from the old (text_hash-only) schema:

    ALTER TABLE dedup_log
        ADD COLUMN classified_as TEXT NOT NULL DEFAULT 'not_lead'
            CHECK (classified_as IN ('in_progress', 'lead', 'not_lead'));

`classified_as` semantics:

    in_progress  — a message from this user is currently being classified.
                   Any other message from the same user is ignored until
                   either classification finishes (see `mark_classified`)
                   or `_IN_PROGRESS_TTL` elapses (stale-lock protection).
    lead         — last known classification was a lead. Ignore
                   everything from this user for the next 24h.
    not_lead     — last known classification was not a lead. Only exact
                   repeats (same text, same 24h window) are ignored.

Sync with `classified_messages_dirty`:

    `classified_as` is a cache, not the source of truth. Before making a
    decision (and only when the current state is NOT `in_progress`), we
    re-check `classified_messages_dirty.manual_approval` for this user's
    latest message and refresh `classified_as` accordingly:

        manual_approval IS NULL or False     -> not_lead
        manual_approval IS True or 'pending' -> lead

    This means a human downgrading a lead to `False` re-opens the window
    for that user on their very next message, even within 24h.

Usage (in the caller, e.g. the FastAPI background task):

    if await is_duplicate(body.username, body.text, body.timestamp, conn):
        ...  # write a "duplicate" row, skip classification, return

    try:
        result = await classify_single_message(...)
        result = await _persist(body.group_id, result)
        await mark_classified(body.username, result.get("is_lead"), conn)
    except Exception:
        await mark_classified(body.username, is_lead=False, conn)
        raise

`mark_classified` MUST be called exactly once for every username that
`is_duplicate` returned False for — including on failure — otherwise the
user stays stuck as `in_progress` until `_IN_PROGRESS_TTL` expires.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import asyncpg

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_WINDOW = timedelta(hours=24)
_IN_PROGRESS_TTL = timedelta(minutes=2)
_TABLE_PREFIX = os.getenv('TABLE_PREFIX') or ""

_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1))"

_SELECT_SQL = f"""
SELECT text_hash, classified_as, created_at
FROM {_TABLE_PREFIX}dedup_log
WHERE username = $1
FOR UPDATE
"""

_CAPTURE_INSERT_SQL = f"""
INSERT INTO {_TABLE_PREFIX}dedup_log (username, text_hash, classified_as, created_at)
VALUES ($1, $2, 'in_progress', $3)
"""

_CAPTURE_UPDATE_SQL = f"""
UPDATE {_TABLE_PREFIX}dedup_log
SET text_hash = $2, classified_as = 'in_progress', created_at = $3
WHERE username = $1
"""

_SYNC_SELECT_SQL = f"""
SELECT manual_approval
FROM {_TABLE_PREFIX}classified_messages_dirty
WHERE username = $1
ORDER BY timestamp DESC
LIMIT 1
"""

_SYNC_UPDATE_SQL = f"""
UPDATE {_TABLE_PREFIX}dedup_log
SET classified_as = $2
WHERE username = $1
"""

_MARK_SQL = f"""
UPDATE {_TABLE_PREFIX}dedup_log
SET classified_as = $2
WHERE username = $1
"""

def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _manual_approval_to_state(manual_approval) -> str:
    """Map classified_messages_dirty.manual_approval to a dedup_log state.

    None / False               -> 'not_lead'  (explicit non-lead, or LLM
                                                 verdict never reached a human)
    True / 'pending' / other   -> 'lead'       (LLM said lead; whether or
                                                 not a human confirmed yet,
                                                 we still don't want more
                                                 duplicates from this user)
    """
    if manual_approval is None or manual_approval is False:
        return "not_lead"
    return "lead"


async def is_duplicate(
    username: Optional[str],
    text: str,
    timestamp: Optional[datetime],
    conn: asyncpg.Connection,
) -> bool:
    """Check whether this message should be skipped as a duplicate.

    Returns:
        True  — duplicate (or another message from this user is already
                being classified) — caller should silently ignore.
        False — caller should proceed to classification. This function
                has already atomically marked the user as `in_progress`
                in dedup_log; the caller MUST eventually call
                `mark_classified()` for this username, even on error.
    """
    if not username:
        return False

    text_norm = text.strip().lower()
    if not text_norm:
        return False

    now = _aware(timestamp) if timestamp is not None else datetime.now(timezone.utc)
    new_hash = _hash(text)

    async with conn.transaction():
        # Serialise all concurrent calls for this username. The lock is
        # held only for this transaction (the check-and-capture step),
        # not for the whole classification that follows.
        await conn.execute(_LOCK_SQL, username)

        row = await conn.fetchrow(_SELECT_SQL, username)

        if row is None:
            await conn.execute(_CAPTURE_INSERT_SQL, username, new_hash, now)
            return False

        classified_as: str = row["classified_as"]
        created_at: datetime = _aware(row["created_at"])
        stored_hash: Optional[str] = row["text_hash"]

        if classified_as == "in_progress":
            if now - created_at < _IN_PROGRESS_TTL:
                return True  # someone else is classifying this user right now
            # stale lock (crashed/timed-out worker) — treat as free
            await conn.execute(_CAPTURE_UPDATE_SQL, username, new_hash, now)
            return False

        # classified_as is 'lead' or 'not_lead' here — sync with the
        # source of truth before deciding.
        sync_row = await conn.fetchrow(_SYNC_SELECT_SQL, username)
        if sync_row is not None:
            synced_state = _manual_approval_to_state(sync_row["manual_approval"])
            if synced_state != classified_as:
                await conn.execute(_SYNC_UPDATE_SQL, username, synced_state)
                classified_as = synced_state

        if classified_as == "lead":
            if now - created_at < _WINDOW:
                return True
            await conn.execute(_CAPTURE_UPDATE_SQL, username, new_hash, now)
            return False

        # classified_as == 'not_lead' -> fall back to exact-text matching
        if stored_hash == new_hash and now - created_at < _WINDOW:
            return True

        await conn.execute(_CAPTURE_UPDATE_SQL, username, new_hash, now)
        return False


async def mark_classified(
    username: Optional[str],
    is_lead: bool,
    conn: asyncpg.Connection,
) -> None:
    """Record the classification outcome, releasing the `in_progress` lock.

    Must be called exactly once for every username that `is_duplicate`
    returned False for — including on classification failure (pass
    `is_lead=False` in that case) so the user is never stuck in
    `in_progress` beyond `_IN_PROGRESS_TTL`.
    """
    if not username:
        return

    state = "lead" if is_lead else "not_lead"
    await conn.execute(_MARK_SQL, username, state)