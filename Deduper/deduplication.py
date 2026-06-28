"""Message deduplication logic.

is_duplicate(username, text, timestamp, conn) -> bool

Logic:
    1. Fetch the most recent message from classified_messages_dirty
       for the same username.
    2. If none found              -> False (first message ever, let it through)
    3. If texts differ (case-insensitive) -> False (new content, reset the window)
    4. If time delta > 24 hours   -> False (window expired, let it through)
    5. Otherwise                  -> True  (duplicate within the 24h window)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import asyncpg


_WINDOW = timedelta(hours=24)

_SELECT_LAST_SQL = """
SELECT id, text, classified_at
FROM classified_messages_dirty
WHERE username = $1
ORDER BY classified_at DESC
LIMIT 1
"""


async def is_duplicate(
        username: Optional[str],
        text: str,
        timestamp: Optional[datetime],
        conn: asyncpg.Connection,
) -> tuple[bool, Optional[int]]:
    """Returns (is_dup, original_id).

    original_id — id рядка-оригіналу в classified_messages_dirty,
    None якщо не дублікат.
    """
    if not username:
        return False, None

    text_norm = text.strip().lower()
    if not text_norm:
        return False, None

    row = await conn.fetchrow(_SELECT_LAST_SQL, username)
    if row is None:
        return False, None

    last_text = (row["text"] or "").strip().lower()
    if last_text != text_norm:
        return False, None

    last_time: datetime = row["classified_at"]

    # Ensure both datetimes are timezone-aware for comparison.
    now = timestamp if timestamp is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)

    delta = now - last_time
    if delta > _WINDOW:
        return False, None

    return True, row["id"]
