"""Manual test for is_duplicate — connects to real DB, no webhook needed.

Run:
    python test_deduplication.py

What it tests:
    Case 1 — перше повідомлення автора (немає в БД)           -> False
    Case 2 — той самий текст < 24 год                         -> True
    Case 3 — інший текст від того ж автора < 24 год           -> False
    Case 4 — той самий текст але > 24 год тому                -> False
    Case 5 — username = None                                   -> False
    Case 6 — текст відрізняється тільки регістром              -> True
    Case 7 — текст відрізняється тільки пробілами на початку  -> True
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from dotenv import load_dotenv
from deduplication import is_duplicate

load_dotenv()


# ---------------------------------------------------------------------------
# Mock asyncpg connection — щоб не лізти в реальну БД для більшості тестів
# ---------------------------------------------------------------------------

def make_conn(last_text: str | None, classified_at: datetime | None):
    """Повертає mock asyncpg connection з одним фіксованим рядком."""
    conn = MagicMock()
    if last_text is None:
        conn.fetchrow = AsyncMock(return_value=None)
    else:
        row = {"text": last_text, "classified_at": classified_at}
        conn.fetchrow = AsyncMock(return_value=row)
    return conn


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_case(name: str, result: bool, expected: bool):
    status = "OK  " if result == expected else "FAIL"
    print(f"[{status}] {name}")
    if result != expected:
        print(f"       got {result}, expected {expected}")


NOW = datetime.now(timezone.utc)
RECENT = NOW - timedelta(hours=1)
EXPIRED = NOW - timedelta(hours=25)


async def main():
    # Case 1 — перше повідомлення, нічого в БД
    conn = make_conn(None, None)
    result = await is_duplicate("ivan", "Привіт!", NOW, conn)
    await run_case("перше повідомлення автора -> False", result, False)

    # Case 2 — той самий текст < 24 год -> дублікат
    conn = make_conn("Привіт! Шукаю Іспанію", RECENT)
    result = await is_duplicate("ivan", "Привіт! Шукаю Іспанію", NOW, conn)
    await run_case("той самий текст < 24 год -> True", result, True)

    # Case 3 — інший текст < 24 год -> не дублікат
    conn = make_conn("Привіт! Шукаю Іспанію", RECENT)
    result = await is_duplicate("ivan", "О! І мені!", NOW, conn)
    await run_case("інший текст від того ж автора -> False", result, False)

    # Case 4 — той самий текст але > 24 год -> вікно минуло
    conn = make_conn("Привіт! Шукаю Іспанію", EXPIRED)
    result = await is_duplicate("ivan", "Привіт! Шукаю Іспанію", NOW, conn)
    await run_case("той самий текст > 24 год тому -> False", result, False)

    # Case 5 — username None -> пропускаємо дедуплікацію
    conn = make_conn("Привіт! Шукаю Іспанію", RECENT)
    result = await is_duplicate(None, "Привіт! Шукаю Іспанію", NOW, conn)
    await run_case("username = None -> False", result, False)

    # Case 6 — відрізняється тільки регістром -> дублікат
    conn = make_conn("привіт! шукаю іспанію", RECENT)
    result = await is_duplicate("ivan", "ПРИВІТ! ШУКАЮ ІСПАНІЮ", NOW, conn)
    await run_case("різний регістр -> True", result, True)

    # Case 7 — пробіли на початку/кінці -> дублікат
    conn = make_conn("Привіт! Шукаю Іспанію", RECENT)
    result = await is_duplicate("ivan", "  Привіт! Шукаю Іспанію  ", NOW, conn)
    await run_case("пробіли на початку/кінці -> True", result, True)


asyncio.run(main())
