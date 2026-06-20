"""Quick manual test for geo_pipeline + DB write.

Run:
    python test_geo_pipeline.py
"""
import asyncio
import json
import os
import asyncpg
from dotenv import load_dotenv
from glossary_builder.geo_pipeline import process_geo

load_dotenv()

# --- захардкоджений result як ніби прийшов з класифікатора ---
MOCK_RESULT = {
    "message_id": 9999,
    "group_id": 100,
    "timestamp": "2026-06-07T10:05:00",
    "username": "test_user",
    "text": "Тестове повідомлення для geo pipeline",
    "is_lead": True,
    "confidence": 0.9,
    "lead_type": "seeking_psp",
    "intent": "buying",
    "interest_level": "high",
    "vertical": ["igaming"],
    "geo": ["KZ", "RU", "Росія", "EU", "Европа", "Казахстан"],   # RU і Росія — навмисний дублікат для тесту
    "payment_methods_mentioned": ["visa", "монобанк"],           # монобанк -> має додати UA
    "evidence_quote": "тест",
    "rationale": "тест",
    "glossary_terms_seen": [],
    "elapsed_ms": 0.0,
    "verdict": "null",
    "judge_reason": "test",
}

def _jsonb(value) -> str:
    return json.dumps(value, ensure_ascii=False)

async def main():
    dsn = os.environ["POSTGRES_DSN"]
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)

    async with pool.acquire() as conn:
        # запускаємо пайплайн (нова сигнатура — додано payment_methods)
        processed_geo = await process_geo(
            message_id=MOCK_RESULT["message_id"],
            raw_geo_list=MOCK_RESULT["geo"],
            payment_methods=MOCK_RESULT["payment_methods_mentioned"],
            conn=conn,
        )
        print(f"Оброблене гео: {processed_geo}")
        assert "UA" in processed_geo, "очікував UA від монобанку, але його немає в результаті"
        print("UA від payment_geo_map присутнє — стадія 5 спрацювала")

        # пишемо в БД
        await conn.execute("""
            INSERT INTO classified_messages_dirty (
                group_id, message_id, timestamp, username, text,
                is_lead, confidence, lead_type, intent, interest_level,
                vertical, geo, payment_methods_mentioned,
                evidence_quote, rationale, glossary_terms_seen, elapsed_ms,
                verdict, judge_reason, db_write_status, db_write_error
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21
            )
            ON CONFLICT ON CONSTRAINT uq_group_message DO UPDATE SET
                geo = EXCLUDED.geo,
                classified_at = NOW()
            RETURNING id;
        """,
            MOCK_RESULT["group_id"],
            MOCK_RESULT["message_id"],
            None,  # timestamp
            MOCK_RESULT["username"],
            MOCK_RESULT["text"],
            MOCK_RESULT["is_lead"],
            MOCK_RESULT["confidence"],
            MOCK_RESULT["lead_type"],
            MOCK_RESULT["intent"],
            MOCK_RESULT["interest_level"],
            _jsonb(MOCK_RESULT["vertical"]),
            _jsonb(processed_geo),
            _jsonb(MOCK_RESULT["payment_methods_mentioned"]),
            MOCK_RESULT["evidence_quote"],
            MOCK_RESULT["rationale"],
            _jsonb(MOCK_RESULT["glossary_terms_seen"]),
            MOCK_RESULT["elapsed_ms"],
            MOCK_RESULT["verdict"],
            MOCK_RESULT["judge_reason"],
            "ok",
            None,
        )
        print("Записано в БД успішно")

    await pool.close()

asyncio.run(main())