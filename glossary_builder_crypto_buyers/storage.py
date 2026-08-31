"""SQLite-backed glossary store.

We keep the schema tiny on purpose: one row per term, one row per sense, plus
the example messages and the adversarial critique. SQLite is enough — the
glossary is dozens to low-thousands of entries, not millions. Downstream
agents can read it directly from disk at runtime, or export to JSON.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .inference import DraftEntry
from .validator import Critique


_SCHEMA = """
CREATE TABLE IF NOT EXISTS terms (
    term            TEXT PRIMARY KEY,
    relevant        INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    source_stage    TEXT NOT NULL,
    reasoning       TEXT,
    critique        TEXT,
    critique_severity TEXT,
    surface_forms   TEXT,   -- JSON array
    frequency       INTEGER,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS senses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    term            TEXT NOT NULL,
    definition      TEXT NOT NULL,
    context_clues   TEXT,   -- JSON array
    example_message_ids TEXT,  -- JSON array
    FOREIGN KEY (term) REFERENCES terms(term) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_senses_term ON senses(term);
CREATE INDEX IF NOT EXISTS idx_terms_relevant ON terms(relevant);
CREATE INDEX IF NOT EXISTS idx_terms_confidence ON terms(confidence);
"""


class GlossaryStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(
        self,
        draft: DraftEntry,
        critique: Critique | None = None,
        *,
        surface_forms: list[str] | None = None,
        frequency: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO terms (term, relevant, confidence, source_stage,
                                   reasoning, critique, critique_severity,
                                   surface_forms, frequency, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    relevant=excluded.relevant,
                    confidence=excluded.confidence,
                    source_stage=excluded.source_stage,
                    reasoning=excluded.reasoning,
                    critique=excluded.critique,
                    critique_severity=excluded.critique_severity,
                    surface_forms=excluded.surface_forms,
                    frequency=excluded.frequency,
                    updated_at=excluded.updated_at
                """,
                (
                    draft.term,
                    1 if draft.relevant else 0,
                    draft.confidence,
                    draft.source_stage,
                    draft.reasoning,
                    critique.critique if critique else None,
                    critique.issue_severity if critique else None,
                    json.dumps(surface_forms or []),
                    frequency,
                    now,
                ),
            )
            conn.execute("DELETE FROM senses WHERE term = ?", (draft.term,))
            for s in draft.senses:
                conn.execute(
                    """
                    INSERT INTO senses (term, definition, context_clues,
                                        example_message_ids)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        draft.term,
                        s.definition,
                        json.dumps(s.context_clues),
                        json.dumps(s.example_message_ids),
                    ),
                )

    def get(self, term: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM terms WHERE term = ?", (term,)).fetchone()
            if not row:
                return None
            senses = conn.execute(
                "SELECT definition, context_clues, example_message_ids "
                "FROM senses WHERE term = ?",
                (term,),
            ).fetchall()
        return _row_to_dict(row, senses)

    def all_entries(
        self,
        *,
        relevant_only: bool = True,
        min_confidence: float = 0.0,
        primary_sense_only: bool = False,
    ) -> list[dict]:
        """Read entries from the store with optional filtering.

        Args:
            relevant_only: drop entries marked relevant=False.
            min_confidence: drop entries below this confidence.
            primary_sense_only: keep only the first sense per entry. Use this
                for the bot-consumer export when you want a clean
                single-meaning lookup. The first sense is the one inferred
                by Stage 3 from corpus evidence; any LLM-enriched additional
                senses come second in the list and are dropped here.
        """
        clauses = []
        params: list = []
        if relevant_only:
            clauses.append("relevant = 1")
        if min_confidence > 0:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM terms {where} ORDER BY confidence DESC",
                params,
            ).fetchall()
            out = []
            for row in rows:
                senses = conn.execute(
                    "SELECT definition, context_clues, example_message_ids "
                    "FROM senses WHERE term = ? ORDER BY id ASC",
                    (row["term"],),
                ).fetchall()
                entry = _row_to_dict(row, senses)
                if primary_sense_only and len(entry["senses"]) > 1:
                    entry["senses"] = entry["senses"][:1]
                out.append(entry)
        return out

    def export_json(self, path: str | Path, **filters) -> int:
        entries = self.all_entries(**filters)
        Path(path).write_text(json.dumps(entries, ensure_ascii=False, indent=2),encoding="utf-8")
        return len(entries)


def _row_to_dict(row: sqlite3.Row, senses: Iterable[sqlite3.Row]) -> dict:
    return {
        "term": row["term"],
        "relevant": bool(row["relevant"]),
        "confidence": row["confidence"],
        "source_stage": row["source_stage"],
        "reasoning": row["reasoning"],
        "critique": row["critique"],
        "critique_severity": row["critique_severity"],
        "surface_forms": json.loads(row["surface_forms"] or "[]"),
        "frequency": row["frequency"],
        "updated_at": row["updated_at"],
        "senses": [
            {
                "definition": s["definition"],
                "context_clues": json.loads(s["context_clues"] or "[]"),
                "example_message_ids": json.loads(s["example_message_ids"] or "[]"),
            }
            for s in senses
        ],
    }
