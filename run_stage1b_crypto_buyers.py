"""Standalone runner for Stage 1b only (LLM-assisted phrase extraction).

The shipped ``build`` command runs the full six-stage pipeline. This script
isolates Stage 1b (``glossary_builder.phrases.extract_phrases``) and persists
the resulting phrase candidates into a dedicated ``intent_glossary`` table in
a SQLite database, leaving the main glossary store untouched.

Usage:
    python3 run_stage1b.py \
        --input data/input/PSP_Judjes_3_months.xlsx \
        --sheet "PSP Judjes" \
        --sample-limit 500 \
        --db data/output/glossary.db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from glossary_builder_crypto_buyers.config import PipelineConfig
from glossary_builder_crypto_buyers.llm import LLMClient
from glossary_builder_crypto_buyers.loader import load_messages, corpus_stats
from glossary_builder_crypto_buyers.phrases import extract_phrases

console = Console()


_INTENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS intent_glossary (
    term                TEXT PRIMARY KEY,
    kind                TEXT NOT NULL,
    language            TEXT,
    frequency           INTEGER NOT NULL,
    surface_forms       TEXT,   -- JSON array
    example_message_ids TEXT,   -- JSON array
    examples            TEXT,   -- JSON array of {message_id,date,username,text}
    source_stage        TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intent_frequency ON intent_glossary(frequency);
"""

# kind -> language label, mirroring the mapping phrases.py uses.
_KIND_TO_LANG = {
    "russian_phrase": "ru",
    "ukrainian_phrase": "uk",
    "mixed_phrase": "mixed",
    "phrase": "",
}


def _persist(db_path: Path, candidates, *, replace: bool) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_INTENT_SCHEMA)
        if replace:
            conn.execute("DELETE FROM intent_glossary")
        for c in candidates:
            example_ids = [
                ex.get("message_id") for ex in c.example_messages
                if ex.get("message_id") is not None
            ]
            conn.execute(
                """
                INSERT INTO intent_glossary
                    (term, kind, language, frequency, surface_forms,
                     example_message_ids, examples, source_stage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    kind=excluded.kind,
                    language=excluded.language,
                    frequency=excluded.frequency,
                    surface_forms=excluded.surface_forms,
                    example_message_ids=excluded.example_message_ids,
                    examples=excluded.examples,
                    source_stage=excluded.source_stage,
                    created_at=excluded.created_at
                """,
                (
                    c.term,
                    c.kind,
                    _KIND_TO_LANG.get(c.kind, ""),
                    c.frequency,
                    json.dumps(c.surface_forms, ensure_ascii=False),
                    json.dumps(example_ids, ensure_ascii=False),
                    json.dumps(c.example_messages, ensure_ascii=False),
                    "stage1b_phrases",
                    now,
                ),
            )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM intent_glossary").fetchone()[0]
    finally:
        conn.close()
    return n


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Path to the .xlsx Telegram export.")
@click.option("--sheet", default="PSP Judjes", help="Sheet name.")
@click.option("--sample-limit", type=int, default=500,
              help="Cap the number of messages loaded.")
@click.option("--db", "db_path", default="data/output/glossary.db",
              type=click.Path(),
              help="SQLite DB for the intent_glossary table.")
@click.option("--replace/--no-replace", default=True,
              help="Wipe the intent_glossary table before writing (default: replace).")
@click.option("--min-frequency", type=int, default=None,
              help="Override PhraseConfig.min_frequency (default 3). "
                   "Lower it for small curated corpora.")
def main(input_path, sheet, sample_limit, db_path, replace, min_frequency):
    """Run Stage 1b (LLM phrase extraction) and store into intent_glossary."""
    cfg = PipelineConfig()
    if min_frequency is not None:
        cfg.phrases.min_frequency = min_frequency

    console.rule("[bold]Loading corpus")
    messages = load_messages(input_path, sheet=sheet, limit=sample_limit)
    stats = corpus_stats(messages)
    console.print(f"Loaded {len(messages)} messages from sheet [cyan]{sheet!r}[/].")
    console.print(f"Corpus stats: {stats}")

    llm = LLMClient()
    console.print(f"LLM provider: [cyan]{llm.provider}[/]  model: [cyan]{llm.model}[/]")

    console.rule("[bold]Stage 1b — LLM phrase extraction")
    candidates = extract_phrases(
        messages,
        llm,
        domain_brief=cfg.domain_brief,
        config=cfg.phrases,
    )
    console.print(f"Extracted [green]{len(candidates)}[/] phrase candidates "
                  f"(min_frequency={cfg.phrases.min_frequency}).")

    db = Path(db_path)
    total = _persist(db, candidates, replace=replace)
    console.print(f"[green]Wrote {len(candidates)} rows → "
                  f"{db} (table intent_glossary, {total} rows total).")

    # Preview the top phrases by frequency.
    top = sorted(candidates, key=lambda c: c.frequency, reverse=True)[:20]
    if top:
        table = Table(show_header=True, header_style="bold", title="Top 20 by frequency")
        table.add_column("Phrase", style="cyan")
        table.add_column("Kind", style="dim")
        table.add_column("Freq", justify="right")
        for c in top:
            table.add_row(c.term, c.kind, str(c.frequency))
        console.print(table)

    usage = llm.usage.to_dict()
    console.print()
    console.print(f"LLM usage: {usage['calls']} calls, "
                  f"{usage['input_tokens']} in + {usage['output_tokens']} out tokens "
                  f"(est. ${usage['estimated_cost_usd']:.4f} — $0 for local ollama).")


if __name__ == "__main__":
    main()
