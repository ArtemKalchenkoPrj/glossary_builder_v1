"""Stage 1b runner — LLM-assisted phrase extraction for the Traffic vertical.

Reads a filtered corpus of affiliate-traffic messages, extracts domain-specific
phrases via LLM, and persists them to an SQLite ``intent_glossary`` table.

The resulting glossary is used by the Traffic classifier's pre-filter
(intent_filter_traffic.py) to reduce LLM calls on the webhook.

Usage:
    # Dry run — validate paths and show corpus stats, no LLM calls
    python run_traffic_glossary.py --input traffic_corpus.xlsx --dry-run

    # Test on first 200 messages
    python run_traffic_glossary.py --input traffic_corpus.xlsx --sample-limit 200

    # Full run
    python run_traffic_glossary.py --input traffic_corpus.xlsx

    # Resume interrupted run (adds to existing DB, no replace)
    python run_traffic_glossary.py --input traffic_corpus.xlsx --no-replace

Output:
    Traffic_classifier/data/traffic_glossary.db  (table: intent_glossary)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from glossary_builder.config import PipelineConfig
from glossary_builder.llm import LLMClient
from glossary_builder.loader import load_messages, corpus_stats
from glossary_builder.phrases import extract_phrases

from traffic_domain_data import phrase_seed_block

console = Console()

# ---------------------------------------------------------------------------
# Domain brief — describes the chat/corpus context to the LLM
# ---------------------------------------------------------------------------

_TRAFFIC_DOMAIN_BRIEF = (
    "A multilingual Telegram group focused on affiliate traffic trading "
    "(arbitrage, media buying, webmaster communities). Participants are "
    "affiliate marketers, arbitrageurs, webmasters, and media buyers who "
    "buy and sell traffic in high-risk verticals: gambling, casino, betting, "
    "crypto, forex, nutra, adult, dating, igaming. Messages are in Russian, "
    "Ukrainian, and English, often mixed within a single message. Common "
    "jargon: CPA, RevShare, FTD, CPL, оффер, лить трафик, арбитражник, "
    "вебмастер, медиабаер, крео, ГЕО, конверт, апрув, лидген."
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_INPUT = "traffic_corpus.xlsx"
_DEFAULT_SHEET = "Sheet1"
_DEFAULT_DB    = "Traffic_classifier/data/traffic_glossary.db"

# ---------------------------------------------------------------------------
# SQLite schema (same as run_stage1b.py — keep in sync)
# ---------------------------------------------------------------------------

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

_KIND_TO_LANG = {
    "russian_phrase": "ru",
    "ukrainian_phrase": "uk",
    "mixed_phrase": "mixed",
    "phrase": "",
}


# ---------------------------------------------------------------------------
# Persist helpers
# ---------------------------------------------------------------------------

def _persist(db_path: Path, candidates, *, replace: bool) -> int:
    """Write extracted phrase candidates to the intent_glossary table."""
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


def _preview_db(db_path: Path, top_n: int = 20) -> None:
    """Print the top N terms from the DB by frequency."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT term, kind, frequency FROM intent_glossary "
            "ORDER BY frequency DESC LIMIT ?",
            (top_n,),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM intent_glossary").fetchone()[0]
    finally:
        conn.close()

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Top {top_n} by frequency (total in DB: {total})",
    )
    table.add_column("Phrase", style="cyan")
    table.add_column("Kind", style="dim")
    table.add_column("Freq", justify="right")
    for term, kind, freq in rows:
        table.add_row(term, kind, str(freq))
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--input", "-i", "input_path",
    default=_DEFAULT_INPUT,
    type=click.Path(exists=True),
    help="Path to the filtered corpus .xlsx (traffic messages).",
)
@click.option(
    "--sheet",
    default=_DEFAULT_SHEET,
    show_default=True,
    help="Sheet name inside the Excel file.",
)
@click.option(
    "--sample-limit", "sample_limit",
    type=int,
    default=None,
    help="Cap the number of messages loaded (default: all). "
         "Use --sample-limit 200 for a quick test run.",
)
@click.option(
    "--db", "db_path",
    default=_DEFAULT_DB,
    type=click.Path(),
    show_default=True,
    help="Output SQLite DB path.",
)
@click.option(
    "--replace/--no-replace",
    default=True,
    show_default=True,
    help="Wipe the intent_glossary table before writing. "
         "Use --no-replace to resume an interrupted run.",
)
@click.option(
    "--min-frequency", "min_frequency",
    type=int,
    default=None,
    help="Override minimum phrase frequency (default from PhraseConfig: 3). "
         "Lower to 2 for small corpora (<1 000 messages).",
)
@click.option(
    "--max-batches", "max_batches",
    type=int,
    default=None,
    help="Override max LLM batches (default from PhraseConfig: 30). "
         "Raise to 150 for full corpus coverage.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load corpus and print stats, but skip LLM calls and DB writes.",
)
def main(
    input_path, sheet, sample_limit, db_path,
    replace, min_frequency, max_batches, dry_run,
):
    """Run Stage 1b phrase extraction for the Traffic vertical.

    Reads a filtered affiliate-traffic corpus, extracts domain phrases via LLM,
    and stores them in intent_glossary (SQLite).
    """
    # ── Config ────────────────────────────────────────────────────────────────
    cfg = PipelineConfig()
    cfg.domain_brief = _TRAFFIC_DOMAIN_BRIEF
    cfg.phrases.seed_block = phrase_seed_block  # traffic-specific examples

    if min_frequency is not None:
        cfg.phrases.min_frequency = min_frequency
    if max_batches is not None:
        cfg.phrases.max_batches = max_batches

    # ── Load corpus ───────────────────────────────────────────────────────────
    console.rule("[bold]Loading corpus")
    messages = load_messages(input_path, sheet=sheet, limit=sample_limit)
    stats = corpus_stats(messages)
    console.print(
        f"Loaded [green]{len(messages)}[/] messages "
        f"from [cyan]{input_path!r}[/] sheet [cyan]{sheet!r}[/]."
    )
    console.print(f"Corpus stats: {stats}")
    console.print(
        f"PhraseConfig: min_frequency=[cyan]{cfg.phrases.min_frequency}[/]  "
        f"max_batches=[cyan]{cfg.phrases.max_batches}[/]  "
        f"messages_per_batch=[cyan]{cfg.phrases.messages_per_batch}[/]"
    )

    if dry_run:
        console.print("\n[yellow]--dry-run: skipping LLM calls and DB write.[/]")
        console.print(f"[dim]Would write to: {db_path}[/]")
        return

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = LLMClient()
    console.print(
        f"LLM provider: [cyan]{llm.provider}[/]  model: [cyan]{llm.model}[/]"
    )

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    console.rule("[bold]Stage 1b — LLM phrase extraction")
    candidates = extract_phrases(
        messages,
        llm,
        domain_brief=cfg.domain_brief,
        config=cfg.phrases,
    )
    console.print(
        f"Extracted [green]{len(candidates)}[/] phrase candidates "
        f"(min_frequency={cfg.phrases.min_frequency})."
    )

    if not candidates:
        console.print(
            "[yellow]No candidates extracted. "
            "Try lowering --min-frequency or increasing --max-batches.[/]"
        )
        return

    # ── Persist ───────────────────────────────────────────────────────────────
    db = Path(db_path)
    total = _persist(db, candidates, replace=replace)
    console.print(
        f"[green]Wrote {len(candidates)} rows → "
        f"{db} (intent_glossary, {total} rows total).[/]"
    )

    # ── Preview ───────────────────────────────────────────────────────────────
    _preview_db(db, top_n=20)

    # ── Usage ─────────────────────────────────────────────────────────────────
    usage = llm.usage.to_dict()
    console.print()
    console.print(
        f"LLM usage: {usage['calls']} calls, "
        f"{usage['input_tokens']} in + {usage['output_tokens']} out tokens "
        f"(est. ${usage['estimated_cost_usd']:.4f})."
    )
    console.print(
        "\n[bold green]Done.[/] Next step:\n"
        "  python run_traffic_inference.py "
        f"--db {db_path}"
    )


if __name__ == "__main__":
    main()
