"""Stage 1b runner for PSP Provider glossary extraction.

Runs LLM-assisted phrase extraction on the seller-intent corpus collected
by collect_corpus.ipynb, and persists the results into a dedicated SQLite
database (provider_glossary.db).

Key differences from run_stage1b.py:
  - Uses payments_igaming_providers.phrase_seed_block (seller-side examples)
  - Uses a provider-focused domain_brief
  - Defaults tuned for a smaller curated corpus (higher max_batches, lower min_frequency)
  - Reads from seller_corpus.xlsx / "corpus" sheet by default

Usage:
    python3 run_provider_glossary.py \\
        --input PSP_providers_glossary_builder/data/seller_corpus.xlsx \\
        --sheet corpus \\
        --db PSP_providers_glossary_builder/data/provider_glossary.db

    # Dry run — no LLM calls, just check loading:
    python3 run_provider_glossary.py --input ... --dry-run

    # Small test on first 200 messages:
    python3 run_provider_glossary.py --input ... --sample-limit 200
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

# Provider-specific domain data — seed block for seller-side phrase extraction.
# Imported from payments_igaming_providers.py in the project root.
from payments_igaming_providers import phrase_seed_block as PROVIDER_SEED_BLOCK

console = Console()

# ── Provider-focused domain brief ─────────────────────────────────────────────
# Replaces the default buyer-side brief in PipelineConfig.
_PROVIDER_DOMAIN_BRIEF = (
    "A multilingual Telegram group focused on payment processing. "
    "These messages are from PSP PROVIDERS — companies and individuals "
    "OFFERING payment processing, acquiring, and gateway services to "
    "high-risk merchants. Verticals covered: iGaming, online casinos, "
    "sportsbooks, forex, crypto, adult, nutra. "
    "Language: Russian, Ukrainian, English, often mixed. "
    "Focus is on SELLER-SIDE language: providers describing their own "
    "coverage, approve rates, routing setups, onboarding terms, and "
    "payment methods they support."
)

# ── SQLite schema (same as run_stage1b.py) ────────────────────────────────────
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
    "russian_phrase":   "ru",
    "ukrainian_phrase": "uk",
    "mixed_phrase":     "mixed",
    "phrase":           "",
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
              help="Path to the seller corpus .xlsx (output of collect_corpus.ipynb).")
@click.option("--sheet", default="corpus",
              help="Sheet name in the Excel file (default: 'corpus').")
@click.option("--sample-limit", type=int, default=None,
              help="Cap messages loaded. None = load all. Use 200-500 for quick tests.")
@click.option("--db", "db_path",
              default="PSP_providers_glossary_builder/data/provider_glossary.db",
              type=click.Path(),
              help="SQLite DB path for output.")
@click.option("--replace/--no-replace", default=True,
              help="Wipe intent_glossary table before writing (default: replace).")
@click.option("--min-frequency", type=int, default=2,
              help="Minimum corpus frequency to keep a phrase (default: 2). "
                   "Lower than run_stage1b.py's default of 3 — seller corpus is smaller.")
@click.option("--max-batches", type=int, default=150,
              help="Max LLM batches (default: 150). "
                   "Higher than run_stage1b.py's default of 30 — cover more of the corpus.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Load and preview corpus without making any LLM calls.")
def main(
    input_path, sheet, sample_limit, db_path,
    replace, min_frequency, max_batches, dry_run,
):
    """Run Stage 1b phrase extraction on the PSP provider seller corpus."""

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = PipelineConfig()

    # Override: seller-side seed block instead of default buyer-side
    cfg.phrases.seed_block = PROVIDER_SEED_BLOCK

    # Override: provider-focused domain brief
    cfg.domain_brief = _PROVIDER_DOMAIN_BRIEF

    # Override: tuned for smaller curated corpus
    cfg.phrases.min_frequency = min_frequency
    cfg.phrases.max_batches   = max_batches

    # ── Load corpus ───────────────────────────────────────────────────────────
    console.rule("[bold]PSP Provider Glossary — Stage 1b")
    console.print(f"Input : [cyan]{input_path}[/]  sheet=[cyan]{sheet!r}[/]")
    console.print(f"Output: [cyan]{db_path}[/]")
    console.print(
        f"Config: min_frequency={cfg.phrases.min_frequency}  "
        f"max_batches={cfg.phrases.max_batches}  "
        f"messages_per_batch={cfg.phrases.messages_per_batch}"
    )

    messages = load_messages(input_path, sheet=sheet, limit=sample_limit)
    stats = corpus_stats(messages)
    console.print(f"\nLoaded [green]{len(messages):,}[/] messages.")
    console.print(f"  unique users    : {stats.get('unique_users', '?'):,}")
    console.print(f"  date range      : {stats.get('date_min', '?')} → {stats.get('date_max', '?')}")

    if not messages:
        console.print("[red]No messages loaded — check --input and --sheet.[/]")
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[yellow]--dry-run: skipping LLM calls.[/]")
        console.print("First 5 messages:")
        for m in messages[:5]:
            preview = (m.text or "")[:120].replace("\n", " ")
            console.print(f"  [dim][{m.message_id}][/] {preview}")
        console.print("\n[green]Dry run OK.[/] Remove --dry-run to run extraction.")
        return

    # ── LLM setup ─────────────────────────────────────────────────────────────
    llm = LLMClient()
    console.print(f"\nLLM: [cyan]{llm.provider}[/]  model: [cyan]{llm.model}[/]")

    # ── Stage 1b: phrase extraction ───────────────────────────────────────────
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
        console.print("[yellow]No candidates extracted. Try lowering --min-frequency or checking the corpus.[/]")
        return

    # ── Persist ───────────────────────────────────────────────────────────────
    db = Path(db_path)
    total = _persist(db, candidates, replace=replace)
    console.print(f"[green]Saved {len(candidates)} phrases → {db}  ({total} rows total)[/]")

    # ── Preview top phrases ───────────────────────────────────────────────────
    top = sorted(candidates, key=lambda c: c.frequency, reverse=True)[:25]
    table = Table(
        show_header=True, header_style="bold cyan",
        title="Top 25 phrases by corpus frequency",
    )
    table.add_column("Phrase",   style="cyan", min_width=30)
    table.add_column("Language", style="dim",  width=10)
    table.add_column("Freq",     justify="right", width=6)
    for c in top:
        lang = _KIND_TO_LANG.get(c.kind, "?")
        table.add_row(c.term, lang, str(c.frequency))
    console.print(table)

    # ── Cost summary ──────────────────────────────────────────────────────────
    usage = llm.usage.to_dict()
    console.print()
    console.print(
        f"LLM usage: {usage['calls']} calls | "
        f"{usage['input_tokens']:,} in + {usage['output_tokens']:,} out tokens | "
        f"est. [bold]${usage['estimated_cost_usd']:.4f}[/]"
    )
    console.print(f"\n[bold green]Done.[/] Provider glossary saved to [cyan]{db}[/]")


if __name__ == "__main__":
    main()
