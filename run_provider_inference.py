"""Stage 3 — LLM inference for PSP Provider glossary.

Takes raw phrase candidates from ``intent_glossary`` (built by
run_provider_glossary.py) and enriches each term with:
  - relevant: bool     — is this a real payment-domain term or noise?
  - senses             — LLM-generated definition + context clues
  - confidence: float  — how certain the LLM is

Results are saved to ``provider_glossary_enriched`` table in the same
SQLite DB. The classifier's load_provider_glossary() will prefer this
enriched table when available.

Usage:
    python run_provider_inference.py \\
        --db PSP_provider_glossary_builder/data/provider_glossary.db

    # Dry run — show candidates without LLM calls:
    python run_provider_inference.py --db ... --dry-run

    # Only process top-N most frequent terms (for cheap test):
    python run_provider_inference.py --db ... --limit 30

    # Skip terms already in enriched table (resume interrupted run):
    python run_provider_inference.py --db ... --resume
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from glossary_builder.candidates import Candidate
from glossary_builder.inference import infer_definition, DraftEntry
from glossary_builder.llm import LLMClient

console = Console()

# ── Provider domain brief (same as in run_provider_glossary.py) ───────────────
_PROVIDER_DOMAIN_BRIEF = (
    "A multilingual Telegram group focused on payment processing. "
    "These messages are from PSP PROVIDERS — companies and individuals "
    "OFFERING payment processing, acquiring, and gateway services to "
    "high-risk merchants. Verticals: iGaming, casinos, sportsbooks, "
    "forex, crypto, adult, nutra. "
    "Language: Russian, Ukrainian, English, often mixed. "
    "Focus: SELLER-SIDE language — providers describing their own "
    "coverage, approve rates, routing setups, onboarding terms, and "
    "payment methods they support."
)

# ── Output table schema ───────────────────────────────────────────────────────
_ENRICHED_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_glossary_enriched (
    term         TEXT PRIMARY KEY,
    relevant     INTEGER NOT NULL,   -- 0 or 1
    senses       TEXT,               -- JSON array of {definition, context_clues, example_message_ids}
    confidence   REAL,
    reasoning    TEXT,
    surface_forms TEXT,              -- JSON array (kept from intent_glossary)
    frequency    INTEGER,
    kind         TEXT,
    source_stage TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_enriched_relevant  ON provider_glossary_enriched(relevant);
CREATE INDEX IF NOT EXISTS idx_enriched_confidence ON provider_glossary_enriched(confidence);
"""


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_candidates(
    db_path: Path,
    limit: int | None,
    resume: bool,
) -> list[Candidate]:
    """Read intent_glossary → list of Candidate objects for inference."""
    conn = sqlite3.connect(db_path)
    try:
        # Terms already enriched (for --resume)
        already_done: set[str] = set()
        if resume:
            try:
                rows = conn.execute(
                    "SELECT term FROM provider_glossary_enriched"
                ).fetchall()
                already_done = {r[0] for r in rows}
                if already_done:
                    console.print(
                        f"[dim]--resume: skipping {len(already_done)} already-enriched terms[/]"
                    )
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet — nothing to skip

        query = "SELECT term, kind, frequency, surface_forms, examples FROM intent_glossary"
        if limit:
            query += f" ORDER BY frequency DESC LIMIT {limit}"
        else:
            query += " ORDER BY frequency DESC"

        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    candidates: list[Candidate] = []
    for term, kind, freq, surface_json, examples_json in rows:
        if not term:
            continue
        if resume and term in already_done:
            continue
        try:
            surfaces: list[str] = json.loads(surface_json or "[]")
        except (json.JSONDecodeError, TypeError):
            surfaces = [term]
        try:
            examples: list[dict] = json.loads(examples_json or "[]")
        except (json.JSONDecodeError, TypeError):
            examples = []

        candidates.append(Candidate(
            term            = term,
            kind            = kind or "phrase",
            frequency       = int(freq or 0),
            surface_forms   = surfaces,
            example_messages= examples,
        ))

    return candidates


def _persist_entry(conn: sqlite3.Connection, entry: DraftEntry, candidate: Candidate) -> None:
    """Write one DraftEntry to provider_glossary_enriched."""
    now = datetime.now(timezone.utc).isoformat()
    senses_json = json.dumps(
        [
            {
                "definition":          s.definition,
                "context_clues":       s.context_clues,
                "example_message_ids": s.example_message_ids,
            }
            for s in entry.senses
        ],
        ensure_ascii=False,
    )
    conn.execute(
        """
        INSERT INTO provider_glossary_enriched
            (term, relevant, senses, confidence, reasoning,
             surface_forms, frequency, kind, source_stage, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(term) DO UPDATE SET
            relevant=excluded.relevant,
            senses=excluded.senses,
            confidence=excluded.confidence,
            reasoning=excluded.reasoning,
            surface_forms=excluded.surface_forms,
            frequency=excluded.frequency,
            kind=excluded.kind,
            source_stage=excluded.source_stage,
            created_at=excluded.created_at
        """,
        (
            entry.term,
            int(entry.relevant),
            senses_json,
            entry.confidence,
            entry.reasoning,
            json.dumps(candidate.surface_forms, ensure_ascii=False),
            candidate.frequency,
            candidate.kind,
            "stage3_inference",
            now,
        ),
    )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--db", "db_path",
              default="PSP_provider_glossary_builder/data/provider_glossary.db",
              type=click.Path(exists=True),
              help="SQLite DB built by run_provider_glossary.py.")
@click.option("--limit", type=int, default=None,
              help="Process only top-N most frequent terms. None = all.")
@click.option("--min-confidence", type=float, default=0.0,
              help="Skip saving entries below this confidence threshold.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip terms already in provider_glossary_enriched (resume interrupted run).")
@click.option("--request-delay", type=float, default=0.5,
              help="Seconds to wait between LLM calls (default: 0.5).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print candidates without making LLM calls.")
def main(db_path, limit, min_confidence, resume, request_delay, dry_run):
    """Stage 3: enrich provider glossary terms with LLM definitions."""

    db = Path(db_path)

    # ── Load candidates ───────────────────────────────────────────────────────
    console.rule("[bold]PSP Provider Glossary — Stage 3 Inference")
    console.print(f"DB     : [cyan]{db}[/]")
    console.print(f"Limit  : {limit or 'all'}")
    console.print(f"Resume : {resume}")

    candidates = _load_candidates(db, limit, resume)
    console.print(f"\nCandidates to enrich: [green]{len(candidates)}[/]")

    if not candidates:
        console.print("[yellow]Nothing to process.[/]")
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[yellow]--dry-run: no LLM calls.[/]")
        table = Table(title="Candidates (top 25)", header_style="bold cyan")
        table.add_column("Term",      min_width=30)
        table.add_column("Kind",      width=12)
        table.add_column("Freq",      justify="right", width=6)
        table.add_column("Surfaces",  width=35)
        for c in candidates[:25]:
            table.add_row(
                c.term,
                c.kind,
                str(c.frequency),
                ", ".join(c.surface_forms[:4]),
            )
        console.print(table)
        console.print("\n[green]Dry run OK.[/] Remove --dry-run to enrich.")
        return

    # ── Init schema + LLM ─────────────────────────────────────────────────────
    conn = sqlite3.connect(db)
    conn.executescript(_ENRICHED_SCHEMA)
    conn.commit()

    llm = LLMClient()
    console.print(f"LLM    : [cyan]{llm.provider}[/] / [cyan]{llm.model}[/]\n")

    # ── Inference loop ─────────────────────────────────────────────────────────
    console.rule("[bold]Enriching terms")

    saved       = 0
    skipped_irr = 0
    skipped_low = 0
    errors      = 0
    total       = len(candidates)

    for n, candidate in enumerate(candidates, 1):
        try:
            entry: DraftEntry = infer_definition(
                candidate   = candidate,
                self_defs   = None,   # self-def mining deferred
                domain_brief= _PROVIDER_DOMAIN_BRIEF,
                llm         = llm,
            )
        except Exception as exc:
            console.print(
                f"  [red][{n}/{total}] ERROR[/] {candidate.term!r}: {exc}"
            )
            errors += 1
            time.sleep(request_delay)
            continue

        # Filter
        if not entry.relevant:
            skipped_irr += 1
            if n % 10 == 0 or n == total:
                console.print(
                    f"  [dim][{n}/{total}] irrelevant: {candidate.term!r} "
                    f"(conf={entry.confidence:.2f})[/]"
                )
            time.sleep(request_delay)
            continue

        if entry.confidence < min_confidence:
            skipped_low += 1
            time.sleep(request_delay)
            continue

        _persist_entry(conn, entry, candidate)
        saved += 1

        # Progress every 10
        if n % 10 == 0 or n == total:
            defn = entry.senses[0].definition if entry.senses else "(no senses)"
            console.print(
                f"  [green][{n}/{total}][/] [cyan]{candidate.term!r}[/]  "
                f"conf={entry.confidence:.2f}  \"{defn[:70]}\""
            )

        time.sleep(request_delay)

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("[bold]Done")
    console.print(f"  Saved (relevant)    : [green]{saved}[/]")
    console.print(f"  Skipped irrelevant  : [yellow]{skipped_irr}[/]")
    console.print(f"  Skipped low-conf    : [yellow]{skipped_low}[/]")
    console.print(f"  Errors              : [red]{errors}[/]")

    usage = llm.usage.to_dict()
    console.print()
    console.print(
        f"  LLM calls  : {usage['calls']}  |  "
        f"tokens: {usage['input_tokens']:,} in + {usage['output_tokens']:,} out  |  "
        f"est. [bold]${usage['estimated_cost_usd']:.4f}[/]"
    )
    console.print(f"\n[bold green]Enriched glossary saved → {db}[/]  "
                  f"(table: provider_glossary_enriched)")
    console.print(
        "\n[dim]Next: update load_provider_glossary() to read from "
        "provider_glossary_enriched instead of intent_glossary.[/]"
    )


if __name__ == "__main__":
    main()
