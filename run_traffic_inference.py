"""Stage 3 enrichment runner for the Traffic vertical.

For each term extracted by run_traffic_glossary.py (table ``intent_glossary``),
calls the LLM to:
  - Write a concise definition with context clues
  - Decide whether the term is domain-relevant (relevant=1/0)
  - Score confidence

Results are written to ``traffic_glossary_enriched`` in the same SQLite DB.
The Traffic classifier's pre-filter (intent_filter_traffic.py) prefers the
enriched table (relevant=1 only) over the raw intent_glossary.

Usage:
    # Dry run — show what would be processed, no LLM calls
    python run_traffic_inference.py --db Traffic_classifier/data/traffic_glossary.db --dry-run

    # Test on first 30 terms
    python run_traffic_inference.py --db Traffic_classifier/data/traffic_glossary.db --limit 30

    # Full run
    python run_traffic_inference.py --db Traffic_classifier/data/traffic_glossary.db

    # Resume interrupted run (skip already-enriched terms)
    python run_traffic_inference.py --db Traffic_classifier/data/traffic_glossary.db --resume
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from glossary_builder.llm import LLMClient

console = Console()

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_DB = "Traffic_classifier/data/traffic_glossary.db"

# ---------------------------------------------------------------------------
# Domain brief (same as run_traffic_glossary.py — keep in sync)
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
# Enriched table schema
# ---------------------------------------------------------------------------

_ENRICHED_SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_glossary_enriched (
    term            TEXT PRIMARY KEY,
    surface_forms   TEXT,       -- JSON array  (copied from intent_glossary)
    frequency       INTEGER,    -- (copied from intent_glossary)
    relevant        INTEGER,    -- 0 or 1 (LLM decision)
    senses          TEXT,       -- JSON array of {definition, context_clues}
    confidence      REAL,       -- 0.0 .. 1.0
    reasoning       TEXT,       -- LLM reasoning (one or two sentences)
    enriched_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_enriched_relevant
    ON traffic_glossary_enriched(relevant);
"""

# ---------------------------------------------------------------------------
# LLM prompts (traffic-adapted from inference.py)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a domain analyst building a glossary for an automated agent that scans
Telegram chats looking for affiliate-traffic leads: people who BUY or SELL
traffic in high-risk verticals (gambling, casino, betting, crypto, forex,
nutra, adult, dating, igaming).

The agent will use the glossary to recognise domain jargon, acronyms, slang
and brand names. Bad glossary entries cause bad classifications. Be strict:
if the evidence is weak, mark relevant=false.

For each term you analyse:
  1. Decide whether it is *domain-relevant* for affiliate traffic. Relevant:
       - Affiliate / arbitrage jargon (лить трафик, оффер, крео, ГЕО, апрув...)
       - Traffic sources (FB, TikTok, SEO, push, native, in-app...)
       - Pricing models (CPA, RevShare, FTD, CPL, hybrid...)
       - Roles (арбитражник, вебмастер, медиабаер, media buyer, affiliate...)
       - Niches (gambling, nutra, adult, dating, crypto, forex, betting...)
       - Affiliate networks and platforms
     NOT relevant:
       - Generic business words without affiliate-specific meaning
       - Payment / PSP terminology unrelated to affiliate traffic
       - User handles, bot artefacts, command fragments
       - Technical IT terms (DDoS, bandwidth, API) not used as traffic jargon
  2. Provide one or more *senses* with a short definition and context clues.
  3. Score *confidence* 0.0..1.0 (1.0 = certain, multiple clear examples).

Output strict JSON only. No prose outside the JSON block.\
"""

_USER_TEMPLATE = """\
DOMAIN CONTEXT:
{brief}

TERM: {term}
Surface forms seen in corpus: {surfaces}
Total occurrences in sample: {freq}

EXAMPLE MESSAGES (with message_id, sender, text):
{examples}

Return JSON matching this schema exactly:
{{
  "term": "{term}",
  "relevant": <bool>,
  "senses": [
    {{
      "definition": "<concise English definition, <=160 chars>",
      "context_clues": ["<word>", "<word>", ...],
      "example_message_ids": [<int>, ...]
    }}
  ],
  "confidence": <float 0..1>,
  "reasoning": "<one or two sentences, English>"
}}

If the term is not domain-relevant, return relevant=false, empty senses array,
and explain briefly in reasoning.\
"""

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class _Term:
    term: str
    surface_forms: list[str]
    frequency: int
    examples: list[dict]


def _load_terms(db_path: Path, *, resume: bool, limit: Optional[int]) -> list[_Term]:
    """Read terms from intent_glossary, optionally skipping already-enriched ones."""
    conn = sqlite3.connect(db_path)
    try:
        # Collect already-enriched terms if resuming
        enriched_terms: set[str] = set()
        if resume:
            try:
                rows = conn.execute(
                    "SELECT term FROM traffic_glossary_enriched"
                ).fetchall()
                enriched_terms = {r[0] for r in rows}
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet

        query = "SELECT term, surface_forms, frequency, examples FROM intent_glossary ORDER BY frequency DESC"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    terms: list[_Term] = []
    for term, surface_json, freq, examples_json in rows:
        if resume and term in enriched_terms:
            continue
        try:
            surfaces = json.loads(surface_json or "[]")
        except (json.JSONDecodeError, TypeError):
            surfaces = []
        try:
            examples = json.loads(examples_json or "[]")
        except (json.JSONDecodeError, TypeError):
            examples = []
        terms.append(_Term(term=term, surface_forms=surfaces, frequency=freq, examples=examples))
        if limit is not None and len(terms) >= limit:
            break

    return terms


def _format_examples(examples: list[dict], limit: int = 15) -> str:
    lines = []
    for ex in examples[:limit]:
        mid  = ex.get("message_id", "?")
        user = ex.get("username") or "anon"
        text = (ex.get("text") or "").replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:277] + "..."
        lines.append(f"  - msg_id={mid} @{user}: {text}")
    return "\n".join(lines) if lines else "  (none)"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _enrich_term(term: _Term, llm: LLMClient, domain_brief: str) -> dict:
    """Call LLM for one term; return parsed result dict."""
    user = _USER_TEMPLATE.format(
        brief=domain_brief,
        term=term.term,
        surfaces=", ".join(term.surface_forms) or term.term,
        freq=term.frequency,
        examples=_format_examples(term.examples),
    )
    data, _ = llm.complete_json(_SYSTEM, user)
    return data


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def _persist_enriched(db_path: Path, term: str, surface_forms: list[str],
                       frequency: int, data: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    senses = data.get("senses") or []
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_ENRICHED_SCHEMA)
        conn.execute(
            """
            INSERT INTO traffic_glossary_enriched
                (term, surface_forms, frequency, relevant, senses, confidence, reasoning, enriched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                surface_forms = excluded.surface_forms,
                frequency     = excluded.frequency,
                relevant      = excluded.relevant,
                senses        = excluded.senses,
                confidence    = excluded.confidence,
                reasoning     = excluded.reasoning,
                enriched_at   = excluded.enriched_at
            """,
            (
                term,
                json.dumps(surface_forms, ensure_ascii=False),
                frequency,
                1 if data.get("relevant") else 0,
                json.dumps(senses, ensure_ascii=False),
                float(data.get("confidence") or 0.0),
                str(data.get("reasoning") or "").strip(),
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_summary(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        total     = conn.execute("SELECT COUNT(*) FROM traffic_glossary_enriched").fetchone()[0]
        relevant  = conn.execute("SELECT COUNT(*) FROM traffic_glossary_enriched WHERE relevant=1").fetchone()[0]
        high_conf = conn.execute(
            "SELECT COUNT(*) FROM traffic_glossary_enriched WHERE relevant=1 AND confidence>=0.8"
        ).fetchone()[0]

        top = conn.execute(
            "SELECT term, frequency, confidence FROM traffic_glossary_enriched "
            "WHERE relevant=1 ORDER BY frequency DESC LIMIT 20"
        ).fetchall()
    finally:
        conn.close()

    from rich.table import Table
    console.print(f"\nEnriched: [green]{total}[/] terms total  "
                  f"[green]{relevant}[/] relevant  "
                  f"[green]{high_conf}[/] high-confidence (≥0.8)")

    if top:
        table = Table(show_header=True, header_style="bold",
                      title="Top 20 relevant terms by frequency")
        table.add_column("Term", style="cyan")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right")
        for term, freq, conf in top:
            table.add_row(term, str(freq), f"{conf:.2f}")
        console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--db", "db_path",
    default=_DEFAULT_DB,
    type=click.Path(),
    show_default=True,
    help="SQLite DB produced by run_traffic_glossary.py.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the top-N terms by frequency (useful for quick tests).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip terms that are already in traffic_glossary_enriched.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load terms and print stats, but skip LLM calls and DB writes.",
)
@click.option(
    "--delay",
    type=float,
    default=0.2,
    show_default=True,
    help="Seconds to wait between LLM calls (rate-limit friendliness).",
)
def main(db_path, limit, resume, dry_run, delay):
    """Run Stage 3 enrichment for the Traffic vertical.

    Reads terms from intent_glossary and writes definitions + relevance flags
    to traffic_glossary_enriched.
    """
    db = Path(db_path)
    if not db.exists():
        console.print(f"[red]DB not found: {db}[/]")
        console.print("Run run_traffic_glossary.py first.")
        raise SystemExit(1)

    # ── Load terms ────────────────────────────────────────────────────────────
    terms = _load_terms(db, resume=resume, limit=limit)
    console.print(
        f"Terms to enrich: [green]{len(terms)}[/]"
        + (" (resume mode — skipping already enriched)" if resume else "")
        + (f" (limited to first {limit})" if limit else "")
    )

    if dry_run:
        console.print("[yellow]--dry-run: skipping LLM calls.[/]")
        for t in terms[:10]:
            console.print(f"  would enrich: [cyan]{t.term!r}[/] freq={t.frequency}")
        if len(terms) > 10:
            console.print(f"  ... and {len(terms) - 10} more")
        return

    if not terms:
        console.print("[yellow]Nothing to enrich. All terms already processed?[/]")
        _print_summary(db)
        return

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = LLMClient()
    console.print(
        f"LLM provider: [cyan]{llm.provider}[/]  model: [cyan]{llm.model}[/]"
    )

    # ── Enrich loop ───────────────────────────────────────────────────────────
    errors = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching terms...", total=len(terms))

        for term in terms:
            progress.update(task, description=f"[cyan]{term.term[:40]}[/]")
            try:
                data = _enrich_term(term, llm, _TRAFFIC_DOMAIN_BRIEF)
                _persist_enriched(db, term.term, term.surface_forms, term.frequency, data)
            except Exception as exc:
                errors += 1
                console.print(f"[red]ERROR[/] term={term.term!r}: {exc}")
            finally:
                progress.advance(task)
                if delay > 0:
                    time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(f"\nDone. Errors: [{'red' if errors else 'green'}]{errors}[/]")
    _print_summary(db)

    usage = llm.usage.to_dict()
    console.print(
        f"\nLLM usage: {usage['calls']} calls, "
        f"{usage['input_tokens']} in + {usage['output_tokens']} out tokens "
        f"(est. ${usage['estimated_cost_usd']:.4f})."
    )

    if not resume and errors == 0:
        console.print(
            "\n[bold green]Enrichment complete.[/] Next step: build the classifier.\n"
            "  See Traffic_classifier/ for traffic_prompt.py, traffic_classifier.py"
        )


if __name__ == "__main__":
    main()
