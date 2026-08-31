"""Command-line entry points.

All user-facing operations are exposed here as ``click`` subcommands of
the top-level ``cli`` group:

    build          — run the full glossary-building pipeline on a corpus
    list           — pretty-print glossary entries from the SQLite store
    show           — show full details for a single term
    export         — export the glossary as a JSON file
    extract-leads  — classify each message as lead/not-lead with evidence
    judge-leads    — second-pass LLM verdict on a leads JSON, output a CSV

Invoke as ``python run.py <subcommand> [options]`` (run.py is the entry
script at the project root) or ``python -m glossary_builder.cli``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import PipelineConfig
from .lead_extraction import (
    LeadExtractionConfig, LeadExtractor, evaluate, load_glossary,
)
from .lead_prompt_versions import BY_VERSION as LEAD_PROMPTS_BY_VERSION, BY_VERSION
from .llm import LLMClient
from .loader import load_messages
from .pipeline import run_pipeline
from .storage import GlossaryStore
from .research import TavilyWebSearch

console = Console()


@click.group()
def cli():
    """Build a domain glossary from a Telegram-export workbook."""


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to the .xlsx Telegram export.")
@click.option("--sheet", default=None, help="Sheet name; defaults to first sheet.")
@click.option("--sample-limit", type=int, default=None,
              help="Cap the number of messages loaded (cheap iteration).")
@click.option("--max-candidates", type=int, default=None,
              help="Only run LLM stages on the top-N candidates by frequency.")
@click.option("--dry-run", is_flag=True,
              help="Run Stages 1-2 only; skip all LLM calls.")
@click.option("--db", "db_path", default=None, type=click.Path(),
              help="Where to write the SQLite glossary (default: data/output/glossary.sqlite).")
@click.option("--enable-research", is_flag=True,
              help="Enable Stage 4 (requires a WebSearchTool implementation).")
def build(input_path, sheet, sample_limit, max_candidates, dry_run, db_path, enable_research):
    """Run the end-to-end pipeline."""
    cfg = PipelineConfig()
    if enable_research:
        cfg.research.enabled = True
    stats = run_pipeline(
        xlsx_path=input_path,
        sheet=sheet,
        sample_limit=sample_limit,
        max_candidates=max_candidates,
        config=cfg,
        dry_run=dry_run,
        db_path=db_path,
        web_search=TavilyWebSearch()
    )
    console.print()
    console.print("[bold]Run summary:")
    for k, v in stats.to_dict().items():
        console.print(f"  {k}: {v}")


@cli.command(name="list")
@click.option("--db", "db_path", default="data/output/glossary.sqlite",
              type=click.Path(exists=True))
@click.option("--min-confidence", type=float, default=0.5)
@click.option("--include-irrelevant", is_flag=True)
@click.option("--limit", type=int, default=50)
def list_entries(db_path, min_confidence, include_irrelevant, limit):
    """Print the glossary as a table."""
    store = GlossaryStore(db_path)
    entries = store.all_entries(
        relevant_only=not include_irrelevant,
        min_confidence=min_confidence,
    )
    entries = entries[:limit]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Term", style="cyan", no_wrap=True)
    table.add_column("Conf", justify="right")
    table.add_column("Definition (sense 1)")
    table.add_column("Source", style="dim")
    for e in entries:
        sense1 = e["senses"][0]["definition"] if e["senses"] else ""
        if len(sense1) > 70:
            sense1 = sense1[:67] + "..."
        table.add_row(
            e["term"],
            f"{e['confidence']:.2f}",
            sense1,
            e["source_stage"],
        )
    console.print(table)
    console.print(f"[dim]{len(entries)} entries shown.")


@cli.command()
@click.argument("term")
@click.option("--db", "db_path", default="data/output/glossary.sqlite",
              type=click.Path(exists=True))
def show(term, db_path):
    """Show full details for a single term."""
    store = GlossaryStore(db_path)
    entry = store.get(term)
    if entry is None:
        console.print(f"[red]Term not found: {term!r}")
        raise SystemExit(1)
    console.print_json(json.dumps(entry, ensure_ascii=False))


@cli.command(name="export")
@click.option("--db", "db_path", default="data/output/glossary.sqlite",
              type=click.Path(exists=True))
@click.option("--out", "out_path", default="data/output/glossary.json",
              type=click.Path())
@click.option("--min-confidence", type=float, default=0.0)
@click.option("--include-irrelevant", is_flag=True)
@click.option("--primary-sense-only", is_flag=True,
              help="Keep only the first sense per entry (cleaner for bot lookup).")
def export_json(db_path, out_path, min_confidence, include_irrelevant, primary_sense_only):
    """Export the glossary as a JSON file (for the bot's RAG layer)."""
    store = GlossaryStore(db_path)
    n = store.export_json(
        out_path,
        relevant_only=not include_irrelevant,
        min_confidence=min_confidence,
        primary_sense_only=primary_sense_only,
    )
    console.print(f"[green]Wrote {n} entries to {out_path}")


@cli.command(name="extract-leads")
@click.option("--fusion", is_flag=True, default=False,
              help="Use fusion panel instead of single model.")
@click.option("--fusion-panel", default="google/gemma-3-12b-it,deepseek/deepseek-v4-flash",
              help="Comma-separated panel models for fusion.")
@click.option("--fusion-judge", default="google/gemini-2.5-flash-lite",
              help="Judge model for fusion.")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to the corpus: .xlsx or .csv with the same column layout.")
@click.option("--sheet", default=None, help="For .xlsx: sheet name (defaults to first).")
@click.option("--glossary", "glossary_path",
              default="data/output/glossary_primary_sense_only.json",
              type=click.Path(exists=True),
              help="Glossary JSON (use the primary-sense-only export for cleanest lookup).")
@click.option("--output", "-o", "output_path",
              default="data/output/leads.json", type=click.Path())
@click.option("--sample", type=int, default=None,
              help="Cap on number of messages to process (cheap iteration).")
@click.option("--concurrency", type=int, default=4)
@click.option("--chunk-size", type=int, default=200,
              help="Executor batch size — lower = less memory, higher = less dispatch overhead.")
@click.option("--no-resume", is_flag=True,
              help="Disable resume: re-process even if a JSONL stream already exists.")
@click.option("--ground-truth", "gt_path", type=click.Path(exists=True), default=None,
              help="Optional CSV/JSON with message_id,is_lead columns for evaluation.")
@click.option("--prompt-version", type=click.Choice(list(BY_VERSION.keys())), default="v5",
              help="Lead-definition prompt version. v3=high precision, "
                   "v5=balanced (default), v6=strictest, v5-multilang=v5 with english and ukrainian support.")
@click.option("--rag", is_flag=True, default=False,
              help="Enable RAG few-shot injection.")
@click.option("--rag-db", default="data/rag/chroma",
              help="Path to Chroma persistent storage.")
@click.option("--rag-model", default="perplexity/pplx-embed-v1-0.6b",
              help="Embedding model via OpenRouter.")
@click.option("--rag-k", type=int, default=2,
              help="Number of examples per class (LEAD + NOT LEAD).")
def extract_leads(input_path, sheet, glossary_path, output_path, sample,
                  concurrency, chunk_size, no_resume, gt_path, prompt_version,
                  fusion, fusion_panel, fusion_judge,
                  rag, rag_db, rag_model, rag_k):
    """Classify each message as lead / not-lead with structured evidence."""

    console.rule("[bold]Loading corpus + glossary")
    if input_path.endswith(".csv"):
        messages = _load_csv(input_path)
    else:
        messages = load_messages(input_path, sheet=sheet)
    console.print(f"Loaded {len(messages)} messages.")

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename="fusion_debug.log",
        filemode="w",
        encoding="utf-8",
    )
    # Вимикаємо httpx шум
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Наші логи на DEBUG
    logging.getLogger("glossary_builder.fusion").setLevel(logging.DEBUG)

    # Exact-duplicate dedup BEFORE any LLM calls. Identical reposts/forwards
    # are common in these chats and would otherwise burn tokens (and inflate
    # lead counts) re-classifying the same text. Normalise (lowercase, strip,
    # collapse whitespace) and keep the FIRST occurrence of each unique text.
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    seen_texts: set[str] = set()
    deduped = []
    for m in messages:
        key = _norm_text(m.text)
        if key and key in seen_texts:
            continue
        if key:
            seen_texts.add(key)
        deduped.append(m)
    dropped = len(messages) - len(deduped)
    if dropped:
        console.print(
            f"[cyan]Dedup: dropped {dropped} exact-duplicate messages "
            f"(by normalised text) — {len(deduped)} unique remain."
        )
    messages = deduped

    glossary = load_glossary(glossary_path)
    console.print(f"Glossary: {len(glossary)} entries.")

    sample_ids = None
    if sample:
        # Sample from the middle of the corpus — first/last are typically
        # noisier (intros, summaries). Deterministic for reproducibility.
        candidates = [m.message_id for m in messages if m.text and len(m.text) >= 20]
        step = max(1, len(candidates) // sample)
        sample_ids = set(candidates[::step][:sample])
        console.print(f"Sampling {len(sample_ids)} messages.")

    llm = LLMClient()
    cfg = LeadExtractionConfig()
    cfg.lead_definition = LEAD_PROMPTS_BY_VERSION[prompt_version]
    console.print(f"Lead-definition prompt version: [cyan]{prompt_version}")
    fusion_config = None
    if fusion:
        from .fusion import FusionConfig
        cfg.validate_evidence_quote = False
        fusion_config = FusionConfig(
            panel_models=fusion_panel.split(","),
            judge_model=fusion_judge,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        console.print(f"[cyan]Fusion enabled: panel={fusion_panel}, judge={fusion_judge}")

    rag_config = None
    if rag:
        from .rag import RagConfig, RagIndex
        rag_config = RagConfig(
            db_path=rag_db,
            embedding_model=rag_model,
            api_key=os.environ["OPENAI_API_KEY"],
            k=rag_k,
        )
        rag_index = RagIndex(rag_config)
        if rag_index.count() == 0:
            console.print("[yellow]RAG index is empty — run build-rag first. RAG disabled.")
            rag_config = None
        else:
            console.print(f"[cyan]RAG enabled: {rag_index.count()} examples, k={rag_k}, model={rag_model}")

    extractor = LeadExtractor(glossary, llm, cfg, fusion_config=fusion_config, rag_config=rag_config)

    # The JSONL stream lives next to the final JSON output so a crash
    # mid-run can be resumed with the same command.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out.with_suffix(".jsonl")

    console.rule("[bold]Extracting leads")
    console.print(
        f"Streaming to {jsonl_path} (chunk_size={chunk_size}, concurrency={concurrency})."
    )
    if not no_resume and jsonl_path.exists():
        console.print(f"[yellow]Existing JSONL detected — resume enabled.")

    decisions = extractor.extract_batch(
        messages,
        sample_message_ids=sample_ids,
        concurrency=concurrency,
        chunk_size=chunk_size,
        stream_jsonl_path=jsonl_path,
        resume=not no_resume,
    )

    # Report how many messages the lexical pre-filter short-circuited (no LLM
    # call) versus how many actually reached the model this run.
    if extractor.prefiltered_count:
        llm_called = max(len(decisions) - extractor.prefiltered_count, 0)
        console.print(
            f"[cyan]Pre-filter: {extractor.prefiltered_count} messages skipped "
            f"(no domain term/intent signal) — {llm_called} reached the LLM this run."
        )
    if extractor.stage1_filtered_count:
        full_called = max(
            len(decisions) - extractor.prefiltered_count - extractor.stage1_filtered_count, 0
        )
        console.print(
            f"[cyan]Stage 1 micro-filter: {extractor.stage1_filtered_count} messages "
            f"screened out (not buyer intent) — {full_called} reached the full v5-short prompt."
        )

    # If we resumed, decisions only contains the NEW ones from this run.
    # Re-read the full JSONL stream so the final JSON export is complete.
    all_decisions_dicts = _read_jsonl(jsonl_path)
    leads = [d for d in all_decisions_dicts if d.get("is_lead")]
    console.print(
        f"Total in JSONL: {len(all_decisions_dicts)} messages — "
        f"flagged {len(leads)} leads ({100*len(leads)/max(len(all_decisions_dicts),1):.1f}%). "
        f"This run added {len(decisions)}."
    )

    out.write_text(json.dumps(all_decisions_dicts, ensure_ascii=False, indent=2),encoding="utf-8")
    console.print(f"[green]Wrote {output_path}")

    leads_path = out.parent / (out.stem + "_leads_only.json")
    leads_path.write_text(json.dumps(leads, ensure_ascii=False, indent=2),encoding="utf-8")
    console.print(f"[green]Leads-only export: {leads_path}")

    # Usage / cost summary.
    usage = llm.usage.to_dict()
    console.print()
    console.print(f"LLM cost (this run): ~${usage['estimated_cost_usd']:.3f} "
                  f"({usage['calls']} calls, {usage['input_tokens']} input + {usage['output_tokens']} output tokens)")

    by_stage = usage.get("by_stage") or {}
    if by_stage:
        tbl = Table(title="Tokens by stage")
        tbl.add_column("stage")
        tbl.add_column("calls", justify="right")
        tbl.add_column("input", justify="right")
        tbl.add_column("output", justify="right")
        tbl.add_column("~$", justify="right")
        for name, s in by_stage.items():
            tbl.add_row(
                name,
                str(s.get("calls", 0)),
                str(s.get("input_tokens", 0)),
                str(s.get("output_tokens", 0)),
                f"{s.get('estimated_cost_usd', 0.0):.4f}",
            )
        console.print(tbl)

    if gt_path:
        console.rule("[bold]Evaluation vs ground truth")
        gt = _load_ground_truth(gt_path)
        from .lead_extraction import LeadDecision as _LD
        # Convert dicts back to a thin shape that evaluate() understands.
        report = evaluate(
            [_LD(**{k: v for k, v in d.items() if k in _LD.__annotations__})
             for d in all_decisions_dicts],
            gt,
        )
        console.print_json(json.dumps(report.to_dict()))


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL stream into a list of dicts."""
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_csv(path: str):
    """Load a Telegram-export CSV (semicolon-delimited, same columns as XLSX).

    Uses ``utf-8-sig`` so the BOM that Excel-exported CSVs commonly carry
    doesn't attach itself to the first column name.
    """
    import csv
    from datetime import datetime
    from .loader import Message

    rows: list[Message] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            text = r.get("text") or ""
            if not text.strip():
                continue
            try:
                mid = int(r["message_id"]) if r.get("message_id") else None
            except (ValueError, TypeError):
                mid = None
            try:
                gid = int(r["group_id"]) if r.get("group_id") else None
            except (ValueError, TypeError):
                gid = None
            try:
                uid = int(r["user_id"]) if r.get("user_id") else None
            except (ValueError, TypeError):
                uid = None
            ts = None
            d = r.get("date") or ""
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    ts = datetime.strptime(d, fmt)
                    break
                except ValueError:
                    pass
            rows.append(Message(
                message_id=mid,
                date=ts,
                text=text,
                group_id=gid,
                user_id=uid,
                username=(r.get("username") or "").strip() or None,
                first_name=(r.get("first_name") or "").strip() or None,
                last_name=(r.get("last_name") or "").strip() or None,
            ))
    return rows


def _load_ground_truth(path: str) -> dict[int, bool]:
    """Load ground truth as { message_id: is_lead } from CSV or JSON."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {int(k): bool(v) for k, v in data.items()}
        return {int(r["message_id"]): bool(r.get("is_lead")) for r in data}
    import csv
    out: dict[int, bool] = {}
    with open(path, newline="", encoding="utf-8") as f:
        # Try comma first, then semicolon.
        sniff = f.read(4096)
        f.seek(0)
        delim = ";" if sniff.count(";") > sniff.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            mid_s = r.get("message_id") or r.get("id")
            if not mid_s:
                continue
            try:
                mid = int(mid_s)
            except ValueError:
                continue
            val = (r.get("is_lead") or r.get("lead") or r.get("label") or "").strip().lower()
            out[mid] = val in ("1", "true", "yes", "y", "t", "lead")
    return out


_JUDGE_SYSTEM = """\
You are an independent reviewer of automatically-extracted sales leads from
a Russian-language Telegram chat about high-risk payment processing
(iGaming, casinos, sportsbooks, forex, crypto, adult).

For each lead I show you, give an INDEPENDENT verdict: is this a REAL_LEAD
or a MISTAKE? Apply this lead definition strictly:

LEAD = a message from a BUYER in this ecosystem (iGaming operator, affiliate,
merchant, platform) who is ACTIVELY SHOPPING for a payment service AND
signals a concrete need: a specific vertical, geo, payment method, volume,
or specific provider being evaluated.

NOT a lead (most common false positives):
  - 'Кто работал с X?' / 'отзывы' / 'кто работает с X' / 'кто использует X' /
    'опыт работы с X' — due-diligence / networking. Only a lead if the
    speaker ALSO states their own concrete need.
  - Market-pulse questions about the ecosystem ('ситуация с X', 'у всех
    отлетела SEPA?').
  - Seller voice: PSP reps pitching, BizDev intros offering services,
    pricing/margin talk, 'у нас есть X', product/capability pitches even
    when ending in 'напишите в ЛС'.
  - Seller inventory pitches: geo+method catalogs with Telegram CTAs.
  - Affiliate/traffic/media buying: 'ищу трафик', 'ищу объёмы'.
  - Advertiser/operator searches: 'реклы' / 'реклов' / 'прямой рекл' =
    advertisers, NOT payment processors. 'Нужны реклы которые принимают гео X'
    = advertiser request, NOT a payment lead. MISTAKE.
  - Brand/operator contact searches: 'контакты казино X' / 'contacts for
    [Brand]' / 'looking for SE/DK/MGA licensed brands/operators' — seeking
    a gaming brand, NOT a payment service. MISTAKE.
  - Crypto wallet for personal use: 'анонимный крипто-кошелёк' / 'crypto
    wallet with AML check' — personal finance tool, NOT payment processing.
    MISTAKE.
  - Corporate/legal services: 'компания на Коста-Рике' / 'готовая компания' /
    'регистрация юрлица' / 'S.A.' — legal/corporate service, NOT payment
    processing. MISTAKE.
  - Empty thread-joining: 'мне тоже надо' / 'и мне' / '+1' WITHOUT any
    payment-specific context in the same message — NOT a lead. MISTAKE.
  - Hiring / staffing.
  - Adjacent industries: consulting, M&A, investing, gaming platform/license.
  - Complaints / blacklist callouts.
  - Personal/one-off transactions (paying for hotels, visas).
  - Bot summaries, jokes, conversation.

UNCONDITIONAL LEADS (always REAL_LEAD even if short):
  - 'у кого есть [thing speaker wants]' / multi-geo possession questions
  - 'я тоже в поиске' / '+1' / 'присоединюсь к запросу' (thread-joining)
  - Explicit 'ищу/нужен/нужно' + concrete need

Output strict JSON only.
"""


_JUDGE_USER_TEMPLATE = """\
LEAD CANDIDATE:
  message_id: {message_id}
  @{username}  [{timestamp}]
  text: {text}

EXTRACTOR's claim:
  lead_type: {lead_type}
  intent: {intent}
  interest_level: {interest_level}
  vertical: {vertical}
  geo: {geo}
  payment_methods_mentioned: {methods}
  evidence_quote: {evidence_quote}
  rationale: {rationale}

Return JSON:
{{
  "verdict": "REAL_LEAD" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}
"""

_JUDGE_SYSTEM_V2 = """\
You are an independent reviewer of automatically-extracted sales leads from
a Telegram chat about high-risk payment processing (iGaming, casinos,
sportsbooks, forex, crypto, adult). Messages are in Russian, Ukrainian,
English, or mixed.

For each lead I show you, give an INDEPENDENT verdict: is this a REAL_LEAD
or a MISTAKE? Apply this lead definition strictly:

LEAD = a message from a BUYER (iGaming operator, affiliate, merchant,
platform) ACTIVELY SHOPPING for a payment SERVICE (PSP, acquirer, gateway,
orchestrator, payment method integration) AND signals a concrete need:
specific vertical, geo, payment method, volume, or provider being evaluated.

═══════════════════════════════════════════════════════
ALWAYS MISTAKE — these patterns are NEVER leads:
═══════════════════════════════════════════════════════

AFFILIATE / MEDIA BUYING INFRASTRUCTURE (never REAL_LEAD):
  - '#buying' / '#WTB' + geo list + 'MB only' / 'direct MB' / 'only MB' —
    media buyers sourcing merchant banks for traffic funnels, NOT casino
    operators seeking PSP.
  - 'лейка' / 'приемка' + geo WITHOUT explicit casino/platform context —
    traffic infrastructure for media buying, NOT payment-service procurement.
  - 'фанел' / 'funnel' + MB — media buying funnel infrastructure.
  - 'филлер' — payment fill partner for media buying, NOT PSP procurement.
  - 'CPA test' / 'CRG' / 'cap for testing' / 'refs' — affiliate marketing
    terms indicating traffic sourcing, NOT payment procurement.
  - 'looking for MB' / 'only MB' / 'direct MB' WITHOUT explicit vertical
    (casino/igaming/sportsbook) — merchant bank sourcing for affiliates,
    NOT a PSP lead.
    
P2P CRYPTO EXCHANGE (most common false positive):
  'нужен USDT TRC20', 'need USDT exchanger', 'INR to USDT',
  'need bulk USDT', 'интересует тезер TRC20' — buying/selling
  cryptocurrency is currency exchange, NOT payment-service procurement.
  Large volume alone does NOT make it a lead.
  Exception: 'нужен USDT settlement для казино' WITH explicit
  business vertical IS a lead.

BANK ACCOUNT / DROP PURCHASES:
  'need indian bank accounts', 'need JK IDBI BOI accounts',
  'нужна приемка по КЗ физ счета', 'need saving/corporate accounts',
  'нужен абанк' — buying bank accounts or acquiring financial
  infrastructure is NOT payment-service procurement.

AGENT / CARDHOLDER RECRUITMENT:
  'looking for Indian P2P agents/traders', 'need powerful Indian agent
  to provide accounts', 'searching for cardholders', 'looking for
  Four Parties agents' — recruiting money infrastructure partners
  is NOT a buyer looking for PSP.

DUE-DILIGENCE / NETWORKING:
  'Кто работал с X?' / 'отзывы о X' / 'кто использует X' — reputation
  check without stating own procurement need.

MARKET-PULSE:
  'ситуация с X', 'у всех отлетела SEPA?' — research, not procurement.

SELLER VOICE:
  'у нас есть X', 'мы предлагаем', capability lists with Telegram CTA,
  sell-side aggregator pitches.

WRONG VERTICAL:
  - Affiliate/traffic: 'ищу трафик', 'ищу реклов', 'объёмы гембл'
  - Brand/operator search: 'контакты казино X', 'MGA licensed brands'
  - Corporate/legal: 'регистрация юрлица', 'готовая компания'
  - Personal crypto wallet, hiring, consulting, M&A, complaints,
    bot summaries, personal transactions (hotels, visas)

═══════════════════════════════════════════════════════
ALWAYS REAL_LEAD — never return MISTAKE for these:
═══════════════════════════════════════════════════════
  - Explicit 'ищу PSP/платежку/процессинг' + vertical/geo/volume
  - 'у кого есть [payment solution]' possession questions
  - '#buying' + specific geo/method (MB, MB only, specific countries)
  - '+1' / 'присоединюсь к запросу' in payment context
  - 'looking for [geo] from MB' — merchant bank sourcing

Output strict JSON only.
"""

JUDGE_PROMPTS_BY_VERSION = {
    "v1": _JUDGE_SYSTEM,
    "v2": _JUDGE_SYSTEM_V2,
}

@cli.command(name="judge-leads")
@click.option(
    "--judge-version",
    type=click.Choice(list(JUDGE_PROMPTS_BY_VERSION.keys())),
    default="v1",
    help="Judge prompt version. v1=original, v2=stricter FP filtering."
)
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to leads JSON (e.g. data/output/foo_leads_only.json).")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(),
              help="Where to write the marked CSV.")
@click.option("--concurrency", type=int, default=6)
@click.option("--chunk-size", type=int, default=50)
@click.option("--resume-jsonl", type=click.Path(), default=None,
              help="Optional JSONL stream for resume. Defaults to <output>.judge.jsonl.")
def judge_leads(input_path, output_path, concurrency, chunk_size, resume_jsonl,judge_version):
    """Have an independent LLM verdict each flagged lead as REAL_LEAD or MISTAKE."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import csv as _csv

    judge_system = JUDGE_PROMPTS_BY_VERSION[judge_version]
    leads = json.loads(Path(input_path).read_text(encoding="utf-8"))
    console.print(f"Loaded {len(leads)} candidate leads from {input_path}")

    jsonl_path = Path(resume_jsonl) if resume_jsonl else Path(output_path).with_suffix(".judge.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support.
    already_done: dict[int, dict] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec.get("message_id"), int):
                    already_done[rec["message_id"]] = rec
            except json.JSONDecodeError:
                continue
        if already_done:
            console.print(f"[yellow]Resuming: {len(already_done)} already judged in {jsonl_path}")

    remaining = [d for d in leads if d.get("message_id") not in already_done]
    console.print(f"Judging {len(remaining)} new leads ({len(already_done)} already done).")

    llm = LLMClient()
    # The judge can run on its own (typically stronger / cross-checking) model
    # via JUDGE_MODEL; otherwise it falls back to the resolved GLOSSARY_MODEL.
    judge_model = os.getenv("JUDGE_MODEL")
    if judge_model:
        llm.model = judge_model
        console.print(f"Judge model: {judge_model}")
    llm.set_stage("lead_judge")
    stream_file = jsonl_path.open("a", encoding="utf-8")

    def _judge_one(d: dict) -> dict:
        user = _JUDGE_USER_TEMPLATE.format(
            message_id=d.get("message_id", "?"),
            username=d.get("username") or "anon",
            timestamp=d.get("timestamp", "?"),
            text=(d.get("text") or "").replace("\n", " ").strip()[:500],
            lead_type=d.get("lead_type"),
            intent=d.get("intent"),
            interest_level=d.get("interest_level"),
            vertical=d.get("vertical"),
            geo=d.get("geo"),
            methods=d.get("payment_methods_mentioned"),
            evidence_quote=d.get("evidence_quote", "")[:200],
            rationale=d.get("rationale", "")[:200],
        )
        try:
            data, _ = llm.complete_json(judge_system, user, max_tokens=200)
            verdict = str(data.get("verdict", "")).strip().upper()
            reason = str(data.get("reason", "")).strip()
        except Exception as exc:  # noqa: BLE001
            verdict = "ERROR"
            reason = f"judge call failed: {exc}"
        return {
            "message_id": d.get("message_id"),
            "verdict": verdict if verdict in ("REAL_LEAD", "MISTAKE") else "REVIEW",
            "judge_reason": reason,
        }

    from tqdm import tqdm
    progress = tqdm(total=len(remaining), desc="Judging")
    try:
        for chunk_start in range(0, len(remaining), chunk_size):
            chunk = remaining[chunk_start:chunk_start + chunk_size]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_judge_one, d) for d in chunk]
                for fut in as_completed(futures):
                    progress.update(1)
                    try:
                        rec = fut.result()
                    except Exception:
                        continue
                    if rec["message_id"] is None:
                        continue
                    already_done[rec["message_id"]] = rec
                    stream_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stream_file.flush()
    finally:
        progress.close()
        stream_file.close()

    console.print(f"[green]Wrote judge stream: {jsonl_path}")

    # Build the marked CSV.
    COLUMNS = [
        "message_id", "verdict", "judge_reason",
        "date", "username", "text",
        "confidence", "lead_type", "intent", "interest_level",
        "vertical", "geo", "payment_methods_mentioned",
        "evidence_quote", "rationale", "glossary_terms_seen",
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = _csv.writer(f, delimiter=";", quoting=_csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        for d in leads:
            mid = d.get("message_id")
            judge = already_done.get(mid, {"verdict": "REVIEW", "judge_reason": ""})
            row = []
            for col in COLUMNS:
                if col == "verdict":
                    v = judge.get("verdict", "REVIEW")
                elif col == "judge_reason":
                    v = judge.get("judge_reason", "")
                else:
                    v = d.get(col, "") if col != "date" else d.get("timestamp", "")
                if isinstance(v, list):
                    v = "; ".join(str(x) for x in v)
                elif v is None:
                    v = ""
                else:
                    v = str(v)
                v = v.replace("\r\n", " | ").replace("\n", " | ")
                row.append(v)
            writer.writerow(row)
    console.print(f"[green]Wrote marked CSV: {output_path}")

    # Summary.
    real = sum(1 for d in leads if already_done.get(d.get("message_id"), {}).get("verdict") == "REAL_LEAD")
    mistake = sum(1 for d in leads if already_done.get(d.get("message_id"), {}).get("verdict") == "MISTAKE")
    review = len(leads) - real - mistake
    console.print()
    console.print(f"Verdict summary:  REAL_LEAD={real}  MISTAKE={mistake}  REVIEW={review}")
    console.print(f"Implied precision: {100*real/max(real+mistake,1):.1f}%")
    usage = llm.usage.to_dict()
    console.print(f"Judge LLM cost (this run): ~${usage['estimated_cost_usd']:.3f}")

@cli.command(name="build-rag")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="CSV with labeled examples. Must have 'text' and 'label' columns.")
@click.option("--output", "-o", "db_path", default="data/rag/chroma",
              help="Path to Chroma persistent storage.")
@click.option("--label-col", default="label",
              help="Column name for label (0/1 or true/false).")
@click.option("--text-col", default="text",
              help="Column name for message text.")
@click.option("--id-col", default="message_id",
              help="Column name for message ID (optional).")
@click.option("--embedding-model", default="perplexity/pplx-embed-v1-0.6b",
              help="Embedding model to use via OpenRouter.")
@click.option("--dedup", is_flag=True, default=True,
              help="Deduplicate by normalized text before indexing.")
def build_rag(input_path, db_path, label_col, text_col, id_col,
              embedding_model, dedup):
    """Build a RAG index from a labeled CSV for few-shot injection."""
    import pandas as pd
    from .rag import RagConfig, RagIndex

    console.rule("[bold]Building RAG index")

    df = pd.read_csv(input_path, sep=None, engine="python")

    console.print(f"Loaded {len(df)} rows from {input_path}")

    # Validate columns
    if text_col not in df.columns:
        console.print(f"[red]Column '{text_col}' not found. Available: {list(df.columns)}")
        return
    if label_col not in df.columns:
        console.print(f"[red]Column '{label_col}' not found. Available: {list(df.columns)}")
        return

    # Drop rows without text or label
    df = df.dropna(subset=[text_col, label_col])
    console.print(f"After dropping nulls: {len(df)} rows")

    # Dedup by normalized text
    if dedup:
        def _norm(t):
            import re
            return re.sub(r"\s+", " ", str(t).strip().lower())
        df["_norm"] = df[text_col].apply(_norm)
        before = len(df)
        df = df.drop_duplicates(subset="_norm")
        console.print(f"After dedup: {len(df)} rows (dropped {before - len(df)})")

    # Build examples list
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "text": str(row[text_col]),
            "label": row[label_col],
            "message_id": str(row[id_col]) if id_col in df.columns else str(_),
        })

    label_counts = df[label_col].value_counts()
    console.print(f"Label distribution: {dict(label_counts)}")

    # Build index
    cfg = RagConfig(
        db_path=db_path,
        embedding_model=embedding_model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    index = RagIndex(cfg)
    index.build(examples)
    console.print(f"[green]RAG index built: {index.count()} examples in {db_path}")

if __name__ == "__main__":
    cli()
