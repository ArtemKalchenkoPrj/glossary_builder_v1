"""End-to-end orchestrator.

Reads messages → extracts candidates → mines self-definitions → runs LLM
inference → optional web research → adversarial validation → writes glossary.

Every stage writes its intermediate artifact to disk so the pipeline is
inspectable and resumable.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rich.console import Console
from tqdm import tqdm

from .brand_mining import confirm_brands, mine_brands
from .candidates import Candidate, extract_candidates, summarize
from .config import PipelineConfig
from .dedup import dedupe_parent_child, prune_redundant_senses
from .gazetteer import scan as scan_gazetteer
from .inference import DraftEntry, infer_definition
from .lemma import cluster_candidates
from .llm import LLMClient
from .loader import Message, load_messages, corpus_stats
from .phrases import extract_phrases
from .research import NoOpWebSearch, WebSearchTool, needs_research, research_term, build_search_backend
from .self_definitions import SelfDefinition, mine_self_definitions
from .storage import GlossaryStore
from .validator import (
    Critique, dedupe_paraphrase_senses,
    enrich_with_missing_senses, validate_entry,
)


console = Console()


@dataclass
class RunStats:
    candidates_extracted: int = 0
    phrase_candidates_extracted: int = 0
    gazetteer_candidates_extracted: int = 0
    brand_candidates_extracted: int = 0
    self_definitions_found: int = 0
    drafts_completed: int = 0
    research_invocations: int = 0
    validated: int = 0
    enriched: int = 0
    relevant_entries: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_by_stage: dict = None  # populated at end of run

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["estimated_cost_usd"] = round(self.estimated_cost_usd, 4)
        return d


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),encoding="utf-8")


def run_pipeline(
    xlsx_path: str | Path,
    *,
    sheet: str | None = None,
    sample_limit: int | None = None,
    max_candidates: int | None = None,
    config: PipelineConfig | None = None,
    web_search: WebSearchTool | None = None,
    dry_run: bool = False,
    db_path: str | Path | None = None,
) -> RunStats:
    """Run the full pipeline.

    Args:
        xlsx_path: input Telegram-export workbook.
        sheet: sheet name (defaults to first sheet).
        sample_limit: cap messages loaded — use this to keep iteration cheap.
        max_candidates: only run LLM stages on the top-N candidates by frequency.
        config: pipeline configuration (defaults are sensible).
        web_search: WebSearchTool implementation (defaults to NoOpWebSearch).
        dry_run: skip LLM stages — only Stages 1 and 2 (no API calls).
        db_path: where to write the glossary SQLite DB.
    """
    cfg = config or PipelineConfig()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = RunStats()

    db_path = Path(db_path) if db_path else out / "glossary.sqlite"
    store = GlossaryStore(db_path)

    # ---- Load ----
    console.rule("[bold]Loading messages")
    messages = load_messages(xlsx_path, sheet=sheet, limit=sample_limit)
    console.print(corpus_stats(messages))
    _write_json(out / "01_corpus_stats.json", corpus_stats(messages))

    # ---- Stage 1: candidates ----
    console.rule("[bold]Stage 1 — candidate extraction (rules)")
    candidates = extract_candidates(messages, cfg.candidates)
    stats.candidates_extracted = len(candidates)
    console.print(summarize(candidates))
    _write_json(
        out / "02_candidates.json",
        [c.to_dict() for c in candidates],
    )

    # ---- Stage 1b: LLM-assisted phrase extraction (Russian + multi-word) ----
    # Only enabled when LLM is available (so --dry-run skips it).
    phrase_candidates: list[Candidate] = []
    llm: LLMClient | None = None
    if cfg.phrases.enabled and not dry_run:
        console.rule("[bold]Stage 1b — LLM-assisted phrase extraction")
        llm = LLMClient(cfg.llm)
        phrase_candidates = extract_phrases(
            messages=messages,
            llm=llm,
            domain_brief=cfg.domain_brief,
            config=cfg.phrases,
        )
        # Lemma-cluster duplicates: мерчант / мерчанта / мерчантов -> one.
        before_cluster = len(phrase_candidates)
        phrase_candidates = cluster_candidates(phrase_candidates)
        stats.phrase_candidates_extracted = len(phrase_candidates)
        console.print(
            f"Found {len(phrase_candidates)} multi-language phrases "
            f"(clustered from {before_cluster})."
        )
        _write_json(
            out / "02b_phrase_candidates.json",
            [c.to_dict() for c in phrase_candidates],
        )

    # ---- Stage 1c: payment-rail gazetteer ----
    gazetteer_candidates: list[Candidate] = []
    if cfg.gazetteer.enabled:
        console.rule("[bold]Stage 1c — payment-rail gazetteer")
        gazetteer_candidates = scan_gazetteer(
            messages,
            min_frequency=cfg.gazetteer.min_frequency,
            max_examples_per_term=cfg.gazetteer.max_examples_per_term,
        )
        stats.gazetteer_candidates_extracted = len(gazetteer_candidates)
        console.print(f"Found {len(gazetteer_candidates)} gazetteer hits.")
        _write_json(
            out / "02c_gazetteer_candidates.json",
            [c.to_dict() for c in gazetteer_candidates],
        )

    # ---- Stage 1d: PSP / brand mining ----
    brand_candidates: list[Candidate] = []
    if cfg.brand_mining.enabled and not dry_run:
        console.rule("[bold]Stage 1d — brand mining")
        if llm is None:
            llm = LLMClient(cfg.llm)
        raw_brands = mine_brands(messages)
        console.print(f"Regex pass: {len(raw_brands)} candidate tokens.")
        confirmed = confirm_brands(
            raw_brands, llm,
            batch_size=cfg.brand_mining.confirmation_batch_size,
        )
        # Dedupe case-variants (axepays / Axepays / AxePays -> one).
        brand_candidates = cluster_candidates(confirmed)
        stats.brand_candidates_extracted = len(brand_candidates)
        console.print(
            f"LLM-confirmed brands: {len(brand_candidates)} "
            f"(from {len(confirmed)} raw confirmations, "
            f"{len(confirmed) - len(brand_candidates)} merged as case-variants)."
        )
        _write_json(
            out / "02d_brand_candidates.json",
            [c.to_dict() for c in brand_candidates],
        )

    # Merge all candidate pools then apply ONE final lemma/case clustering
    # pass across the combined set. This collapses cross-pool duplicates
    # the previous loop missed — e.g. ``Casino`` (Stage 1) and ``casino``
    # (Stage 1c) would survive lowercased-key dedup if they came in
    # different orders, and the lemma key handles Russian inflections that
    # crossed pool boundaries. Result: one canonical entry per concept,
    # with all surface variants recorded.
    #
    # Order of precedence (for canonical-term choice when frequencies are
    # equal): Stage 1 > 1c > 1b > 1d. We achieve this by sorting pools
    # before clustering.
    all_candidates: list[Candidate] = list(candidates)
    for pool in (gazetteer_candidates, phrase_candidates, brand_candidates):
        all_candidates.extend(pool)
    candidates = cluster_candidates(all_candidates)

    if max_candidates is not None:
        candidates = candidates[:max_candidates]
        console.print(f"[yellow]Truncated to top {len(candidates)} candidates by frequency.")

    # ---- Stage 2: inline self-definitions ----
    console.rule("[bold]Stage 2 — self-definition mining")
    self_defs = mine_self_definitions(candidates, messages)
    stats.self_definitions_found = sum(len(v) for v in self_defs.values())
    console.print(
        f"Found inline self-definitions for {len(self_defs)} / {len(candidates)} candidates "
        f"({stats.self_definitions_found} total hits)."
    )
    _write_json(
        out / "03_self_definitions.json",
        {term: [d.to_dict() for d in defs] for term, defs in self_defs.items()},
    )

    if dry_run:
        console.print("[yellow]--dry-run: skipping LLM stages.")
        _write_json(out / "00_stats.json", stats.to_dict())
        return stats

    # ---- Stage 3: LLM inference (Tier 1) ----
    console.rule("[bold]Stage 3 — in-corpus LLM inference")
    if llm is None:
        llm = LLMClient(cfg.llm)
    llm.set_stage("stage3_inference")
    drafts: list[DraftEntry] = []

    def _infer(c: Candidate) -> DraftEntry:
        return infer_definition(
            candidate=c,
            self_defs=self_defs.get(c.term),
            domain_brief=cfg.domain_brief,
            llm=llm,
            config=cfg.inference,
        )

    with ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as pool:
        futures = {pool.submit(_infer, c): c for c in candidates}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tier 1"):
            cand = futures[fut]
            try:
                draft = fut.result()
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Tier 1 failed for {cand.term!r}: {exc}")
                continue
            drafts.append(draft)
            stats.drafts_completed += 1

    _write_json(out / "04_drafts.json", [d.to_dict() for d in drafts])

    # ---- Stage 4: research (Tier 3) ----
    research_backend = web_search or NoOpWebSearch()
    if cfg.research.enabled:
        console.rule("[bold]Stage 4 — external research")
        llm.set_stage("stage4_research")
        cand_by_term = {c.term: c for c in candidates}
        low_conf = [d for d in drafts if needs_research(d, cfg.research)]
        console.print(f"{len(low_conf)} drafts below confidence floor {cfg.research.confidence_floor}")
        for draft in tqdm(low_conf, desc="Tier 3"):
            try:
                examples = cand_by_term[draft.term].example_messages
                research_term(draft, examples, llm, research_backend)
                stats.research_invocations += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Research failed for {draft.term!r}: {exc}")
        _write_json(out / "05_drafts_after_research.json", [d.to_dict() for d in drafts])
    else:
        console.print("[dim]Stage 4 (research) is disabled — see ResearchConfig to enable.")

    # ---- Stage 5: adversarial validation + polysemy enrichment ----
    console.rule("[bold]Stage 5 — adversarial validation")
    llm.set_stage("stage5_validation")
    cand_by_term = {c.term: c for c in candidates}
    validated: list[tuple[DraftEntry, Critique]] = []

    def _validate(d: DraftEntry) -> tuple[DraftEntry, Critique]:
        examples = cand_by_term[d.term].example_messages
        return validate_entry(d, examples, cfg.domain_brief, llm, cfg.validator)

    with ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as pool:
        futures = {pool.submit(_validate, d): d for d in drafts}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Validate"):
            d = futures[fut]
            try:
                draft, critique = fut.result()
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Validation failed for {d.term!r}: {exc}")
                continue
            validated.append((draft, critique))
            stats.validated += 1

    # Polysemy enrichment — for any draft where the critique flagged missing
    # senses, draft them in one extra call and append. Done sequentially to
    # keep the per-stage usage stats clean; this set is small (typically <20%
    # of drafts).
    enrichable = [
        (d, c) for d, c in validated
        if c.needs_enrichment and d.relevant
    ]
    if enrichable:
        console.rule(f"[bold]Stage 5b — polysemy enrichment ({len(enrichable)} entries)")
        llm.set_stage("stage5b_enrichment")
        for draft, critique in tqdm(enrichable, desc="Enrich"):
            try:
                examples = cand_by_term[draft.term].example_messages
                enrich_with_missing_senses(draft, critique, examples, cfg.domain_brief, llm)
                stats.enriched += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Enrichment failed for {draft.term!r}: {exc}")

    _write_json(
        out / "06_validated.json",
        [
            {**d.to_dict(), "critique": c.to_dict()}
            for d, c in validated
        ],
    )

    # ---- Stage 5c: structural sense pruning + parent-child dedup ----
    console.rule("[bold]Stage 5c — structural dedup")
    structural_sense_drops = prune_redundant_senses(validated)
    if structural_sense_drops:
        console.print(
            f"Pruned {len(structural_sense_drops)} redundant senses "
            f"(structural: lexical + example overlap)."
        )

    # ---- Stage 5d: semantic sense dedup (LLM-driven) ----
    # Catches the LLM-generated paraphrase senses that *say the same thing*
    # using different words — the structural check above can't see those.
    # Runs only on entries that still have >=2 senses after 5c, so the cost
    # is small (one call per multi-sense entry).
    semantic_sense_drops: list[dict] = []
    multi_sense_count = sum(
        1 for d, _ in validated if d.relevant and len(d.senses) >= 2
    )
    if multi_sense_count > 0:
        console.rule(
            f"[bold]Stage 5d — semantic sense dedup ({multi_sense_count} entries)"
        )
        semantic_sense_drops = dedupe_paraphrase_senses(validated, llm)
        if semantic_sense_drops:
            console.print(
                f"Dropped {len(semantic_sense_drops)} paraphrase senses."
            )

    # ---- Stage 5e: parent-child dedup ----
    validated, parent_drops = dedupe_parent_child(validated)
    if parent_drops:
        console.print(f"Demoted {len(parent_drops)} redundant parent entries.")
        for drop in parent_drops:
            console.print(
                f"  [dim]{drop['dropped']!r} -> covered by "
                f"{drop['in_favour_of']!r} (overlap {drop['example_overlap']:.0%})"
            )

    _write_json(out / "07_dedup_drops.json", {
        "sense_prune_structural": structural_sense_drops,
        "sense_prune_semantic": semantic_sense_drops,
        "parent_child": parent_drops,
    })

    # ---- Persist ----
    console.rule("[bold]Persisting glossary")
    for draft, critique in validated:
        cand = cand_by_term.get(draft.term)
        store.upsert(
            draft,
            critique=critique,
            surface_forms=cand.surface_forms if cand else [],
            frequency=cand.frequency if cand else None,
        )
        if draft.relevant:
            stats.relevant_entries += 1

    store.export_json(out / "glossary.json", relevant_only=True)
    store.export_json(
        out / "glossary_primary_sense_only.json",
        relevant_only=True,
        primary_sense_only=True,
    )
    console.print(
        f"[green]Saved {stats.relevant_entries} relevant entries to {db_path}"
    )
    console.print(f"[green]JSON exports:")
    console.print(f"[green]  {out / 'glossary.json'}   (full, multi-sense)")
    console.print(
        f"[green]  {out / 'glossary_primary_sense_only.json'}   "
        f"(single-sense per entry — recommended for downstream agents)"
    )

    if llm is not None:
        stats.llm_calls = llm.usage.calls
        stats.llm_input_tokens = llm.usage.input_tokens
        stats.llm_output_tokens = llm.usage.output_tokens
        stats.estimated_cost_usd = llm.usage.estimated_cost_usd
        stats.cost_by_stage = llm.usage.to_dict()["by_stage"]

    _write_json(out / "00_stats.json", stats.to_dict())
    return stats
