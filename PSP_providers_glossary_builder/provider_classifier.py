"""PSP Provider classifier — single-message interface.

Public API:
    classify_provider_message()   — classify one message, returns dict
    load_provider_glossary()      — load glossary from SQLite DB

Pipeline per message:
    1. Lexical pre-filter    (intent_filter_psp_providers.pre_filter)
    2. Stage 1 micro-LLM    (YES/NO: is this a seller?)
    3. Full LLM              (structured classification + extraction)
    → ProviderDecision

No judge stage yet — deferred to a later iteration.

Usage::

    from PSP_providers_glossary_builder.provider_classifier import (
        classify_provider_message,
        load_provider_glossary,
    )
    from glossary_builder.llm import LLMClient

    glossary = load_provider_glossary("PSP_providers_glossary_builder/data/provider_glossary.db")
    llm = LLMClient()

    result = classify_provider_message(
        text="Работаю по EU картам, апрув 72%, пишите в лс",
        glossary=glossary,
        llm=llm,
    )
    # result is a flat dict — write to DB, compare with test dataset, etc.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from glossary_builder.lead_extraction import GlossaryIndex
from glossary_builder.llm import LLMClient
from glossary_builder.loader import Message

from .provider_decision import (
    ProviderDecision,
    ProviderExtractionConfig,
    _build_provider_decision,
)
from .provider_prompt import (
    PROVIDER_DEFINITION,
    _PROVIDER_STAGE1_SYSTEM,
    _PROVIDER_STAGE1_USER_TEMPLATE,
    _PROVIDER_SYSTEM_TEMPLATE,
    _PROVIDER_USER_TEMPLATE,
    _PROVIDER_JUDGE_SYSTEM,
    _PROVIDER_JUDGE_USER_TEMPLATE,
)

import logging
logger = logging.getLogger(__name__)


def make_provider_llm_client() -> LLMClient:
    """LLMClient configured for PSP provider pipeline via OpenRouter.
    Uses PSP_PROVIDERS_OPENROUTER_API_KEY — separate from main pipeline key.
    """
    import os
    from glossary_builder.config import LLMConfig

    api_key = os.environ.get("PSP_PROVIDERS_OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("PSP_PROVIDERS_OPENROUTER_API_KEY not set in .env")

    return LLMClient(LLMConfig(
        api_key   = api_key,
        base_url  = "https://openrouter.ai/api/v1",
        model     = os.environ.get("PSP_PROVIDERS_MODEL", "openai/gpt-4.1-nano"),
       # temperature = 0
    ))

# ── Glossary loader ───────────────────────────────────────────────────────────

def load_provider_glossary(
    db_path: str | Path,
) -> list[dict]:
    """Load the provider glossary from SQLite.

    Prefers the enriched table ``provider_glossary_enriched`` (built by
    run_provider_inference.py) which contains real LLM-generated definitions,
    context clues, and relevance flags.

    Falls back to raw ``intent_glossary`` (Stage 1b only) when the enriched
    table is absent — useful during development before inference has run.

    Returns a list of dicts compatible with GlossaryIndex:
        {"term": str, "surface_forms": list[str], "senses": [{"definition": str}]}
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Provider glossary not found: {path}\n"
            f"Run run_provider_glossary.py first."
        )

    conn = sqlite3.connect(path)
    entries: list[dict] = []

    try:
        # ── Try enriched table first ──────────────────────────────────────────
        enriched_exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='provider_glossary_enriched'"
        ).fetchone()

        if enriched_exists:
            rows = conn.execute(
                """
                SELECT term, surface_forms, senses, frequency
                FROM provider_glossary_enriched
                WHERE relevant = 1
                ORDER BY frequency DESC
                """
            ).fetchall()

            for term, surface_json, senses_json, freq in rows:
                if not term:
                    continue
                try:
                    surface_forms: list[str] = json.loads(surface_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    surface_forms = []
                try:
                    senses: list[dict] = json.loads(senses_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    senses = []

                if not senses:
                    senses = [{"definition": f"payment processing term (freq: {freq})"}]

                if term.lower() not in {s.lower() for s in surface_forms}:
                    surface_forms = [term] + surface_forms

                entries.append({
                    "term":         term,
                    "surface_forms": surface_forms,
                    "senses":       senses,
                })

            logger.info(
                "[provider_classifier] loaded %d enriched glossary terms from %s",
                len(entries), path,
            )
            return entries

        # ── Fallback: raw intent_glossary (Stage 1b only) ─────────────────────
        logger.warning(
            "[provider_classifier] provider_glossary_enriched not found — "
            "falling back to intent_glossary (run run_provider_inference.py "
            "to get full definitions)"
        )
        rows = conn.execute(
            "SELECT term, surface_forms, frequency FROM intent_glossary"
        ).fetchall()

        for term, surface_json, freq in rows:
            if not term:
                continue
            try:
                surface_forms = json.loads(surface_json or "[]")
            except (json.JSONDecodeError, TypeError):
                surface_forms = []
            if term.lower() not in {s.lower() for s in surface_forms}:
                surface_forms = [term] + surface_forms
            entries.append({
                "term":         term,
                "surface_forms": surface_forms,
                "senses": [{"definition": f"payment processing industry term (corpus freq: {freq})"}],
            })

        logger.info(
            "[provider_classifier] loaded %d raw glossary terms (no definitions)",
            len(entries),
        )

    finally:
        conn.close()

    return entries


# ── Glossary block formatter ──────────────────────────────────────────────────

def _format_glossary_block(entries: list[dict]) -> str:
    if not entries:
        return "  (no glossary terms detected in the target message)"
    lines = []
    for e in entries:
        defn = e["senses"][0]["definition"] if e.get("senses") else ""
        lines.append(f"  - {e['term']}: {defn}")
    return "\n".join(lines)


def _format_context(context: list[Message]) -> str:
    if not context:
        return "  (no context)"
    lines = []
    for m in context:
        text = (m.text or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:197] + "..."
        ts    = m.date.isoformat() if m.date else "?"
        uname = m.username or "anon"
        mid   = m.message_id if m.message_id is not None else "?"
        lines.append(f"  [msg_id={mid} @{uname} {ts}] {text}")
    return "\n".join(lines)


# ── Few-shot formatter for provider RAG ──────────────────────────────────────

def format_provider_few_shot(examples: dict[str, list[str]]) -> str:
    """Format retrieved RAG examples for injection into the provider prompt."""
    if not examples.get("leads") and not examples.get("not_leads"):
        return ""

    lines = ["SIMILAR EXAMPLES FROM LABELED DATA:"]

    if examples.get("not_leads"):
        lines.append("\nNOT PROVIDER:")
        for i, text in enumerate(examples["not_leads"], 1):
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            lines.append(f"  {i}. {snippet}")

    if examples.get("leads"):
        lines.append("\nPROVIDER:")
        for i, text in enumerate(examples["leads"], 1):
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            lines.append(f"  {i}. {snippet}")

    lines.append("")
    return "\n".join(lines)


# ── Judge helpers ─────────────────────────────────────────────────────────────

def _run_provider_judge(
    decision: ProviderDecision,
    llm: LLMClient,
) -> tuple[str, str]:
    """Call the independent judge LLM on a is_provider=True decision.

    Returns (verdict, reason).
    verdict is one of: 'REAL_PROVIDER', 'MISTAKE', 'REVIEW' (fallback).
    """
    user = _PROVIDER_JUDGE_USER_TEMPLATE.format(
        message_id=decision.message_id or "?",
        username=decision.username or "anon",
        timestamp=decision.timestamp or "?",
        text=(decision.text or "").replace("\n", " ").strip()[:500],
        confidence=f"{decision.confidence:.2f}",
        geo=", ".join(decision.geo) or "—",
        methods=", ".join(decision.methods) or "—",
        vertical=", ".join(decision.vertical) or "—",
        company=decision.company or "—",
        position=getattr(decision, "position", None) or "—",
        evidence_quote=(decision.evidence_quote or "")[:200],
        rationale=(decision.rationale or "")[:200],
    )
    try:
        llm.set_stage("provider_judge")
        data, _ = llm.complete_json(_PROVIDER_JUDGE_SYSTEM, user, max_tokens=200)
        verdict = str(data.get("verdict", "")).strip().upper()
        reason  = str(data.get("reason", "")).strip()
        if verdict not in ("REAL_PROVIDER", "MISTAKE"):
            verdict = "REAL_PROVIDER"
    except Exception as exc:
        verdict = "REVIEW"
        reason  = f"judge call failed: {exc}"
    return verdict, reason


def _merge_provider(
    decision: ProviderDecision,
    verdict: str | None,
    judge_reason: str | None,
) -> dict:
    """Merge extract + judge fields into one flat dict."""
    row = decision.to_row()
    row["verdict"]      = verdict
    row["judge_reason"] = judge_reason
    return row


# ── ProviderExtractor ─────────────────────────────────────────────────────────

class ProviderExtractor:
    """Per-message PSP provider classifier.

    Mirrors LeadExtractor from lead_extraction.py but for the seller side.
    Designed for single-message webhook use — no batching, no JSONL streaming.
    """

    def __init__(
        self,
        glossary: list[dict],
        llm: LLMClient,
        cfg: ProviderExtractionConfig | None = None,
        rag_index=None,
    ):
        self.glossary_index = GlossaryIndex(glossary)
        self.llm = llm
        self.cfg = cfg or ProviderExtractionConfig()
        self.rag_index = rag_index

        # Inject provider definition if not set externally
        if not self.cfg.provider_definition:
            self.cfg.provider_definition = PROVIDER_DEFINITION

        # Counters
        self.prefiltered_count  = 0
        self.stage1_filtered_count = 0
        self._lock = threading.Lock()

        # Wire up lexical pre-filter
        self._pre_filter = None
        if self.cfg.pre_filter_enabled:
            try:
                from intent_filter_psp_providers import pre_filter, load_glossary_terms
                terms = load_glossary_terms(self.cfg.pre_filter_db_path)
                if not terms:
                    raise ValueError("Provider glossary terms are empty")
                self._pre_filter = pre_filter
                logger.info(
                    "[provider_classifier] pre-filter loaded: %d terms", len(terms)
                )
            except Exception as exc:
                logger.warning(
                    "[provider_classifier] pre-filter disabled: %s — "
                    "all messages will go to LLM", exc
                )

    # ── Core classification ───────────────────────────────────────────────────

    def extract_one(
        self,
        target:  Message,
        context: list[Message] | None = None,
    ) -> ProviderDecision | None:
        """Classify one message. Returns None if skipped (too short / excluded)."""

        cfg  = self.cfg
        text = (target.text or "").strip()

        # Guard: too short
        if not text or len(text) < cfg.skip_min_text_length:
            return None

        # Guard: excluded authors
        if target.username and target.username in cfg.excluded_authors:
            return None

        context = context or []

        # Glossary lookup (domain-agnostic substring match)
        glossary_hits = self.glossary_index.find_in(
            text, limit=cfg.max_glossary_entries_per_call
        )

        # ── Step 1: Lexical pre-filter ────────────────────────────────────────
        if self._pre_filter is not None:
            passed = self._pre_filter(text, cfg.pre_filter_db_path)
            if not passed:
                with self._lock:
                    self.prefiltered_count += 1
                return ProviderDecision(
                    message_id           = target.message_id,
                    timestamp            = target.date.isoformat() if target.date else None,
                    username             = target.username,
                    text                 = text,
                    is_provider          = False,
                    confidence           = 1.0,
                    geo                  = [],
                    methods              = [],
                    vertical             = [],
                    company              = None,
                    evidence_quote       = "",
                    rationale            = "[pre-filtered: no domain term or seller signal]",
                    glossary_terms_seen  = [e["term"] for e in glossary_hits],
                    elapsed_ms           = 0.0,
                )

        # ── Step 2: Stage 1 micro-LLM (YES/NO) ───────────────────────────────
        if cfg.stage1_enabled:
            try:
                self.llm.set_stage("provider_stage1_micro")
                s1 = self.llm.complete(
                    _PROVIDER_STAGE1_SYSTEM,
                    _PROVIDER_STAGE1_USER_TEMPLATE.format(text=text),
                    max_tokens=10,
                    temperature=0.0,
                )
                is_yes = "yes" in s1.text.strip().lower()
            except Exception as exc:
                logger.warning("[provider_classifier] stage1 failed (%s) — fail open", exc)
                is_yes = True  # fail open: defer to full LLM

            if not is_yes:
                with self._lock:
                    self.stage1_filtered_count += 1
                return ProviderDecision(
                    message_id           = target.message_id,
                    timestamp            = target.date.isoformat() if target.date else None,
                    username             = target.username,
                    text                 = text,
                    is_provider          = False,
                    confidence           = 1.0,
                    geo                  = [],
                    methods              = [],
                    vertical             = [],
                    company              = None,
                    evidence_quote       = "",
                    rationale            = "[stage1-filtered: not a seller-intent message]",
                    glossary_terms_seen  = [e["term"] for e in glossary_hits],
                    elapsed_ms           = 0.0,
                )

        # ── Step 3: Full LLM classification ───────────────────────────────────
        system = _PROVIDER_SYSTEM_TEMPLATE.replace(
            "{provider_definition}", cfg.provider_definition
        )

        # RAG few-shot injection
        few_shot_block = ""
        if self.rag_index is not None:
            try:
                examples = self.rag_index.query(text)
                few_shot_block = format_provider_few_shot(examples)
            except Exception as exc:
                logger.warning("[provider_classifier] RAG query failed: %s", exc)

        user = _PROVIDER_USER_TEMPLATE.format(
            glossary_block = _format_glossary_block(glossary_hits),
            few_shot_block = few_shot_block,
            context_block  = _format_context(context),
            message_id     = target.message_id or "?",
            username       = target.username or "anon",
            timestamp      = target.date.isoformat() if target.date else "?",
            text           = text,
        )

        t0 = time.perf_counter()
        try:
            self.llm.set_stage("provider_full_prompt")
            data, raw = self.llm.complete_json(system, user, max_tokens=500)
        except Exception as exc:
            return ProviderDecision(
                message_id           = target.message_id,
                timestamp            = target.date.isoformat() if target.date else None,
                username             = target.username,
                text                 = text,
                is_provider          = False,
                confidence           = 0.0,
                geo                  = [],
                methods              = [],
                vertical             = [],
                company              = None,
                evidence_quote       = "",
                rationale            = f"[extraction failed: {exc}]",
                glossary_terms_seen  = [e["term"] for e in glossary_hits],
                elapsed_ms           = (time.perf_counter() - t0) * 1000.0,
            )

        decision = _build_provider_decision(target, text, data, glossary_hits, raw, cfg=cfg)
        decision.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return decision


# ── Public API ────────────────────────────────────────────────────────────────

def classify_provider_message(
    text:       str,
    glossary:   list[dict],
    llm:        LLMClient,
    cfg:        Optional[ProviderExtractionConfig] = None,
    rag_index=None,
    *,
    message_id: Optional[int]      = None,
    username:   Optional[str]      = None,
    timestamp:  Optional[datetime] = None,
    context:    Optional[list[Message]] = None,
) -> dict:
    """Classify one message through the PSP provider pipeline.

    Args:
        text:       Raw message text.
        glossary:   Provider glossary (output of load_provider_glossary()).
        llm:        Shared LLMClient (caller owns it for token accounting).
        cfg:        Optional config. Defaults to ProviderExtractionConfig().
        message_id: Optional upstream identifier (Telegram message id, DB row, etc.)
        username:   Optional sender handle.
        timestamp:  Optional message datetime.
        context:    Optional list of surrounding Message objects.

    Returns:
        Flat JSON-serialisable dict (ProviderDecision.to_row() + is_provider bool).
        Always returns a dict — never raises on LLM failure (fails gracefully).
    """
    cfg = cfg or ProviderExtractionConfig()

    message = Message(
        message_id = message_id,
        date       = timestamp,
        text       = text,
        username   = username,
        group_id   = None,
        user_id    = None,
        first_name = None,
        last_name  = None,
    )

    extractor = ProviderExtractor(glossary, llm, cfg, rag_index=rag_index)

    decision: Optional[ProviderDecision] = extractor.extract_one(
        message,
        context=context or [],
    )

    if decision is None:
        # Skipped (too short / excluded author)
        return _merge_provider(
            ProviderDecision(
                message_id           = message_id,
                timestamp            = timestamp.isoformat() if timestamp else None,
                username             = username,
                text                 = text,
                is_provider          = False,
                confidence           = 1.0,
                geo                  = [],
                methods              = [],
                vertical             = [],
                company              = None,
                evidence_quote       = "",
                rationale            = "[skipped: message too short or author excluded]",
                glossary_terms_seen  = [],
                elapsed_ms           = 0.0,
            ),
            verdict      = None,
            judge_reason = None,
        )

    # Not a provider — skip judge entirely
    if not decision.is_provider:
        return _merge_provider(decision, verdict=None, judge_reason=None)

    # ── Judge stage: independent second-pass validation ───────────────────────
    if cfg.judge_enabled:
        # Optionally use a stronger model for the judge
        judge_model = os.environ.get("PROVIDER_JUDGE_MODEL")
        if judge_model:
            llm.model = judge_model

        verdict, judge_reason = _run_provider_judge(decision, llm)
        decision.verdict      = verdict
        decision.judge_reason = judge_reason
    else:
        verdict      = None
        judge_reason = None

    return _merge_provider(decision, verdict=verdict, judge_reason=judge_reason)