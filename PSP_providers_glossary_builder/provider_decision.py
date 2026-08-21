"""Provider decision dataclasses and JSON-to-dataclass builder.

ProviderDecision   — structured result for one message classification.
ProviderExtractionConfig — config for the provider extractor.
_build_provider_decision() — parses raw LLM JSON dict into ProviderDecision.

Seller-side mirror of LeadDecision / _build_decision() in lead_extraction.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── ProviderDecision ──────────────────────────────────────────────────────────

@dataclass
class ProviderDecision:
    """Structured result of classifying one message as a PSP provider or not."""

    # ── Message identity ──────────────────────────────────────────────────────
    message_id: int | None
    timestamp:  str | None
    username:   str | None
    text:       str

    # ── Classification ────────────────────────────────────────────────────────
    is_provider: bool
    confidence:  float          # 0.0 – 1.0

    # ── Extracted provider attributes (populated when is_provider=True) ───────
    geo:      list[str]         # GEOs the provider covers ('EU', 'UK', 'LATAM')
    methods:  list[str]         # payment methods they support (Visa, USDT, etc.)
    vertical: list[str]         # verticals they serve (igaming, forex, adult…)
    company:  str | None        # company / brand name if stated


    # ── Evidence ──────────────────────────────────────────────────────────────
    evidence_quote:     str     # verbatim quote from the message
    rationale:          str     # one-two sentence English explanation
    glossary_terms_seen: list[str]  # which glossary terms appeared in message

    # ── Judge stage ───────────────────────────────────────────────────────────
    verdict:      str | None = None
    """REAL_PROVIDER | MISTAKE | REVIEW — set after judge call, None if skipped."""

    judge_reason: str | None = None
    """One-sentence English explanation from the judge LLM."""

    # ── Diagnostics ───────────────────────────────────────────────────────────
    elapsed_ms:       float | None = None
    raw_llm_response: str   | None = None

    position: str | None = None  # job title / role of the author if stated

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_llm_response", None)
        return d

    def to_row(self) -> dict:
        """Flat dict suitable for a pandas DataFrame row."""
        return {
            "message_id":          self.message_id,
            "timestamp":           self.timestamp,
            "username":            self.username,
            "is_provider":         self.is_provider,
            "confidence":          self.confidence,
            "verdict":             self.verdict,
            "judge_reason":        self.judge_reason,
            "geo":                 ", ".join(self.geo),
            "methods":             ", ".join(self.methods),
            "vertical":            ", ".join(self.vertical),
            "company":             self.company or "",
            "position":            self.position or "",
            "evidence_quote":      self.evidence_quote,
            "rationale":           self.rationale,
            "glossary_terms_seen": ", ".join(self.glossary_terms_seen),
            "elapsed_ms":          self.elapsed_ms,
            "text":                self.text,
        }

# ── ProviderExtractionConfig ──────────────────────────────────────────────────

@dataclass
class ProviderExtractionConfig:
    """Configuration for the provider extractor."""

    # Injected into the full LLM system prompt as the classification criteria.
    # Defaults to PROVIDER_DEFINITION from provider_prompt.py.
    # Override to experiment with different definitions without changing code.
    provider_definition: str = ""
    """Set at runtime from provider_prompt.PROVIDER_DEFINITION."""

    # ── Lexical pre-filter ────────────────────────────────────────────────────
    pre_filter_enabled: bool = True
    """Run intent_filter_psp_providers.pre_filter() before LLM calls.
    If False, every message goes to the LLM (higher cost, higher recall)."""

    pre_filter_db_path: str = (
        "PSP_providers_glossary_builder/data/provider_glossary.db"
    )
    """SQLite DB holding the intent_glossary table built by run_provider_glossary.py."""

    # ── Stage 1 micro-filter ──────────────────────────────────────────────────
    stage1_enabled: bool = True
    """Run the cheap YES/NO _PROVIDER_STAGE1_SYSTEM call before the full prompt.
    A NO short-circuits to is_provider=False with no expensive structured call."""

    # ── Message handling ──────────────────────────────────────────────────────
    skip_min_text_length: int = 20
    """Messages shorter than this are skipped (greetings, emoji-only, etc.)."""

    max_glossary_entries_per_call: int = 12
    """Cap on glossary entries injected per LLM call."""

    context_messages_before: int = 3
    """Preceding messages from same group to include as context.
    Smaller than the buyer extractor — providers usually pitch in standalone messages."""

    context_messages_after: int = 0
    """Following messages to include as context."""

    validate_evidence_quote: bool = False
    """Downgrade is_provider=True if the evidence quote is not found in the message."""

    judge_enabled: bool = True
    """Run the independent judge LLM after is_provider=True decisions.
    Set False to skip the judge stage (faster, cheaper, lower precision)."""

    # ── Excluded authors ──────────────────────────────────────────────────────
    excluded_authors: set[str] = field(default_factory=lambda: {
        "ChatLogixBot",
    })


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_provider_decision(
    target,                     # Message (from glossary_builder.loader)
    text:          str,
    data:          dict,
    glossary_hits: list[dict],
    raw:           str,
    cfg:           ProviderExtractionConfig | None = None,
) -> ProviderDecision:
    """Parse raw LLM JSON dict into a ProviderDecision.

    Mirrors _build_decision() from lead_extraction.py.
    Handles malformed / partial LLM output gracefully.
    """

    def _list_str(v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def _bound_conf(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    is_provider    = bool(data.get("is_provider", False))
    evidence_quote = str(data.get("evidence_quote", "")).strip()
    rationale      = str(data.get("rationale", "")).strip()

    company_raw = data.get("company")
    company: str | None = (
        str(company_raw).strip() if isinstance(company_raw, str) and company_raw.strip()
        else None
    )

    position_raw = data.get("position")
    position: str | None = (
        str(position_raw).strip() if isinstance(position_raw, str) and position_raw.strip()
        else None
    )

    # Code-level safeguard: if the LLM quoted something not in the message,
    # downgrade to is_provider=False (same logic as lead_extraction.py).
    if is_provider and evidence_quote and cfg and cfg.validate_evidence_quote:
        if not _quote_supported_by_text(evidence_quote, text):
            is_provider = False
            rationale = (
                "[downgraded by evidence-quote validator: quoted evidence "
                "not found in target message] " + rationale
            )

    return ProviderDecision(
        message_id          = target.message_id,
        timestamp           = target.date.isoformat() if target.date else None,
        username            = target.username,
        text                = text,
        is_provider         = is_provider,
        confidence          = _bound_conf(data.get("confidence", 0.0)),
        geo                 = _list_str(data.get("geo")),
        methods             = _list_str(data.get("methods")),
        vertical            = _list_str(data.get("vertical")),
        company             = company if is_provider else None,
        position            = position if is_provider else None,
        evidence_quote      = evidence_quote,
        rationale           = rationale,
        glossary_terms_seen = [e["term"] for e in glossary_hits],
        raw_llm_response    = raw,
    )

# ── Evidence-quote validator (copied from lead_extraction.py) ─────────────────

def _quote_supported_by_text(
    quote: str,
    text: str,
    *,
    min_overlap_chars: int = 25,
) -> bool:
    """Return True iff the quoted evidence is meaningfully present in text."""
    if not quote or not text:
        return False
    nt = _norm_for_quote_match(text)
    nq = _norm_for_quote_match(quote)
    if not nq or not nt:
        return False
    if nq in nt:
        return True
    if len(nq) < min_overlap_chars:
        return False
    for i in range(0, len(nq) - min_overlap_chars + 1):
        chunk = nq[i:i + min_overlap_chars]
        if chunk in nt:
            return True
    return False


def _norm_for_quote_match(s: str) -> str:
    """Lowercase + collapse non-alphanumeric runs for quote-substring checks."""
    return re.sub(r"[^a-zA-Z0-9\u0400-\u04FF]+", " ", s.lower()).strip()