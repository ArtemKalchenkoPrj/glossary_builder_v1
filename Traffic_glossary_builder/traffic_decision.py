"""TrafficDecision dataclass and TrafficExtractionConfig.

TrafficDecision    — holds every field produced by classify_traffic_message()
TrafficExtractionConfig — pipeline knobs (what stages to run, model names, paths)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrafficDecision:
    """Full result of one classify_traffic_message() call."""

    # ── Identity ──────────────────────────────────────────────────────────────
    message_id:  Optional[int]      = None
    timestamp:   Optional[datetime] = None
    username:    Optional[str]      = None
    text:        Optional[str]      = None

    # ── Extractor output ──────────────────────────────────────────────────────
    is_lead:     bool               = False
    lead_type:   Optional[str]      = None   # "traffic_provider" | "seeking_traffic"
    confidence:  Optional[str]      = None   # "high" | "medium" | "low"

    # ── Extracted fields ──────────────────────────────────────────────────────
    vertical:        list[str] = field(default_factory=list)   # niche
    traffic_source:  list[str] = field(default_factory=list)
    geo:             list[str] = field(default_factory=list)
    pricing_model:   list[str] = field(default_factory=list)

    evidence_quote:  Optional[str] = None
    rationale:       Optional[str] = None

    # ── Glossary / diagnostics ────────────────────────────────────────────────
    glossary_terms_seen: list[str] = field(default_factory=list)
    elapsed_ms:          Optional[float] = None

    # ── Judge output ──────────────────────────────────────────────────────────
    verdict:      Optional[str] = None   # "TRAFFIC_PROVIDER" | "TRAFFIC_LEAD" | "MISTAKE"
    judge_reason: Optional[str] = None

    # ── Certainty gate output ─────────────────────────────────────────────────
    certainty:               Optional[str] = None   # "CERTAIN" | "UNCERTAIN" | None (gate off / not a lead)
    gate_uncertainty_reason: Optional[str] = None   # sms | infra | role_unclear | gate_api_error | ...
    gate_reason:             Optional[str] = None   # free-text explanation from the gate model

    # ── Pipeline tag (for DB) ─────────────────────────────────────────────────
    pipeline: str = "traffic"

    def to_dict(self) -> dict:
        return {
            "message_id":         self.message_id,
            "timestamp":          self.timestamp.isoformat() if self.timestamp else None,
            "username":           self.username,
            "text":               self.text,
            "is_lead":            self.is_lead,
            "lead_type":          self.lead_type,
            "confidence":         self.confidence,
            "vertical":           self.vertical,
            "traffic_source":     self.traffic_source,
            "geo":                self.geo,
            "pricing_model":      self.pricing_model,
            "evidence_quote":     self.evidence_quote,
            "rationale":          self.rationale,
            "glossary_terms_seen": self.glossary_terms_seen,
            "elapsed_ms":         self.elapsed_ms,
            "verdict":            self.verdict,
            "judge_reason":       self.judge_reason,
            "certainty":          self.certainty,
            "gate_uncertainty_reason": self.gate_uncertainty_reason,
            "gate_reason":        self.gate_reason,
            "pipeline":           self.pipeline,
        }


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrafficExtractionConfig:
    """Knobs for the Traffic classification pipeline."""

    # ── Pre-filter ────────────────────────────────────────────────────────────
    pre_filter_enabled: bool = True
    """If False, skip the lexical pre-filter (useful for testing without a glossary)."""

    pre_filter_db_path: str = str(
        _PROJECT_ROOT / "Traffic_glossary_builder" / "data" / "traffic_glossary.db"
    )
    """Path to the SQLite DB produced by run_traffic_glossary.py + run_traffic_inference.py."""

    # ── Stage 1 (micro-LLM YES/NO) ───────────────────────────────────────────
    stage1_enabled: bool = True
    """If False, every message that passes the pre-filter goes directly to full LLM."""

    # ── Judge ─────────────────────────────────────────────────────────────────
    judge_enabled: bool = True
    """If False, extractor result is used directly without judge verification."""

    # ── Minimum text length ───────────────────────────────────────────────────
    skip_min_text_length: int = 20
    """Messages shorter than this are skipped entirely."""

    # ── Excluded authors ──────────────────────────────────────────────────────
    excluded_authors: set[str] = field(default_factory=lambda: {
        "ChatLogixBot",
        "Summarizer_aibot",
    })
    """Bot usernames that should never be classified as leads."""

    # ── Model overrides (read from ENV at runtime) ────────────────────────────
    # TRAFFIC_STAGE1_MODEL — model for Stage 1 YES/NO  (default: same as main)
    # TRAFFIC_JUDGE_MODEL  — model for judge            (default: same as main)
    # These are read directly in traffic_classifier.py via os.environ.get()

    def stage1_model(self) -> Optional[str]:
        return os.environ.get("TRAFFIC_STAGE1_MODEL")

    def judge_model(self) -> Optional[str]:
        return os.environ.get("TRAFFIC_JUDGE_MODEL")

    # ── Certainty gate (Anthropic models, read from ENV at runtime) ───────────
    # TRAFFIC_GATE_ENABLED        — "1"/"true" turns the gate on (default: off)
    # TRAFFIC_GATE_MODEL          — checks confirmed leads   (default: claude-haiku-5-5)
    # TRAFFIC_GATE_RESOLVER_MODEL — resolves extractor/judge mismatches (default: claude-sonnet-5-5)

    def gate_enabled(self) -> bool:
        return os.environ.get("TRAFFIC_GATE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

    def gate_model(self) -> str:
        return os.environ.get("TRAFFIC_GATE_MODEL") or "claude-haiku-5-5"

    def gate_resolver_model(self) -> str:
        return os.environ.get("TRAFFIC_GATE_RESOLVER_MODEL") or "claude-sonnet-5-5"
