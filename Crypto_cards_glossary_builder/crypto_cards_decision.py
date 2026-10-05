"""CryptoCardsDecision dataclass and CryptoCardsExtractionConfig.

CryptoCardsDecision       — holds every field produced by classify_crypto_cards_message()
CryptoCardsExtractionConfig — pipeline knobs (what stages to run, model names, paths)
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
class CryptoCardsDecision:
    """Full result of one classify_crypto_cards_message() call."""

    # ── Identity ──────────────────────────────────────────────────────────────
    message_id:  Optional[int]      = None
    timestamp:   Optional[datetime] = None
    username:    Optional[str]      = None
    text:        Optional[str]      = None

    # ── Stage 3 extractor output ──────────────────────────────────────────────
    is_lead:        bool          = False
    lead_type:      Optional[str] = None   # "card_seeker" | "card_provider"
    confidence:     Optional[str] = None   # "high" | "medium" | "low"
    #   confidence meaning depends on lead_type:
    #     card_seeker  → warmth   (how hot is the lead)
    #     card_provider → overlap (how directly do we compete)

    evidence_quote: Optional[str] = None   # verbatim quote from the message
    rationale:      Optional[str] = None   # Stage 3 reasoning

    # ── Stage 4 judge output ──────────────────────────────────────────────────
    verdict: Optional[str] = None
    #   "REAL_CARD_SEEKER" | "REAL_CARD_PROVIDER" | "MISTAKE"

    notes: Optional[str] = None
    #   Stage 4 explanation for the BD team; written to client_ready_leads.notes

    # ── Diagnostics ───────────────────────────────────────────────────────────
    glossary_terms_seen: list[str] = field(default_factory=list)
    elapsed_ms:          Optional[float] = None

    # ── Pipeline tag (for DB) ─────────────────────────────────────────────────
    pipeline: str = "crypto_cards"

    def to_dict(self) -> dict:
        return {
            "message_id":          self.message_id,
            "timestamp":           self.timestamp.isoformat() if self.timestamp else None,
            "username":            self.username,
            "text":                self.text,
            "is_lead":             self.is_lead,
            "lead_type":           self.lead_type,
            "confidence":          self.confidence,
            "evidence_quote":      self.evidence_quote,
            "rationale":           self.rationale,
            "verdict":             self.verdict,
            "notes":               self.notes,
            "glossary_terms_seen": self.glossary_terms_seen,
            "elapsed_ms":          self.elapsed_ms,
            "pipeline":            self.pipeline,
        }


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CryptoCardsExtractionConfig:
    """Knobs for the Crypto Cards classification pipeline."""

    # ── Pre-filter ────────────────────────────────────────────────────────────
    pre_filter_enabled: bool = True
    """If False, skip the lexical pre-filter (useful for testing without a glossary)."""

    pre_filter_db_path: str = str(
        _PROJECT_ROOT / "Crypto_cards_classifier" / "data" / "crypto_cards_glossary.db"
    )
    """Path to the SQLite DB produced by run_crypto_cards_glossary.py + run_crypto_cards_inference.py."""

    # ── Stage 1 (micro-LLM YES/NO) ───────────────────────────────────────────
    stage1_enabled: bool = True
    """If False, every message that passes the pre-filter goes directly to Stage 3."""

    # ── Judge (Stage 4) ───────────────────────────────────────────────────────
    judge_enabled: bool = True
    """If False, Stage 3 extractor result is used directly without judge verification."""

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
    # CRYPTO_CARDS_STAGE1_MODEL — model for Stage 1 YES/NO triage
    # CRYPTO_CARDS_JUDGE_MODEL  — model for Stage 4 judge
    # Set in .env on the droplet; fall back to None (classifier uses its default).

    def stage1_model(self) -> Optional[str]:
        return os.environ.get("CRYPTO_CARDS_STAGE1_MODEL")

    def judge_model(self) -> Optional[str]:
        return os.environ.get("CRYPTO_CARDS_JUDGE_MODEL")
