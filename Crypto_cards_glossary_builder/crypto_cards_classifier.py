"""Crypto Cards vertical classifier.

Synchronous pipeline — designed to be called via asyncio.to_thread() from
the webhook, or directly from a Jupyter notebook for testing.

Public API:
    classify_crypto_cards_message(text, glossary, llm, cfg, ...) -> dict
    load_crypto_cards_glossary(db_path) -> list[dict]
    make_crypto_cards_llm_client() -> LLMClient

Pipeline stages:
    0.  Length check (< skip_min_text_length → skip)
    1.  Excluded-author check
    2.  Glossary lookup (finds matching terms)
    3.  Pre-filter (intent_filter_crypto_cards.pre_filter)  [if enabled]
    4.  Stage 1 micro-LLM YES/NO                            [if enabled]
    5.  Stage 3 role-play LLM → lead_type + confidence + evidence_quote + rationale
    6.  Stage 4 judge LLM → REAL_CARD_SEEKER | REAL_CARD_PROVIDER | MISTAKE [if enabled]
    7.  Verdict alignment check (Stage 3 lead_type must match Stage 4 verdict)

Result is always a dict (same shape as CryptoCardsDecision.to_dict()) so the
caller doesn't have to know about the dataclass.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from glossary_builder.llm import LLMClient

from .crypto_cards_decision import CryptoCardsDecision, CryptoCardsExtractionConfig
from .crypto_cards_prompt import (
    _CRYPTO_CARDS_STAGE1_SYSTEM,
    _CRYPTO_CARDS_STAGE3_SYSTEM,
    _CRYPTO_CARDS_STAGE3_USER,
    _CRYPTO_CARDS_STAGE4_SYSTEM,
    _CRYPTO_CARDS_STAGE4_USER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter endpoint
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Glossary loader
# ---------------------------------------------------------------------------

def load_crypto_cards_glossary(db_path: str) -> list[dict]:
    """Load terms from crypto_cards_glossary_enriched (preferred) or intent_glossary.

    Returns a list of dicts with keys: term, surface_forms, relevant, confidence.
    Returns [] if the DB doesn't exist yet (pre-filter will be skipped gracefully).
    """
    path = Path(db_path)
    if not path.exists():
        logger.warning(
            "crypto_cards_glossary: DB not found at %s — glossary will be empty", db_path
        )
        return []

    conn = sqlite3.connect(db_path)
    glossary: list[dict] = []
    try:
        # ── Prefer enriched table ─────────────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT term, surface_forms, relevant, confidence "
                "FROM crypto_cards_glossary_enriched WHERE relevant = 1"
            ).fetchall()
            for term, surface_json, relevant, confidence in rows:
                glossary.append({
                    "term": term,
                    "surface_forms": json.loads(surface_json or "[]"),
                    "relevant": bool(relevant),
                    "confidence": confidence or 0.0,
                })
            if glossary:
                logger.info(
                    "crypto_cards_glossary: loaded %d terms from crypto_cards_glossary_enriched",
                    len(glossary),
                )
                return glossary
        except sqlite3.OperationalError:
            pass  # enriched table doesn't exist yet

        # ── Fallback: raw intent_glossary ─────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT term, surface_forms FROM intent_glossary"
            ).fetchall()
            for term, surface_json in rows:
                glossary.append({
                    "term": term,
                    "surface_forms": json.loads(surface_json or "[]"),
                    "relevant": True,
                    "confidence": 1.0,
                })
            logger.info(
                "crypto_cards_glossary: loaded %d terms from intent_glossary (fallback)",
                len(glossary),
            )
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()

    return glossary


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def make_crypto_cards_llm_client() -> LLMClient:
    """Create an LLMClient for the crypto cards pipeline.

    Uses CRYPTO_CARDS_OPENROUTER_API_KEY for isolated cost tracking.
    Falls back to the default LLMClient if the key is not set.
    """
    key = os.environ.get("CRYPTO_CARDS_OPENROUTER_API_KEY")
    if not key:
        logger.warning(
            "CRYPTO_CARDS_OPENROUTER_API_KEY not set — falling back to default LLMClient. "
            "Crypto cards pipeline costs will NOT be tracked separately."
        )
        return LLMClient()

    client = LLMClient()
    client.base_url = _OPENROUTER_URL.replace("/chat/completions", "")
    client.api_key  = key
    return client


# ---------------------------------------------------------------------------
# Glossary lookup helper
# ---------------------------------------------------------------------------

def _find_glossary_terms(text: str, glossary: list[dict]) -> list[str]:
    """Return glossary terms that appear in text (case-insensitive substring match)."""
    if not text or not glossary:
        return []
    low = text.lower()
    found: list[str] = []
    for entry in glossary:
        term = entry.get("term", "")
        surfaces = entry.get("surface_forms") or []
        all_forms = [term] + surfaces
        if any(f.lower() in low for f in all_forms if f):
            found.append(term)
    return found


# ---------------------------------------------------------------------------
# LLM call helpers (synchronous httpx — whole pipeline stays sync)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Extract a JSON object from an LLM response string.

    Handles markdown fences, trailing text, and truncated responses.
    """
    clean = re.sub(r"```json\s*|```", "", raw).strip()
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1]
    return clean


def _call_llm_plain(
    *,
    system: str,
    user: str,
    max_tokens: int = 10,
    model: Optional[str] = None,
) -> str:
    """Call OpenRouter synchronously, return raw text (no JSON parsing).

    Used for Stage 1 YES/NO — must NOT set response_format: json_object.
    Returns "_error" on any failure (caller treats it as fail-open YES).
    """
    import httpx

    _key = os.environ.get("CRYPTO_CARDS_OPENROUTER_API_KEY", "")
    if not _key:
        return "_error"

    _model = (
        model
        or os.environ.get("CRYPTO_CARDS_STAGE1_MODEL")
        or os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")
    )

    payload = {
        "model": _model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("crypto_cards_classifier: plain LLM call failed: %s", exc)
        return "_error"


def _call_llm(
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: Optional[str] = None,
) -> dict:
    """Call OpenRouter synchronously, return parsed JSON or {"_error": ...}."""
    import httpx

    _key = os.environ.get("CRYPTO_CARDS_OPENROUTER_API_KEY", "")
    if not _key:
        return {"_error": "no CRYPTO_CARDS_OPENROUTER_API_KEY"}

    _model = model or os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")

    payload = {
        "model": _model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    raw = ""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            clean = _extract_json(raw)
            return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error(
            "crypto_cards_classifier: JSON parse error (model=%s): %s | raw=%r",
            _model, exc, raw[:300],
        )
        return {"_error": f"json_parse: {exc}"}
    except Exception as exc:
        logger.error("crypto_cards_classifier: LLM call failed (model=%s): %s", _model, exc)
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_stage1(
    text: str,
    username: Optional[str],
    cfg: CryptoCardsExtractionConfig,
) -> bool:
    """Stage 1 micro-LLM: YES/NO. Returns True = proceed to Stage 3.

    Uses plain text response, NOT json_object — Stage 1 returns a single word.
    Fail-open: any error or ambiguous response defaults to YES.
    """
    author = username or "unknown"
    user_msg = f"[Author]: {author}\n[Message]: {text}"

    raw = _call_llm_plain(
        system=_CRYPTO_CARDS_STAGE1_SYSTEM,
        user=user_msg,
        max_tokens=10,
        model=cfg.stage1_model(),
    )

    if raw == "_error":
        logger.warning(
            "crypto_cards_classifier: stage1 error — defaulting to YES (fail-open)"
        )
        return True

    return "NO" not in raw.upper()


def _run_stage3(
    text: str,
    username: Optional[str],
    glossary_terms: list[str],
) -> dict:
    """Stage 3 role-play extractor: lead_type + confidence + evidence_quote + rationale."""
    author = username or "unknown"
    terms_hint = (
        f"\n\nGlossary terms found in message: {', '.join(glossary_terms)}"
        if glossary_terms else ""
    )
    user_msg = _CRYPTO_CARDS_STAGE3_USER.format(
        username=author,
        text=text,
        terms_hint=terms_hint,
    )
    return _call_llm(
        system=_CRYPTO_CARDS_STAGE3_SYSTEM,
        user=user_msg,
        max_tokens=400,
    )


def _run_stage4(
    text: str,
    stage3_result: dict,
    cfg: CryptoCardsExtractionConfig,
) -> dict:
    """Stage 4 skeptical judge: REAL_CARD_SEEKER | REAL_CARD_PROVIDER | MISTAKE."""
    user_msg = _CRYPTO_CARDS_STAGE4_USER.format(
        text=text,
        lead_type=stage3_result.get("lead_type"),
        confidence=stage3_result.get("confidence"),
        evidence_quote=stage3_result.get("evidence_quote"),
        rationale=stage3_result.get("rationale"),
    )
    return _call_llm(
        system=_CRYPTO_CARDS_STAGE4_SYSTEM,
        user=user_msg,
        max_tokens=200,
        model=cfg.judge_model(),
    )


# ---------------------------------------------------------------------------
# Verdict alignment
# ---------------------------------------------------------------------------

_LEAD_TYPE_TO_VERDICT = {
    "card_seeker":  "REAL_CARD_SEEKER",
    "card_provider": "REAL_CARD_PROVIDER",
}


def _verdict_matches(lead_type: Optional[str], verdict: Optional[str]) -> bool:
    """True iff Stage 3 lead_type and Stage 4 verdict are consistent."""
    expected = _LEAD_TYPE_TO_VERDICT.get(lead_type or "")
    return expected is not None and verdict == expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_crypto_cards_message(
    text: str,
    glossary: list[dict],
    llm: LLMClient,
    cfg: CryptoCardsExtractionConfig,
    *,
    message_id: Optional[int]      = None,
    username:   Optional[str]      = None,
    timestamp:  Optional[datetime] = None,
) -> dict:
    """Classify one message through the full crypto cards pipeline.

    Always returns a dict (CryptoCardsDecision.to_dict() shape). Never raises —
    errors are logged and reflected in the result fields.

    Args:
        text:       Raw message text.
        glossary:   Output of load_crypto_cards_glossary(). Pass [] to skip.
        llm:        LLMClient from make_crypto_cards_llm_client().
        cfg:        CryptoCardsExtractionConfig.
        message_id, username, timestamp: Message metadata.
    """
    t0 = time.perf_counter()

    decision = CryptoCardsDecision(
        message_id=message_id,
        timestamp=timestamp,
        username=username,
        text=text,
    )

    def _done(reason: str) -> dict:
        decision.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "crypto_cards_classifier: DONE message_id=%s reason=%s elapsed=%.0fms",
            message_id, reason, decision.elapsed_ms,
        )
        return decision.to_dict()

    # ── Stage 0: length check ─────────────────────────────────────────────────
    text = (text or "").strip()
    if len(text) < cfg.skip_min_text_length:
        decision.rationale = "[skipped: text too short]"
        return _done("too_short")

    # ── Stage 0b: excluded authors ────────────────────────────────────────────
    if username and username in cfg.excluded_authors:
        decision.rationale = f"[skipped: excluded author {username!r}]"
        return _done("excluded_author")

    # ── Stage 2: glossary lookup ──────────────────────────────────────────────
    glossary_terms = _find_glossary_terms(text, glossary)
    decision.glossary_terms_seen = glossary_terms
    logger.debug(
        "crypto_cards_classifier: glossary_terms=%s message_id=%s",
        glossary_terms, message_id,
    )

    # ── Stage 3: pre-filter ───────────────────────────────────────────────────
    if cfg.pre_filter_enabled:
        from intent_filter_crypto_cards import pre_filter
        passes = pre_filter(text, cfg.pre_filter_db_path)
        if not passes:
            decision.rationale = "[pre_filter: blocked]"
            return _done("pre_filter_miss")

    # ── Stage 4: Stage 1 micro-LLM YES/NO ────────────────────────────────────
    if cfg.stage1_enabled:
        yes = _run_stage1(text, username, cfg)
        if not yes:
            decision.rationale = "[stage1: micro-LLM said NO]"
            return _done("stage1_miss")

    # ── Stage 5: Stage 3 role-play extractor ─────────────────────────────────
    ext = _run_stage3(text, username, glossary_terms)
    if ext.get("_error"):
        logger.error(
            "crypto_cards_classifier: stage3 error message_id=%s: %s",
            message_id, ext["_error"],
        )
        decision.rationale = f"[stage3_error: {ext['_error']}]"
        return _done("stage3_error")

    decision.lead_type      = ext.get("lead_type")
    decision.confidence     = ext.get("confidence")
    decision.evidence_quote = ext.get("evidence_quote")
    decision.rationale      = ext.get("rationale")

    # If Stage 3 found no lead type → stop here
    if not decision.lead_type:
        decision.is_lead = False
        return _done("stage3_miss")

    decision.is_lead = True

    # ── Stage 6: Stage 4 judge ────────────────────────────────────────────────
    if cfg.judge_enabled:
        judge = _run_stage4(text, ext, cfg)
        if judge.get("_error"):
            logger.warning(
                "crypto_cards_classifier: stage4 error, using stage3 result message_id=%s",
                message_id,
            )
            # Judge failed → derive verdict from Stage 3 lead_type
            decision.verdict = _LEAD_TYPE_TO_VERDICT.get(decision.lead_type or "")
            decision.notes   = "[stage4_error: using stage3 result]"
        else:
            decision.verdict = judge.get("verdict")
            decision.notes   = judge.get("notes")

        # ── Stage 7: alignment check ──────────────────────────────────────────
        if not _verdict_matches(decision.lead_type, decision.verdict):
            logger.info(
                "crypto_cards_classifier: type_mismatch message_id=%s "
                "lead_type=%s verdict=%s",
                message_id, decision.lead_type, decision.verdict,
            )
            decision.is_lead = False
            decision.verdict = "MISTAKE"
            decision.notes   = (
                f"[type_mismatch: stage3={decision.lead_type}, stage4={decision.verdict}] "
                + (decision.notes or "")
            )
            return _done("type_mismatch")

        # MISTAKE from judge
        if decision.verdict == "MISTAKE":
            decision.is_lead = False
            return _done("judge_mistake")

    else:
        # Judge disabled — derive verdict from Stage 3
        decision.verdict = _LEAD_TYPE_TO_VERDICT.get(decision.lead_type or "")

    return _done("ok")
