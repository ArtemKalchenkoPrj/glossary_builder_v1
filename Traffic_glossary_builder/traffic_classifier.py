"""Traffic vertical classifier.

Synchronous pipeline — designed to be called via asyncio.to_thread() from
the webhook, or directly from a Jupyter notebook for testing.

Public API:
    classify_traffic_message(text, glossary, llm, cfg, ...) -> dict
    load_traffic_glossary(db_path) -> list[dict]
    make_traffic_llm_client() -> LLMClient

Pipeline stages:
    0. Length check (< skip_min_text_length → skip)
    1. Excluded-author check
    2. Glossary lookup (finds matching terms)
    3. Pre-filter (intent_filter_traffic.pre_filter)  [if enabled]
    4. Stage 1 micro-LLM YES/NO                       [if enabled]
    5. Full LLM → is_lead + lead_type + fields
    6. Judge LLM → TRAFFIC_PROVIDER | TRAFFIC_LEAD | MISTAKE  [if enabled]
    7. Verdict alignment check (extractor lead_type must match judge verdict)

Result is always a dict (same shape as TrafficDecision.to_dict()) so the
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

from .traffic_decision import TrafficDecision, TrafficExtractionConfig
from .traffic_prompt import (
    _TRAFFIC_STAGE1_SYSTEM,
    TRAFFIC_DEFINITION,
    _TRAFFIC_SYSTEM_TEMPLATE,
    _TRAFFIC_JUDGE_SYSTEM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter URL (same as other classifiers in the project)
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Glossary loader
# ---------------------------------------------------------------------------

def load_traffic_glossary(db_path: str) -> list[dict]:
    """Load terms from traffic_glossary_enriched (preferred) or intent_glossary.

    Returns a list of dicts with keys: term, surface_forms, relevant, confidence.
    Returns [] if the DB doesn't exist yet (pre-filter will be skipped gracefully).
    """
    path = Path(db_path)
    if not path.exists():
        logger.warning("traffic_glossary: DB not found at %s — glossary will be empty", db_path)
        return []

    conn = sqlite3.connect(db_path)
    glossary: list[dict] = []
    try:
        # ── Prefer enriched table ─────────────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT term, surface_forms, relevant, confidence "
                "FROM traffic_glossary_enriched WHERE relevant = 1"
            ).fetchall()
            for term, surface_json, relevant, confidence in rows:
                glossary.append({
                    "term": term,
                    "surface_forms": json.loads(surface_json or "[]"),
                    "relevant": bool(relevant),
                    "confidence": confidence or 0.0,
                })
            if glossary:
                logger.info("traffic_glossary: loaded %d terms from traffic_glossary_enriched", len(glossary))
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
            logger.info("traffic_glossary: loaded %d terms from intent_glossary (fallback)", len(glossary))
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()

    return glossary


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def make_traffic_llm_client() -> LLMClient:
    """Create an LLMClient for the traffic pipeline.

    Uses TRAFFIC_OPENROUTER_API_KEY for isolated cost tracking.
    Falls back to the default LLMClient if the key is not set.
    """
    key = os.environ.get("TRAFFIC_OPENROUTER_API_KEY")
    if not key:
        logger.warning(
            "TRAFFIC_OPENROUTER_API_KEY not set — falling back to default LLMClient. "
            "Traffic pipeline costs will NOT be tracked separately."
        )
        return LLMClient()

    # Point to OpenRouter with the traffic-specific key
    client = LLMClient()
    client.base_url = _OPENROUTER_URL.replace("/chat/completions", "")
    client.api_key  = key
    return client


# ---------------------------------------------------------------------------
# Glossary lookup helper
# ---------------------------------------------------------------------------

def _find_glossary_terms(text: str, glossary: list[dict]) -> list[str]:
    """Return glossary terms that appear in text (case-insensitive substring)."""
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
# LLM call helpers (synchronous httpx, so the whole function stays sync)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Aggressively extract a JSON object from an LLM response string.

    Handles:
    - markdown fences (```json ... ```)
    - trailing text / commentary after the closing brace
    - truncated responses — finds first '{' and last '}'
    """
    clean = re.sub(r"```json\s*|```", "", raw).strip()
    start = clean.find("{" )
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

    Used for Stage 1 YES/NO — must NOT set response_format: json_object
    because OpenAI/Azure require 'json' in the messages when that mode is on.
    Reads TRAFFIC_OPENROUTER_API_KEY from env.
    Returns "_error" on any failure (caller treats it as fail-open YES).
    """
    import httpx

    _key = os.environ.get("TRAFFIC_OPENROUTER_API_KEY", "")
    if not _key:
        return "_error"

    _model = model or os.environ.get("TRAFFIC_STAGE1_MODEL") or os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")

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
        logger.error("traffic_classifier: plain LLM call failed: %s", exc)
        return "_error"


def _call_llm(
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Call OpenRouter synchronously, return parsed JSON or {"_error": ...}."""
    import httpx

    _key = api_key or os.environ.get("TRAFFIC_OPENROUTER_API_KEY") or ""
    if not _key:
        # Fallback: try the standard LLM path by raising so caller can use LLMClient
        return {"_error": "no TRAFFIC_OPENROUTER_API_KEY"}

    _model = model or os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")

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
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            clean = _extract_json(raw)
            return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error("traffic_classifier: JSON parse error (model=%s): %s | raw=%r",
                     _model, exc, raw[:300])
        return {"_error": f"json_parse: {exc}"}
    except Exception as exc:
        logger.error("traffic_classifier: LLM call failed (model=%s): %s", _model, exc)
        return {"_error": str(exc)}


def _call_llm_via_client(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    model_override: Optional[str] = None,
) -> dict:
    """Use LLMClient.complete_json() with optional model override.

    NOTE: only call this when the prompt explicitly asks for JSON output.
    OpenAI/Azure require the word 'json' to appear in the messages when
    json_object response format is used. For YES/NO calls use
    _call_llm_plain_via_client() instead.
    """
    original_model = llm.model
    try:
        if model_override:
            llm.model = model_override
        data, _ = llm.complete_json(system, user)
        return data
    except Exception as exc:
        logger.error("traffic_classifier: LLMClient call failed: %s", exc)
        return {"_error": str(exc)}
    finally:
        llm.model = original_model


def _call_llm_plain_via_client(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    model_override: Optional[str] = None,
) -> str:
    """Use LLMClient.complete() — plain text response, no JSON parsing.

    Use this for Stage 1 YES/NO calls where response_format: json_object
    must NOT be set (OpenAI rejects it if 'json' isn't in the messages,
    and Stage 1 intentionally returns a plain word, not a JSON object).
    """
    original_model = llm.model
    try:
        if model_override:
            llm.model = model_override
        resp = llm.complete(system, user)
        return resp.text.strip()
    except Exception as exc:
        logger.error("traffic_classifier: LLMClient plain call failed: %s", exc)
        return "_error"
    finally:
        llm.model = original_model


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_stage1(text: str, username: Optional[str], llm: LLMClient,
                cfg: TrafficExtractionConfig) -> bool:
    """Stage 1 micro-LLM: YES/NO. Returns True = proceed to full LLM.

    Uses complete() (plain text), NOT complete_json() — Stage 1 returns a
    single word and must NOT use response_format: json_object because:
      (a) the prompt doesn't ask for JSON, so OpenAI/Azure reject the request
      (b) a plain YES/NO is cheaper and faster to parse than JSON
    """
    author = username or "unknown"
    user_msg = f"[Автор]: {author}\n[Сообщение]: {text}"

    raw = _call_llm_plain(
        system=_TRAFFIC_STAGE1_SYSTEM,
        user=user_msg,
        max_tokens=10,
        model=cfg.stage1_model(),
    )

    if raw == "_error":
        logger.warning("traffic_classifier: stage1 error — defaulting to YES (fail-open)")
        return True

    return "NO" not in raw.upper()


def _run_full_llm(text: str, username: Optional[str], llm: LLMClient,
                  glossary_terms: list[str]) -> dict:
    """Full LLM extraction: returns structured dict with is_lead, lead_type, etc."""
    system = _TRAFFIC_SYSTEM_TEMPLATE.format(traffic_definition=TRAFFIC_DEFINITION)
    author = username or "unknown"
    terms_hint = (
        f"\n\nГлоссарные термины найдены в тексте: {', '.join(glossary_terms)}"
        if glossary_terms else ""
    )
    user_msg = f"[Автор]: {author}\n[Сообщение]: {text}{terms_hint}"

    # Use httpx directly so TRAFFIC_OPENROUTER_API_KEY is always picked up.
    # _call_llm_via_client(llm) would use the LLMClient's internal SDK key
    # which is set at __init__ time and ignores TRAFFIC_OPENROUTER_API_KEY.
    return _call_llm(system=system, user=user_msg, max_tokens=600)


def _run_judge(text: str, username: Optional[str], extractor_result: dict,
               llm: LLMClient, cfg: TrafficExtractionConfig) -> dict:
    """Judge LLM: TRAFFIC_PROVIDER | TRAFFIC_LEAD | MISTAKE."""
    author = username or "unknown"
    user_msg = (
        f"[Автор]: {author}\n"
        f"[Сообщение]: {text}\n\n"
        f"[Решение экстрактора]:\n"
        f"  is_lead:   {extractor_result.get('is_lead')}\n"
        f"  lead_type: {extractor_result.get('lead_type')}\n"
        f"  rationale: {extractor_result.get('rationale')}"
    )
    return _call_llm(
        system=_TRAFFIC_JUDGE_SYSTEM,
        user=user_msg,
        max_tokens=200,
        model=cfg.judge_model(),
    )


# ---------------------------------------------------------------------------
# Verdict alignment check
# ---------------------------------------------------------------------------

_LEAD_TYPE_TO_VERDICT = {
    "traffic_provider": "TRAFFIC_PROVIDER",
    "seeking_traffic":  "TRAFFIC_LEAD",
}


def _verdict_matches(lead_type: Optional[str], verdict: Optional[str]) -> bool:
    """True iff the extractor lead_type and judge verdict are consistent."""
    expected = _LEAD_TYPE_TO_VERDICT.get(lead_type or "")
    return expected is not None and verdict == expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_traffic_message(
    text: str,
    glossary: list[dict],
    llm: LLMClient,
    cfg: TrafficExtractionConfig,
    *,
    message_id: Optional[int]      = None,
    username:   Optional[str]      = None,
    timestamp:  Optional[datetime] = None,
    context:    Optional[list]     = None,   # reserved, not used yet
) -> dict:
    """Classify one message through the full traffic pipeline.

    Always returns a dict (TrafficDecision.to_dict() shape). Never raises —
    errors are logged and reflected in the result fields.

    Args:
        text:       Raw message text.
        glossary:   Output of load_traffic_glossary(). Pass [] to skip.
        llm:        LLMClient from make_traffic_llm_client().
        cfg:        TrafficExtractionConfig (set pre_filter_enabled=False for notebook testing).
        message_id, username, timestamp: Message metadata.
        context:    Reserved for future use (surrounding messages).
    """
    t0 = time.perf_counter()

    decision = TrafficDecision(
        message_id=message_id,
        timestamp=timestamp,
        username=username,
        text=text,
    )

    def _done(reason: str) -> dict:
        decision.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("traffic_classifier: DONE message_id=%s reason=%s elapsed=%.0fms",
                     message_id, reason, decision.elapsed_ms)
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
    logger.debug("traffic_classifier: glossary_terms=%s message_id=%s", glossary_terms, message_id)

    # ── Stage 3: pre-filter ───────────────────────────────────────────────────
    if cfg.pre_filter_enabled:
        from intent_filter_traffic import pre_filter
        passes = pre_filter(text, cfg.pre_filter_db_path)
        if not passes and not glossary_terms:
            decision.rationale = "[pre_filter: no domain terms or intent signals]"
            return _done("pre_filter_miss")

    # ── Stage 4: Stage 1 micro-LLM ───────────────────────────────────────────
    if cfg.stage1_enabled:
        yes = _run_stage1(text, username, llm, cfg)
        if not yes:
            decision.rationale = "[stage1: micro-LLM said NO]"
            return _done("stage1_miss")

    # ── Stage 5: Full LLM ────────────────────────────────────────────────────
    ext = _run_full_llm(text, username, llm, glossary_terms)
    if ext.get("_error"):
        logger.error("traffic_classifier: full LLM error message_id=%s: %s",
                     message_id, ext["_error"])
        decision.rationale = f"[full_llm_error: {ext['_error']}]"
        return _done("llm_error")

    decision.is_lead        = bool(ext.get("is_lead", False))
    decision.lead_type      = ext.get("lead_type")
    decision.confidence     = ext.get("confidence")
    decision.vertical       = _ensure_list(ext.get("vertical"))
    decision.traffic_source = _ensure_list(ext.get("traffic_source"))
    decision.geo            = _ensure_list(ext.get("geo"))
    decision.pricing_model  = _ensure_list(ext.get("pricing_model"))
    decision.evidence_quote = ext.get("evidence_quote")
    decision.rationale      = ext.get("rationale")

    if not decision.is_lead:
        return _done("llm_miss")

    # ── Stage 6: Judge ────────────────────────────────────────────────────────
    if cfg.judge_enabled:
        judge = _run_judge(text, username, ext, llm, cfg)
        if judge.get("_error"):
            logger.warning("traffic_classifier: judge error, using extractor result message_id=%s", message_id)
            # Judge failed → use extractor verdict as-is
            decision.verdict      = _lead_type_to_verdict(decision.lead_type)
            decision.judge_reason = "[judge_error: using extractor result]"
        else:
            decision.verdict      = judge.get("verdict")
            decision.judge_reason = judge.get("judge_reason")

        # ── Stage 7: alignment check ──────────────────────────────────────────
        if not _verdict_matches(decision.lead_type, decision.verdict):
            logger.info(
                "traffic_classifier: type_mismatch message_id=%s lead_type=%s verdict=%s",
                message_id, decision.lead_type, decision.verdict,
            )
            decision.is_lead      = False
            decision.verdict      = "MISTAKE"
            decision.judge_reason = (
                f"[type_mismatch: extractor={decision.lead_type}, judge={decision.verdict}] "
                + (decision.judge_reason or "")
            )
            return _done("type_mismatch")
    else:
        # Judge disabled — derive verdict from lead_type
        decision.verdict = _lead_type_to_verdict(decision.lead_type)

    return _done("ok")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _lead_type_to_verdict(lead_type: Optional[str]) -> Optional[str]:
    return _LEAD_TYPE_TO_VERDICT.get(lead_type or "")