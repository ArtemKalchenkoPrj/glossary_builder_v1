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
import threading
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
from .traffic_gate_prompt import TRAFFIC_GATE_SYSTEM, TRAFFIC_GATE_USER_TEMPLATE
from .traffic_merchant_prompt import TRAFFIC_MERCHANT_SYSTEM, TRAFFIC_MERCHANT_USER_TEMPLATE

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

_shared_httpx_client: "httpx.Client | None" = None


def _get_shared_client() -> "httpx.Client":
    """Lazy singleton httpx.Client for prompt caching via HTTP keep-alive."""
    import httpx
    global _shared_httpx_client
    if _shared_httpx_client is None:
        _shared_httpx_client = httpx.Client(timeout=60.0)
    return _shared_httpx_client


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
        client = _get_shared_client()
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
        client = _get_shared_client()
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


_anthropic_clients: dict = {}
_anthropic_clients_lock = threading.Lock()


def _get_anthropic_client(model: str):
    with _anthropic_clients_lock:
        client = _anthropic_clients.get(model)
        if client is None:
            from glossary_builder.llm import make_anthropic_llm
            client = make_anthropic_llm(model)
            _anthropic_clients[model] = client
        return client


def _call_llm_anthropic(*, system: str, user: str, max_tokens: int, model: str) -> dict:
    """Call an Anthropic model (OpenRouter fallback inside the client), parse JSON."""
    raw = ""
    try:
        raw = _get_anthropic_client(model).complete(system, user, max_tokens=max_tokens).text.strip()
        return json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        logger.error("traffic_classifier: anthropic JSON parse error (model=%s): %s | raw=%r",
                     model, exc, raw[:300])
        return {"_error": f"json_parse: {exc}"}
    except Exception as exc:
        logger.error("traffic_classifier: anthropic call failed (model=%s): %s", model, exc)
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
    from vertical_enum import VERTICAL_RULES
    system = _TRAFFIC_SYSTEM_TEMPLATE.format(
        traffic_definition=TRAFFIC_DEFINITION,
        vertical_rules=VERTICAL_RULES,
    )
    author = username or "unknown"
    terms_hint = (
        f"\n\nГлоссарные термины найдены в тексте: {', '.join(glossary_terms)}"
        if glossary_terms else ""
    )
    user_msg = f"[Автор]: {author}\n[Сообщение]: {text}{terms_hint}"

    # Optional: Anthropic extractor (e.g. claude-haiku-5-5) via ANTHROPIC_API_KEY.
    # AnthropicLLMClient falls back to the same model on OpenRouter on spend/rate
    # limit; if the call still fails we fall through to the OpenRouter model below.
    anthropic_model = os.environ.get("TRAFFIC_EXTRACTOR_ANTHROPIC_MODEL")
    if anthropic_model:
        result = _call_llm_anthropic(
            system=system, user=user_msg, max_tokens=600, model=anthropic_model,
        )
        if not result.get("_error"):
            return result
        logger.warning("traffic_classifier: anthropic extractor failed (%s) — "
                       "falling back to OpenRouter model", result["_error"])

    # Use httpx directly so TRAFFIC_OPENROUTER_API_KEY is always picked up.
    # _call_llm_via_client(llm) would use the LLMClient's internal SDK key
    # which is set at __init__ time and ignores TRAFFIC_OPENROUTER_API_KEY.
    return _call_llm(
        system=system,
        user=user_msg,
        max_tokens=600,
        model=os.environ.get("TRAFFIC_EXTRACTOR_MODEL"),
    )


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
# Certainty gate (runs only after the judge)
# ---------------------------------------------------------------------------
#
#  * extractor and judge agree on a lead -> gate model (Haiku) checks how obvious it is:
#        lead -> CERTAIN;  uncertain / not_lead -> UNCERTAIN (operator decides)
#  * extractor says lead, judge disagrees (other role or MISTAKE) -> resolver (Sonnet)
#    makes the final call:
#        lead + clear role               -> lead with that role, CERTAIN
#        lead + ambiguous role/uncertain -> lead, UNCERTAIN ("role_unclear" or its reason)
#        not_lead                        -> MISTAKE (not persisted)
#  * any gate API / parse error -> UNCERTAIN with reason "gate_api_error" (fail-safe:
#    the operator sees it, nothing goes to the client automatically)

_GATE_LABELS = ("lead", "not_lead", "uncertain")
_ROLE_TO_LEAD_TYPE = {"seller": "traffic_provider", "buyer": "seeking_traffic"}
_GATE_REASONS = ("type_unspecified", "infra", "ad_vs_payment", "offer_no_context", "sms", "other")


def _call_gate(text: str, username: Optional[str], model: str) -> dict:
    """Run the gate prompt on an Anthropic model. Returns parsed JSON or {"_error": ...}."""
    user = TRAFFIC_GATE_USER_TEMPLATE.format(username=username or "unknown", text=text)
    result = _call_llm_anthropic(system=TRAFFIC_GATE_SYSTEM, user=user, max_tokens=400, model=model)
    if result.get("_error"):
        return result
    if result.get("label") not in _GATE_LABELS:
        return {"_error": f"bad gate label: {result.get('label')!r}"}
    return result


def _gate_reason_text(model: str, gate: dict) -> str:
    return (f"[gate {model}: {gate.get('label')}/{gate.get('role')}/"
            f"{gate.get('uncertainty_reason')}] {gate.get('reason') or ''}").strip()


def _apply_gate_to_lead(decision: TrafficDecision, text: str, username: Optional[str],
                        cfg: TrafficExtractionConfig) -> None:
    """Extractor and judge agree: mark the lead CERTAIN or UNCERTAIN."""
    model = cfg.gate_model()
    gate = _call_gate(text, username, model)
    if gate.get("_error"):
        logger.warning("traffic_classifier: gate error message_id=%s: %s",
                       decision.message_id, gate["_error"])
        decision.certainty = "UNCERTAIN"
        decision.gate_uncertainty_reason = "gate_api_error"
        decision.gate_reason = f"[gate {model} error] {gate['_error']}"[:500]
        return
    decision.gate_reason = _gate_reason_text(model, gate)
    if gate["label"] == "lead":
        decision.certainty = "CERTAIN"
        decision.gate_uncertainty_reason = None
    else:
        decision.certainty = "UNCERTAIN"
        reason = gate.get("uncertainty_reason")
        decision.gate_uncertainty_reason = (
            reason if reason in _GATE_REASONS
            else "gate_not_lead" if gate["label"] == "not_lead" else "other"
        )


def _resolve_mismatch(decision: TrafficDecision, text: str, username: Optional[str],
                      cfg: TrafficExtractionConfig, judge_verdict: Optional[str]) -> None:
    """Extractor says lead, judge disagrees: the resolver model makes the final call."""
    model = cfg.gate_resolver_model()
    gate = _call_gate(text, username, model)
    prefix = f"[mismatch: extractor={decision.lead_type}, judge={judge_verdict}] "
    if gate.get("_error"):
        logger.warning("traffic_classifier: resolver error message_id=%s: %s",
                       decision.message_id, gate["_error"])
        # keep the extractor's lead, but never auto-send it
        decision.is_lead = True
        decision.verdict = _lead_type_to_verdict(decision.lead_type)
        decision.certainty = "UNCERTAIN"
        decision.gate_uncertainty_reason = "gate_api_error"
        decision.gate_reason = (prefix + f"[resolver {model} error] {gate['_error']}")[:500]
        return

    decision.gate_reason = prefix + _gate_reason_text(model, gate)
    label, role = gate["label"], gate.get("role")
    if label == "not_lead":
        decision.is_lead = False
        decision.verdict = "MISTAKE"
        decision.certainty = None
        decision.gate_uncertainty_reason = None
        return

    # lead or uncertain: keep it, take the resolver's role when it is clear
    if role in _ROLE_TO_LEAD_TYPE:
        decision.lead_type = _ROLE_TO_LEAD_TYPE[role]
    decision.is_lead = True
    decision.verdict = _lead_type_to_verdict(decision.lead_type)
    if label == "lead" and role in _ROLE_TO_LEAD_TYPE:
        decision.certainty = "CERTAIN"
        decision.gate_uncertainty_reason = None
    else:
        decision.certainty = "UNCERTAIN"
        reason = gate.get("uncertainty_reason")
        decision.gate_uncertainty_reason = (
            "role_unclear" if role not in _ROLE_TO_LEAD_TYPE
            else reason if reason in _GATE_REASONS else "other"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# is_merchant gate (runs last, seeking_traffic leads only)
# ---------------------------------------------------------------------------
#
#  True  = the author explicitly presents himself as the owner / direct advertiser.
#  False = anything else (intermediary or no self-description) - precision first.
#  NULL  = gate off, not a seeking_traffic lead, or API/parse error (never guessed).
#  Never changes is_lead / verdict / certainty.

_MERCHANT_BASES = ("self_owner", "self_intermediary", "no_self_description")


def _apply_merchant_gate(decision: TrafficDecision, text: str, username: Optional[str],
                         cfg: TrafficExtractionConfig) -> None:
    model = cfg.merchant_gate_model()
    user = TRAFFIC_MERCHANT_USER_TEMPLATE.format(username=username or "unknown", text=text)
    result = _call_llm_anthropic(system=TRAFFIC_MERCHANT_SYSTEM, user=user, max_tokens=300, model=model)
    value = str(result.get("is_merchant", "")).strip().lower()
    if result.get("_error") or value not in ("true", "false"):
        logger.warning("traffic_classifier: merchant gate error message_id=%s: %s",
                       decision.message_id, result.get("_error") or f"bad value {result.get('is_merchant')!r}")
        decision.is_merchant = None
        decision.merchant_reason = f"[merchant gate {model} error] {result.get('_error') or 'bad value'}"[:300]
        return
    decision.is_merchant = value == "true"
    basis = result.get("basis") if result.get("basis") in _MERCHANT_BASES else "?"
    decision.merchant_reason = f"[merchant gate {model}: {basis}] {result.get('reason') or ''}".strip()[:300]


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
        if (cfg.merchant_gate_enabled() and decision.is_lead
                and decision.verdict == "TRAFFIC_LEAD" and decision.lead_type == "seeking_traffic"):
            _apply_merchant_gate(decision, text, username, cfg)
            logger.info("traffic_classifier: merchant gate message_id=%s is_merchant=%s %s",
                        message_id, decision.is_merchant, decision.merchant_reason)
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
        if not passes:
            decision.rationale = "[pre_filter: blocked]"
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
    gate_on = cfg.gate_enabled()
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
            judge_verdict = decision.verdict
            logger.info(
                "traffic_classifier: type_mismatch message_id=%s lead_type=%s verdict=%s",
                message_id, decision.lead_type, judge_verdict,
            )
            if gate_on:
                # ── Stage 8a: resolver decides instead of dropping the lead ──
                _resolve_mismatch(decision, text, username, cfg, judge_verdict)
                logger.info(
                    "traffic_classifier: mismatch resolved message_id=%s verdict=%s certainty=%s reason=%s",
                    message_id, decision.verdict, decision.certainty, decision.gate_uncertainty_reason,
                )
                return _done("type_mismatch_resolved" if decision.is_lead else "type_mismatch")
            decision.is_lead      = False
            decision.verdict      = "MISTAKE"
            decision.judge_reason = (
                f"[type_mismatch: extractor={decision.lead_type}, judge={judge_verdict}] "
                + (decision.judge_reason or "")
            )
            return _done("type_mismatch")
    else:
        # Judge disabled — derive verdict from lead_type
        decision.verdict = _lead_type_to_verdict(decision.lead_type)

    # ── Stage 8b: certainty gate on confirmed leads ───────────────────────────
    if gate_on and decision.is_lead and decision.verdict in ("TRAFFIC_PROVIDER", "TRAFFIC_LEAD"):
        _apply_gate_to_lead(decision, text, username, cfg)
        logger.info(
            "traffic_classifier: gate message_id=%s certainty=%s reason=%s",
            message_id, decision.certainty, decision.gate_uncertainty_reason,
        )

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

# ---------------------------------------------------------------------------
# Effective configuration (for the webhook startup log)
# ---------------------------------------------------------------------------

def describe_traffic_config(cfg: TrafficExtractionConfig) -> list[tuple[str, str]]:
    """Return (label, value) pairs describing which models the traffic pipeline
    will actually call, resolved with the same fallbacks the stages use."""
    from glossary_builder.llm import _ANTHROPIC_TO_OPENROUTER

    default = os.environ.get("GLOSSARY_MODEL") or "openai/gpt-4.1-nano"

    def _openrouter(env: str) -> str:
        value = os.environ.get(env)
        return value if value else f"{default} (GLOSSARY_MODEL / nano fallback, {env} not set)"

    def _anthropic(model: str) -> str:
        or_model = _ANTHROPIC_TO_OPENROUTER.get(model, f"anthropic/{model}")
        return f"{model} via ANTHROPIC_API_KEY (on 400/429: OpenRouter {or_model})"

    lines: list[tuple[str, str]] = []
    lines.append(("Pre-filter", str(cfg.pre_filter_enabled)))
    lines.append(("Stage1", f"{cfg.stage1_enabled} | model: {_openrouter('TRAFFIC_STAGE1_MODEL')}"))

    anthropic_model = os.environ.get("TRAFFIC_EXTRACTOR_ANTHROPIC_MODEL")
    if anthropic_model:
        lines.append(("Extractor", _anthropic(anthropic_model)))
        lines.append(("Extractor fallback", _openrouter("TRAFFIC_EXTRACTOR_MODEL")))
    else:
        lines.append(("Extractor", f"{_openrouter('TRAFFIC_EXTRACTOR_MODEL')} (OpenRouter; "
                                   "TRAFFIC_EXTRACTOR_ANTHROPIC_MODEL not set)"))

    lines.append(("Judge", f"{cfg.judge_enabled} | model: {_openrouter('TRAFFIC_JUDGE_MODEL')}"))

    if cfg.gate_enabled():
        lines.append(("Certainty gate", "ON (TRAFFIC_GATE_ENABLED)"))
        lines.append(("Gate: leads", _anthropic(cfg.gate_model())))
        lines.append(("Gate: mismatches", _anthropic(cfg.gate_resolver_model())))
        lines.append(("Gate: on error", "UNCERTAIN / gate_api_error"))
    else:
        lines.append(("Certainty gate", "OFF (TRAFFIC_GATE_ENABLED not set) - certainty=NULL, "
                                        "extractor/judge mismatches are dropped"))

    if cfg.merchant_gate_enabled():
        lines.append(("Merchant gate", "ON (TRAFFIC_MERCHANT_GATE_ENABLED) - seeking_traffic leads only"))
        lines.append(("Merchant gate model", _anthropic(cfg.merchant_gate_model())))
        lines.append(("Merchant gate: on error", "is_merchant = NULL"))
    else:
        lines.append(("Merchant gate", "OFF (TRAFFIC_MERCHANT_GATE_ENABLED not set) - is_merchant=NULL"))

    uses_anthropic = bool(anthropic_model) or cfg.gate_enabled() or cfg.merchant_gate_enabled()
    if uses_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        lines.append(("WARNING", "ANTHROPIC_API_KEY is not set - Anthropic calls will fail"))
    return lines
