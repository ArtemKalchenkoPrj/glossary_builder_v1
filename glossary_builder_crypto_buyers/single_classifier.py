"""Single-message lead classifier adapter.

Thin adapter that runs a single message through the full two-stage pipeline
(extract → judge) and returns a flat JSON-serialisable dict combining fields
from both stages.

Intended as an integration point for external systems that feed one message
at a time (e.g. a Telegram bot or a streaming ingest service). Unlike the
CLI commands ``extract-leads`` and ``judge-leads``, this module has no file
I/O, no JSONL streaming, and no concurrency — the caller owns all of that.

Usage example::

    from glossary_builder.lead_extraction import LeadExtractor, LeadExtractionConfig, load_glossary
    from glossary_builder.llm import LLMClient
    from glossary_builder.single_classifier import classify_single_message

    glossary = load_glossary("data/output/glossary_primary_sense_only.json")
    llm = LLMClient()
    result = classify_single_message(
        text="Ищу PSP для iGaming, нужен KZ и RU, объём от 500k в месяц.",
        glossary=glossary,
        llm=llm,
    )
    # result is a plain dict — write it to a DB, pass it downstream, etc.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from .lead_extraction import LeadExtractor, LeadExtractionConfig, LeadDecision
from .llm import LLMClient
from .loader import Message
from .cli import _JUDGE_SYSTEM, _JUDGE_USER_TEMPLATE
from .rag import RagIndex

import logging

logger = logging.getLogger("__name__")

# ---------------------------------------------------------------------------
# Judge prompt constants
#
# NOTE: these are intentional duplicates of _JUDGE_SYSTEM and
# _JUDGE_USER_TEMPLATE defined in cli.py.  They live here too so that this
# module has no dependency on the CLI layer (importing business logic from a
# CLI module would invert the natural dependency direction).  If the judge
# prompt is ever updated, both copies must be kept in sync.
# ---------------------------------------------------------------------------

# _JUDGE_SYSTEM = """\
# You are an independent reviewer of automatically-extracted sales leads from
# a Russian-language Telegram chat about high-risk payment processing
# (iGaming, casinos, sportsbooks, forex, crypto, adult).
#
# For each lead I show you, give an INDEPENDENT verdict: is this a REAL_LEAD
# or a MISTAKE? Apply this lead definition strictly:
#
# LEAD = a message from a BUYER in this ecosystem (iGaming operator, affiliate,
# merchant, platform) who is ACTIVELY SHOPPING for a payment service AND
# signals a concrete need: a specific vertical, geo, payment method, volume,
# or specific provider being evaluated.
#
# NOT a lead (most common false positives):
#   - 'Кто работал с X?' / 'отзывы' / 'кто работает с X' / 'кто использует X' /
#     'опыт работы с X' — due-diligence / networking. Only a lead if the
#     speaker ALSO states their own concrete need.
#   - Market-pulse questions about the ecosystem ('ситуация с X', 'у всех
#     отлетела SEPA?').
#   - Seller voice: PSP reps pitching, BizDev intros offering services,
#     pricing/margin talk, 'у нас есть X', product/capability pitches even
#     when ending in 'напишите в ЛС'.
#   - Seller inventory pitches: geo+method catalogs with Telegram CTAs.
#   - Affiliate/traffic/media buying: 'ищу трафик', 'ищу объёмы'.
#   - Advertiser/operator searches: 'реклы' / 'реклов' / 'прямой рекл' =
#     advertisers, NOT payment processors. 'Нужны реклы которые принимают гео X'
#     = advertiser request, NOT a payment lead. MISTAKE.
#   - Brand/operator contact searches: 'контакты казино X' / 'contacts for
#     [Brand]' / 'looking for SE/DK/MGA licensed brands/operators' — seeking
#     a gaming brand, NOT a payment service. MISTAKE.
#   - Crypto wallet for personal use: 'анонимный крипто-кошелёк' / 'crypto
#     wallet with AML check' — personal finance tool, NOT payment processing.
#     MISTAKE.
#   - Corporate/legal services: 'компания на Коста-Рике' / 'готовая компания' /
#     'регистрация юрлица' / 'S.A.' — legal/corporate service, NOT payment
#     processing. MISTAKE.
#   - Empty thread-joining: 'мне тоже надо' / 'и мне' / '+1' WITHOUT any
#     payment-specific context in the same message — NOT a lead. MISTAKE.
#   - Hiring / staffing.
#   - Adjacent industries: consulting, M&A, investing, gaming platform/license.
#   - Complaints / blacklist callouts.
#   - Personal/one-off transactions (paying for hotels, visas).
#   - Bot summaries, jokes, conversation.
#
# UNCONDITIONAL LEADS (always REAL_LEAD even if short):
#   - 'у кого есть [thing speaker wants]' / multi-geo possession questions
#   - 'я тоже в поиске' / '+1' / 'присоединюсь к запросу' (thread-joining)
#   - Explicit 'ищу/нужен/нужно' + concrete need
#
# Output strict JSON only.
# """
#
# _JUDGE_USER_TEMPLATE = """\
# LEAD CANDIDATE:
#   message_id: {message_id}
#   @{username}  [{timestamp}]
#   text: {text}
#
# EXTRACTOR's claim:
#   lead_type: {lead_type}
#   intent: {intent}
#   interest_level: {interest_level}
#   vertical: {vertical}
#   geo: {geo}
#   payment_methods_mentioned: {methods}
#   evidence_quote: {evidence_quote}
#   rationale: {rationale}
#
# Return JSON:
# {{
#   "verdict": "REAL_LEAD" | "MISTAKE",
#   "reason": "<one short sentence, English>"
# }}
# """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_judge(decision: LeadDecision, llm: LLMClient, judge_system:str) -> tuple[str, str]:
    """Call the judge LLM for a single lead decision.

    Returns a (verdict, judge_reason) tuple.  verdict is one of
    "REAL_LEAD", "MISTAKE", or "REVIEW" (fallback when the model returns
    something unexpected or the call fails entirely).
    """
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        message_id=decision.message_id or "?",
        username=decision.username or "anon",
        timestamp=decision.timestamp or "?",
        text=(decision.text or "").replace("\n", " ").strip()[:500],
        lead_type=decision.lead_type,
        intent=decision.intent,
        interest_level=decision.interest_level,
        vertical=decision.vertical,
        geo=decision.geo,
        methods=decision.payment_methods_mentioned,
        evidence_quote=(decision.evidence_quote or "")[:200],
        rationale=(decision.rationale or "")[:200],
    )

    try:
        llm.set_stage("lead_judge")
        data, _ = llm.complete_json(judge_system, user_prompt, max_tokens=200)
        verdict = str(data.get("verdict", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        if verdict not in ("REAL_LEAD", "MISTAKE"):
            verdict = "REVIEW"
    except Exception as exc:  # noqa: BLE001
        verdict = "REVIEW"
        reason = f"judge call failed: {exc}"

    return verdict, reason


def _merge(decision: LeadDecision, verdict: Optional[str], judge_reason: Optional[str]) -> dict:
    """Merge extract-stage fields and judge-stage fields into one flat dict.

    Fields from both stages are always present.  For messages that did not
    reach the judge (is_lead=False), verdict and judge_reason are None.
    """
    base = decision.to_dict()
    base["verdict"] = verdict
    base["judge_reason"] = judge_reason
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_single_message(
    text: str,
    glossary: list[dict],
    llm: LLMClient,
    cfg: Optional[LeadExtractionConfig],
    *,
    message_id: Optional[int] = None,
    username: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    context: Optional[list[Message]] = None,
    rag_index : Optional[RagIndex] = None,
    judge_system : Optional[str] = None
) -> dict:
    """Classify one message through the full extract → judge pipeline.

    Runs the same two-stage logic as the ``extract-leads`` + ``judge-leads``
    CLI commands, but for a single message with no file I/O.

    Args:
        text:        The raw message text to classify.
        glossary:    Loaded glossary (output of ``load_glossary()``).
        llm:         Shared LLMClient instance.  The caller is responsible for
                     creating and reusing it so token-usage accounting stays
                     consistent across calls.
        cfg:         Optional extraction config.  Defaults to
                     ``LeadExtractionConfig()`` (standard settings).
        message_id:  Optional identifier carried through to the output dict.
                     Use whatever the upstream system assigns (DB row id,
                     Telegram message id, etc.).  None if unavailable.
        username:    Optional sender handle, passed to the LLM as context.
        timestamp:   Optional message datetime, passed to the LLM as context.
        context:     Optional list of surrounding ``Message`` objects (e.g.
                     the few messages before and after this one in the same
                     chat).  When omitted or None, the classifier runs with no
                     surrounding context.

    Returns:
        A flat, JSON-serialisable dict that is the union of fields from both
        stages:

        From extract-leads:
            message_id, timestamp, username, text, is_lead, confidence,
            lead_type, intent, interest_level, vertical, geo,
            payment_methods_mentioned, evidence_quote, rationale,
            glossary_terms_seen, elapsed_ms

        From judge-leads (None when is_lead=False):
            verdict, judge_reason
    """
    # Build a Message object so we can pass it to the existing extract_one()
    # method which expects the internal Message dataclass, not a raw string.
    message = Message(
        message_id=message_id,
        date=timestamp,
        text=text,
        username=username,
        # Fields below are not used by the lead extractor; set to None.
        group_id=None,
        user_id=None,
        first_name=None,
        last_name=None,
    )

    extractor = LeadExtractor(glossary, llm, cfg, rag_index=rag_index)

    logger.info(
        "[classify] config prompt_version=%s judge_system_len=%d rag=%s",
        cfg.lead_definition[:30].replace("\n", " "),  # перші 30 символів промпту
        len(judge_system or ""),
        "enabled" if rag_index is not None else "disabled",
    )

    # extract_one() returns None only when the message is too short or the
    # author is in the excluded-authors list.  For a single-message adapter
    # we treat that as a definitive not-a-lead rather than raising.
    decision: Optional[LeadDecision] = extractor.extract_one(
        message,
        context=context if context is not None else [],
    )

    if decision is None:
        # Message was skipped by the extractor (too short, excluded author,
        # etc.).  Construct a minimal not-a-lead decision so the caller always
        # gets a consistent dict shape.
        decision = LeadDecision(
            message_id=message_id,
            timestamp=timestamp.isoformat() if timestamp else None,
            username=username,
            text=text,
            is_lead=False,
            confidence=1.0,
            lead_type=None,
            intent="other",
            interest_level="low",
            vertical=[],
            geo=[],
            payment_methods_mentioned=[],
            evidence_quote="",
            rationale="[skipped by extractor: message too short or author excluded]",
            glossary_terms_seen=[],
            elapsed_ms=0.0,
        )
        return _merge(decision, verdict=None, judge_reason=None)

    # Stage 1 complete.  If the extractor decided this is not a lead, skip
    # the judge entirely — it only makes sense to call it on actual leads.
    if not decision.is_lead:
        return _merge(decision, verdict=None, judge_reason=None)

    # Stage 2: independent judge verdict.
    judge_model = os.getenv("JUDGE_MODEL")
    if judge_model:
        llm.model = judge_model

    verdict, judge_reason = _run_judge(decision, llm, judge_system)

    return _merge(decision, verdict=verdict, judge_reason=judge_reason)
