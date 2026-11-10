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
from .llm import LLMClient, AnthropicLLMClient
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


_VALID_UNCERTAINTY_REASONS = frozenset({
    "ambiguous_role", "mixed_psp_and_traffic", "proxy_or_forwarded",
    "crypto_psp_boundary", "advice_form_real_need", "jargon_reinterpretation",
    "implicit_vertical", "multi_intent_contains_psp", "partial_override_context",
})


# ---------------------------------------------------------------------------
# Stage1: micro-LLM binary YES/NO pre-filter (haiku)
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = """\
You are screening messages from a Telegram chat about high-risk payments (PSP, acquiring, iGaming, crypto, forex).

Answer YES if the message is from someone BUYING or SEEKING a payment service.
Answer NO for everything else.

YES — buyer signals:
- Seeking PSP / acquirer / payment method / payment processor / gateway
- "нужен PSP", "ищу эквайринг", "looking for payment processor"
- "ищу решения по [гео]" / "looking for solutions for [country]" / "решение на [гео]" — seeking payment solutions = YES
- "ищу провайдеров по [гео]" / "looking for providers in [country]" = YES
- "у кого есть [метод/гео/решение]?" / "есть ли у кого?" — buyer asking who has what = YES
- "в поиске решений / провайдеров / методов" = YES
- "+1" or "same need" replies to payment requests
- Person asks providers to contact them: "напишите в лс", "постучитесь", "велком в лс", "скиньте оффер", "DM me", "write in PM"
- Person HAS gaming/gambling/forex traffic and needs payment solutions / processing = YES
  Examples: "gamble traffic, looking for processing", "gaming traffic, need payment solutions", "трафик есть, ищу решения"

NO — clearly not a buyer:
- Selling / pitching services without asking for anything ("у нас есть X", "предлагаем X", "работаем с X")
- Reviews / reputation checks ("кто работал с X?", "отзывы о X", "кто использует X?")
- Person SEEKS ad/media traffic (not payment): "looking for gambling traffic", "looking for dating traffic", "ищу рекламный трафик"
- Market questions: "ситуация с X", "у всех пропал X?"
- Hiring, complaints, blacklists, general chat

TRAFFIC RULE — disambiguate carefully:
- "трафик есть / имеем трафик / high traffic" + "looking for solutions / processing / PSP" = YES (they have traffic, need payment processing)
- "looking for gambling/gaming/dating traffic" (wants to buy ad traffic) = NO

Reply with a single word: YES or NO."""


def _run_stage1(text: str, llm: LLMClient) -> bool:
    """Return True if stage1 says YES (potential buyer)."""
    llm.set_stage("buyer_stage1")
    resp = llm.complete(
        STAGE1_SYSTEM,
        text.replace("\n", " ").strip()[:1000],
        max_tokens=5,
        temperature=0.0,
    )
    return resp.text.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Gate v5: certainty gate (haiku) — runs after judge
# ---------------------------------------------------------------------------

_GATE_V5_SYSTEM = """\
You are a quality filter for a PSP (payment service provider) lead classifier.

A previous LLM decided this Telegram message is a REAL_LEAD (someone seeking payment services).
Your job: verify that the message EXPLICITLY mentions payment infrastructure procurement.

=== CERTAIN — the message directly talks about payment services ===
The message contains at least one EXPLICIT payment term:
- "платёжное решение", "платёжная система", "платежка", "платёжку", "платега"
- "PSP", "payment solution", "payment provider", "payment gateway", "payment processing"
- "процессинг", "эквайринг", "acquiring", "gateway"
- "приём платежей", "приём карт", "card processing"
- specific payment methods as the object of search: "ищу/нужен/в поиске" + SBP/СБП, C2C/С2С, P2P/П2П, NSPK/НСПК, QR, Visa, Mastercard, e-wallet, JazzCash, EasyPaisa, MoMo, Boleto, QRIS, UPI, bank transfer, "выплаты по [method]", "прием по [method]"
- "мерчант ищет", "мы мерчант, ищем" (merchant self-ID + search)

=== UNCERTAIN — no explicit payment term, only vague signals ===
The message does NOT contain any explicit payment term from the list above.
Instead it only has vague words like:
- "провайдер" / "провайдеры" alone (could be any kind of provider)
- "решение" / "решения" alone (could be any solution)
- "приёмка" / "прием" alone without "платежей"/"карт"
- "обработка" alone (could be data processing, not payments)
- only traffic/volume terms: "трафик", "объем", "дадим объем"
- only geo without payment context: "Узбекистан живой", "у кого есть [country]"

Also UNCERTAIN:
- Message is about traffic/affiliates, not payments
- Message forwards someone else's need (proxy request)
- Message only asks a survey question without stating own payment need

=== OUTPUT ===
Respond with ONLY a JSON object, no other text:
{"certainty": "CERTAIN" or "UNCERTAIN", "uncertainty_reason": "no_explicit_payment_term" or "traffic_not_payments" or "proxy_request" or "survey_question" or null}

Default to CERTAIN if an explicit payment term is present.
Default to UNCERTAIN if none is present."""

_GATE_V5_USER = """\
MESSAGE:
  @{username}
  text: {text}

JUDGE VERDICT: {verdict}

Respond with ONLY a JSON object:
{{"certainty": "CERTAIN" | "UNCERTAIN", "uncertainty_reason": null | "no_explicit_payment_term" | "traffic_not_payments" | "proxy_request" | "survey_question"}}"""

_GATE_V5_VALID_REASONS = frozenset({
    "no_explicit_payment_term",
    "traffic_not_payments",
    "proxy_request",
    "survey_question",
})


def _run_gate_v5(
    text: str,
    username: str,
    verdict: str,
    llm: LLMClient,
) -> tuple[str, str | None]:
    """Run certainty gate v5. Returns (certainty, reason)."""
    user_prompt = _GATE_V5_USER.format(
        username=username or "anon",
        text=text.replace("\n", " ").strip()[:500],
        verdict=verdict,
    )
    try:
        llm.set_stage("certainty_gate")
        data, _ = llm.complete_json(_GATE_V5_SYSTEM, user_prompt, max_tokens=60)
        certainty = str(data.get("certainty", "CERTAIN")).strip().upper()
        if certainty not in ("CERTAIN", "UNCERTAIN"):
            certainty = "CERTAIN"
        raw = data.get("uncertainty_reason")
        reason = (
            str(raw).strip()
            if isinstance(raw, str) and raw.strip() in _GATE_V5_VALID_REASONS
            else None
        )
        if certainty == "UNCERTAIN" and reason is None:
            reason = "no_explicit_payment_term"
        return certainty, reason
    except Exception as exc:
        logger.warning("certainty gate failed: %s — defaulting to CERTAIN", exc)
        return "CERTAIN", None


def _run_judge(decision: LeadDecision, llm: LLMClient, judge_system: str) -> tuple[str, str, str | None]:
    """Call the judge LLM for a single lead decision.

    Returns a (verdict, judge_reason, uncertainty_reason) tuple.
    verdict is one of "REAL_LEAD", "MISTAKE", or "REVIEW".
    uncertainty_reason is a category ID string or None.
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
        data, _ = llm.complete_json(judge_system, user_prompt, max_tokens=250)
        verdict = str(data.get("verdict", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        raw_uncertainty = data.get("uncertainty_reason")
        uncertainty_reason = (
            str(raw_uncertainty).strip()
            if isinstance(raw_uncertainty, str) and raw_uncertainty.strip() in _VALID_UNCERTAINTY_REASONS
            else None
        )
        if verdict not in ("REAL_LEAD", "MISTAKE"):
            verdict = "REVIEW"
    except Exception as exc:  # noqa: BLE001
        verdict = "REVIEW"
        reason = f"judge call failed: {exc}"
        uncertainty_reason = None

    return verdict, reason, uncertainty_reason


def _merge(
    decision: LeadDecision,
    verdict: Optional[str],
    judge_reason: Optional[str],
    uncertainty_reason: Optional[str] = None,
    *,
    certainty: Optional[str] = None,
    gate_uncertainty_reason: Optional[str] = None,
) -> dict:
    """Merge extract-stage fields and judge-stage fields into one flat dict.

    Fields from both stages are always present.  For messages that did not
    reach the judge (is_lead=False), verdict, judge_reason, and
    uncertainty_reason are None.
    """
    base = decision.to_dict()
    base["verdict"] = verdict
    base["judge_reason"] = judge_reason
    base["uncertainty_reason"] = uncertainty_reason
    base["certainty"] = certainty
    base["gate_uncertainty_reason"] = gate_uncertainty_reason
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
    judge_system : Optional[str] = None,
    group_context_block: str = "",
    bio_block: str = "",
    stage1_llm: Optional[LLMClient] = None,
    gate_llm: Optional[LLMClient] = None,
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
        "[classify] config prompt_version=%s judge_system_len=%d rag=%s stage1=%s gate=%s",
        cfg.lead_definition[:30].replace("\n", " "),
        len(judge_system or ""),
        "enabled" if rag_index is not None else "disabled",
        "enabled" if stage1_llm is not None else "disabled",
        "enabled" if gate_llm is not None else "disabled",
    )

    # extract_one() returns None only when the message is too short or the
    # author is in the excluded-authors list.  For a single-message adapter
    # we treat that as a definitive not-a-lead rather than raising.
    decision: Optional[LeadDecision] = extractor.extract_one(
        message,
        context=context if context is not None else [],
        group_context_block=group_context_block,
        bio_block=bio_block,
    )

    if decision is None:
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

    # If the extractor decided this is not a lead, skip remaining stages.
    if not decision.is_lead:
        return _merge(decision, verdict=None, judge_reason=None)

    # ── Stage1: micro-LLM binary pre-filter (haiku) ──────────────────────
    if stage1_llm is not None:
        stage1_yes = _run_stage1(text, stage1_llm)
        if not stage1_yes:
            logger.info("[buyer] stage1 rejected message_id=%s", message_id)
            decision.is_lead = False
            decision.rationale = "[rejected by stage1 micro-LLM]"
            return _merge(decision, verdict=None, judge_reason=None)
        logger.info("[buyer] stage1 passed message_id=%s", message_id)

    # ── Judge: independent verdict ────────────────────────────────────────
    judge_model = os.getenv("JUDGE_MODEL")
    if judge_model:
        llm.model = judge_model

    verdict, judge_reason, uncertainty_reason = _run_judge(decision, llm, judge_system)

    # ── Gate v5: certainty check (haiku) ──────────────────────────────────
    certainty: str | None = None
    gate_reason: str | None = None
    if gate_llm is not None and verdict == "REAL_LEAD":
        certainty, gate_reason = _run_gate_v5(
            text, username or "", verdict, gate_llm,
        )
        logger.info(
            "[buyer] gate message_id=%s certainty=%s reason=%s",
            message_id, certainty, gate_reason,
        )

    return _merge(
        decision,
        verdict=verdict,
        judge_reason=judge_reason,
        uncertainty_reason=uncertainty_reason,
        certainty=certainty,
        gate_uncertainty_reason=gate_reason,
    )
