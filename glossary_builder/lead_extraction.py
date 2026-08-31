"""Lead-extraction simulator.

Given a corpus of chat messages and the glossary, decide for each message
(or a sampled subset) whether it represents a *buyer lead* — someone in
the chat actively shopping for a payment solution that a high-risk iGaming
PSP might want to pursue. Returns structured decisions with evidence.

Two design choices worth knowing about:

1. **Per-message with surrounding context.** We classify one message at a
   time but include the few preceding messages from the same group as
   context. This is much simpler than full thread reconstruction and gives
   the LLM enough conversational grounding to disambiguate.

2. **Glossary-injected RAG.** For each target message we identify which
   glossary terms appear (substring match on surface forms), then inject
   those definitions into the prompt. This is the whole point of the
   glossary — the agent sees `FTD` and gets the definition right there.

Domain-agnostic: the lead criteria are configurable via ``LeadExtractionConfig``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .fusion import FusionConfig, FusionExtractor
from .llm import LLMClient
from .loader import Message
from .rag import RagConfig, RagIndex, format_few_shot

import logging
logger = logging.getLogger(__name__)

@dataclass
class LeadDecision:
    message_id: int | None
    timestamp: str | None
    username: str | None
    text: str

    is_lead: bool
    confidence: float
    lead_type: str | None         # "seeking_psp" | "seeking_method" | "seeking_acquiring" | "seeking_other" | None
    intent: str                   # "buying" | "comparing" | "researching" | "complaining" | "other"
    interest_level: str           # "high" | "medium" | "low"

    vertical: list[str]
    geo: list[str]
    payment_methods_mentioned: list[str]

    evidence_quote: str
    rationale: str
    glossary_terms_seen: list[str]   # which glossary terms we found in this message

    elapsed_ms: float | None = None  # wall-clock latency of the LLM call for this message
    raw_llm_response: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_llm_response", None)
        return d


@dataclass
class LeadExtractionConfig:
    """How leads are defined and how the extractor behaves."""
    validate_evidence_quote: bool = False
    """Включить/выключить валидатор - есть риск галюцинаций"""

    context_messages_before: int = 5
    """Number of preceding messages from the same group to include as context."""

    context_messages_after: int = 1
    """Number of following messages to include."""

    max_glossary_entries_per_call: int = 12
    """Cap on glossary entries injected per message — keeps prompt size reasonable."""

    skip_min_text_length: int = 20
    """Messages shorter than this are skipped (greetings, emoji-only, '+', etc.)."""

    pre_filter_enabled: bool = True
    """Run the cheap lexical pre-filter (intent_filter.pre_filter) before each
    LLM call. If it returns False (no domain term AND no intent signal), the
    message is decided is_lead=False locally with no model call. Recall-tuned
    on the confirmed-lead audit (~99% recall, OR logic). Set False to send
    every message to the LLM."""

    pre_filter_db_path: str = "data/output/glossary.db"
    """SQLite DB holding the intent_glossary table the pre-filter reads."""

    stage1_enabled: bool = True
    """Run the cheap Stage 1 micro LLM call after the lexical pre-filter and
    before the full v5-short prompt. It's a single-word YES/NO buyer-intent
    screen; a NO short-circuits to is_lead=False with no expensive call. Set
    False to send every pre-filter survivor straight to the full prompt."""

    # The role each criterion plays in the agent's lead definition.
    # Domain-specific bits live here so they can be replaced wholesale for
    # a different vertical.
    #
    # IMPORTANT: this definition was tightened after an independent audit
    # found 60% precision on the looser earlier version. The change is to
    # require BOTH (a) explicit buyer-intent language AND (b) a concrete
    # need specification, AND to list common false-positive patterns
    # explicitly as NOT-LEAD.
    lead_definition: str = (
        "A LEAD is a message from a BUYER in the high-risk payments "
        "ecosystem — an iGaming/casino operator, affiliate, merchant, or "
        "platform — who is actively shopping for a payment service (PSP, "
        "acquirer, payment method, gateway, orchestrator, KYC provider, "
        "etc.) AND signals a concrete need.\n"
        "\n"
        "The message is a LEAD if it has BOTH a buyer-intent signal AND a "
        "concrete need spec. Each side can be expressed several ways:\n"
        "\n"
        "  (a) BUYER-INTENT SIGNAL — any of:\n"
        "      * Explicit verb: 'ищу', 'ищем', 'нужен', 'нужно', 'требуется',\n"
        "        'рассматриваем', 'буду рад предложениям', 'поделитесь "
        "предложениями',\n"
        "        'looking for', 'we need', 'we are seeking', 'send offers'.\n"
        "      * BUYER-possession question form: 'у кого есть [thing I want]',\n"
        "        'у кого можно взять', 'кто может дать', 'есть ли у кого',\n"
        "        'у кого что есть'. CRITICAL: the speaker must be the one\n"
        "        WANTING the thing, not the one HAVING it. 'У нас есть X' is\n"
        "        SELLER voice and never a lead. 'У кого свой Y есть?' is\n"
        "        SELLER hunting for merchants — NOT a lead either.\n"
        "      * Solicitation form: a need + invitation to contact —\n"
        "        the message names a concrete geo/method/vertical AND ends with\n"
        "        'в лс', 'пишите в личку', 'напишите', 'поделитесь контактами',\n"
        "        'share offers'. Pure listings like 'первичка, вторичка KZ, лс'\n"
        "        ARE leads under this clause.\n"
        "      * Thread-joining / +1 form (UNCONDITIONAL LEAD — do not require\n"
        "        anything else in the message itself):\n"
        "          'присоединюсь к запросу', '+ к запросу', '+1', 'плюс 1',\n"
        "          'мне тоже', 'тоже интересует', 'я тоже в поиске',\n"
        "          'в поиске данного решения', 'в поиске X', 'тоже нужно'.\n"
        "        These are buyers joining an active sourcing request. Treat\n"
        "        as LEAD even if the message has no geo/method of its own —\n"
        "        the surrounding thread carries that.\n"
        "      * 'у кого есть [thing]' / 'у кого есть [GEO/method]' is a\n"
        "        LEAD even when nothing else is said and even when only\n"
        "        a country list is given (e.g. 'у кого есть решение на\n"
        "        Израиль и Египет', 'у кого свой e-com есть в KG/KZ' when\n"
        "        the speaker is a buyer looking to integrate). Multi-geo\n"
        "        listings count.\n"
        "  (b) CONCRETE NEED specification — at least one of:\n"
        "      a specific vertical (iGaming/casino/forex/crypto/adult/etc.),\n"
        "      a specific geography (country/region),\n"
        "      a specific payment method (Blik/iDeal/USDT/etc.),\n"
        "      a volume / FTD / STD / approve-rate requirement,\n"
        "      a specific provider whose offering they want to use (not just\n"
        "      check reputation).\n"
        "\n"
        "EXPLICITLY NOT a lead (the system was over-flagging these):\n"
        "  - 'Кто работал с X?' / 'отзывы' / 'кто знает про X' / 'кто-то "
        "    оперирует на Y?' / 'опыт работы с X' / 'кто работает с X' / "
        "    'кто-то работает по [GEO]' / 'кто работает на [GEO]' / "
        "    'есть у кого-то опыт с X' / 'кто использует X' / 'кто провайдит' "
        "    — these are ALWAYS due-diligence / reputation checks / "
        "    networking, NOT leads, even when X is a specific PSP brand, "
        "    a list of PSPs, or a country/geo. They are also NOT leads when "
        "    the speaker adds 'поделитесь' / 'отпишите' / 'напишите' as a "
        "    call-to-action. The ONLY exception is when the SAME message "
        "    explicitly states the author's own procurement need with verbs "
        "    like 'ищу', 'ищем', 'нужен', 'нужны', 'интересует' in addition "
        "    to the networking question.\n"
        "    CRITICAL DISAMBIGUATION: 'у кого есть X' (the speaker WANTS the "
        "    thing — LEAD) is structurally different from 'кто работает с X' "
        "    or 'кто использует X' (the speaker wants to FIND PEOPLE — NOT "
        "    a lead). The grammatical clue: 'у кого есть [object I want to "
        "    receive]' vs 'кто [verb of using/working with] [thing already "
        "    in market]'. Treat them differently.\n"
        "  - MARKET-PULSE / SITUATIONAL questions about the broader market: "
        "    'ситуация с X', 'у всех ли отлетела SEPA?', 'как обстоят дела "
        "    с FTD A/G Pay?', 'поделитесь, что используете для тестирования' "
        "    — these are research about the ecosystem, not procurement.\n"
        "  - Complaints / blacklist callouts / dispute outreach about past "
        "    providers (recovery, not buying).\n"
        "  - Seller-side messages: PSP reps pitching, BizDev intros offering "
        "    services, pricing/margin discussions ('торгуемся на 6.5%'), "
        "    'у нас есть X, кому надо', 'наши возможности', "
        "    'мерчант для вас', '[Brand] — больше никаких задержек', "
        "    'cashier service', 'white label', product/capability pitches, "
        "    even when they end with 'напишите в ЛС'.\n"
        "  - SELLER INVENTORY PITCHES — messages that look like ads with ANY of:\n"
        "    * bullet/comma list of capabilities under phrases like 'Что у "
        "      нас есть:', 'Наши преимущества:', 'Мы предлагаем', "
        "      '[Brand] — [capability], [capability]'\n"
        "    * a geo / method / vertical CATALOG (e.g. listing many "
        "      countries and product features) followed by 'кто хочет "
        "      работать', 'мы поможем', 'обращайтесь', 'свяжитесь с нами'\n"
        "    * possessing-inventory phrasing where the speaker offers: "
        "      '[method/geo] есть :)', '[method/geo] доступны', 'есть "
        "      [method/geo], напишу в лс' — speaker has the thing and is "
        "      offering it; this is NOT 'у кого есть' buyer-question form\n"
        "    * a Telegram CTA (@username, 'Telegram: @x', 'пишите нам', "
        "      'напишите мне для информации') combined with a product/geo "
        "      description.\n"
        "    These are advertising messages from PSPs, not buyer leads, "
        "    regardless of whether they list geos or methods.\n"
        "  - Sell-side aggregator messages: 'у нас как у агрегатора кончился "
        "    лишний трафик / трейдеры', 'пристроить трафик к вашим решениям', "
        "    offering volume — these are sellers looking for buyers, NOT buyers.\n"
        "  - Affiliate / traffic / media buying requests: 'находимся в поиске "
        "    трафика', 'ищу объёмы гембл/беттинг', 'у кого свой e-com есть' "
        "    (looking for merchants to integrate with, not payment service). "
        "    These are commercial offers in different markets.\n"
        "  - Single-PSP relationship continuation: 'ищу представителей [Brand]', "
        "    'хочу обсудить возобновление сотрудничества с [Brand]' — "
        "    re-engaging with a known counterparty, not open shopping.\n"
        "  - Networking / contact-hunt asks without a procurement signal: "
        "    'свяжите с X', 'контакты представителей Y', 'дайте контакт Z' "
        "    when not paired with a stated need.\n"
        "  - Hiring / staffing requests ('ищет в штат', 'ищу работу', "
        "    job postings).\n"
        "  - Advisory / reply voice — author is advising someone else, not "
        "    expressing their own need.\n"
        "  - Market research / strategy discussion ('п2п Индия перенасыщен?', "
        "    'какие сервисы используют...').\n"
        "  - Bot-generated daily summaries, jokes, congratulations, general "
        "    conversation, technical questions, conference invites.\n"
        "  - One-off personal transactions: 'Нужно провести транзакцию через "
        "    X, оплатить отель / визу / билеты' — individual consumer use, "
        "    not merchant payment-service procurement.\n"
        "  - Adjacent-industry asks that aren't payment-service procurement:\n"
        "    buying consulting ('хочу купить вашу консультацию'),\n"
        "    M&A / acquisition ('ищу на выкуп RGS', 'покупаю казино'),\n"
        "    fund-raising ('ищем соинвестора', 'рассмотрим инвестиции'),\n"
        "    affiliate / traffic / media buying ('ищу рекламодателей',\n"
        "    'ищу медиа', 'ищу трафик'), gaming platform/license deals\n"
        "    (unless paired with a payment-service need)."
    )

    excluded_authors: set[str] = field(default_factory=lambda: {
        # Bots and chat infrastructure — skip wholesale.
        "ChatLogixBot",
    })


# -------- glossary indexing -----------------------------------------------


class GlossaryIndex:
    """Fast substring lookup from surface form -> glossary entry."""

    def __init__(self, glossary: list[dict]):
        # Build a list of (surface_lower, entry) pairs sorted longest-first
        # so multi-word terms match before shorter substrings.
        self._entries: list[tuple[str, dict]] = []
        seen: set[tuple[str, str]] = set()
        for e in glossary:
            forms = e.get("surface_forms") or [e["term"]]
            if not forms:
                forms = [e["term"]]
            for form in forms:
                key = form.lower().strip()
                if not key or len(key) < 2:
                    continue
                pair_key = (key, e["term"])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                self._entries.append((key, e))
        self._entries.sort(key=lambda kv: len(kv[0]), reverse=True)

    def find_in(self, text: str, limit: int = 12) -> list[dict]:
        """Return up to ``limit`` glossary entries that appear in the text.

        Returns entries in the order they appear, deduped by term.
        """
        if not text:
            return []
        lower = text.lower()
        seen_terms: set[str] = set()
        hits: list[dict] = []
        for key, entry in self._entries:
            if entry["term"] in seen_terms:
                continue
            if len(key) <= 3:
                matched = bool(re.search(rf"(?<!\w){re.escape(key)}(?!\w)", lower))
            else:
                matched = key in lower
            if not matched:
                continue
            seen_terms.add(entry["term"])
            hits.append(entry)
            if len(hits) >= limit:
                break
        return hits


def load_glossary(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -------- prompt assembly -------------------------------------------------


_SYSTEM_TEMPLATE = """\
You are a lead-qualification agent that reads messages from a Russian-language
Telegram chat about high-risk payment processing (iGaming, casinos,
sportsbooks, forex, crypto, adult, nutra, dating, trading, MLM,
subscription services, call centers). Your job is to decide whether a target
message represents a sales lead and to extract structured information about it.

LEAD DEFINITION
{lead_definition}

OUTPUT
For each decision, return strict JSON with this shape:
{{
  "is_lead": <bool>,
  "confidence": <float 0..1>,
  "lead_type": "seeking_psp" | "seeking_method" | "seeking_acquiring" | "seeking_other" | null,
  "intent": "buying" | "comparing" | "researching" | "complaining" | "other",
  "interest_level": "high" | "medium" | "low",
  "vertical": ["igaming" | "casino" | "sportsbook" | "forex" | "crypto" | "adult" | "nutra" | "dating" | "trading" | "mlm" | "subscription" | "call_centers" | "resale" | "other" | "unknown", ...],
  "geo": ["<country code or short name>", ...],
  "payment_methods_mentioned": ["<method name>", ...],
  "evidence_quote": "<short verbatim quote from the target message, <=160 chars>",
  "rationale": "<one or two sentences explaining the call, in English>"
}}

RULES
- Quote evidence VERBATIM from the target message — do not paraphrase.
- If is_lead is false, set lead_type=null, interest_level="low", confidence 0.6-1.0.
- ``geo`` should hold short tokens ('UK', 'DE', 'LATAM') or country names; empty list if none.
- ``vertical`` guidance:
  - "nutra" — supplements, weight loss, peptides, health products.
  - "dating" — dating/matchmaking services.
  - "trading" — binary options, CFD, financial trading platforms (distinct from "forex" which covers currency exchange/FX brokers).
  - "mlm" — multi-level marketing / network marketing.
  - "subscription" — recurring-billing apps/services (not gambling/dating).
  - "call_centers" — outbound/inbound phone sales operations, telemarketing teams, or traffic generated by them. Often abbreviated as "кц", "сс", "колл-центр", "трафик кц".
  - Use "other" only when none of the above or the standard verticals apply.
  - Use "unknown" only when the message gives NO vertical signal at all — do not guess a vertical that isn't stated or implied.
- CRITICAL VERTICAL RESOLUTION: If the target message contains "кц", "сс", "колл-центр", or mentions call center traffic, you MUST classify the vertical as "call_centers". Do NOT classify it as "casino" or "igaming" just because metrics like "деп", "депозит", or "объем" are present.
- Be skeptical of profile intros — they look like leads but usually aren't unless the speaker explicitly states what they're shopping for.
- BE STRICT. The default is is_lead=false. Set is_lead=true only when BOTH conditions (a) and (b) from the LEAD DEFINITION are present in the TARGET MESSAGE itself (not just the surrounding context).
- A request for REVIEWS / FEEDBACK / "anyone worked with X" is NOT a lead unless the asker also states their own need.
- A request for CONTACTS / introductions is NOT a lead unless paired with the asker's stated need.
- A complaint about a past provider is NOT a lead.
- Use the surrounding context to interpret SHORT target messages and to detect SELLER VOICE (if recent context shows the same author pitching, this message is unlikely to be a buyer signal).
- The provided glossary entries are AUTHORITATIVE for jargon: trust their definitions when the message uses those terms.
"""


_USER_TEMPLATE = """\
GLOSSARY (terms relevant to the target message):
{glossary_block}

{few_shot_block}\
CONTEXT (preceding and following messages from the same group):
{context_block}

TARGET MESSAGE:
  msg_id={message_id} @{username} [{timestamp}]
  {text}

Return your JSON decision for the TARGET MESSAGE.
"""


# Stage 1 micro-filter: a single cheap YES/NO classification that runs between
# the lexical pre-filter and the full v5-short prompt. It catches the obvious
# non-buyers (sellers, reviewers, market chatter) with one short completion so
# we never pay for the big structured call on them.
_STAGE1_SYSTEM = """\
You are screening messages from a Telegram chat about high-risk payments (PSP, acquiring, iGaming, crypto, forex).

Answer YES if the message is from someone BUYING or SEEKING a payment service.
Answer NO for everything else.

YES examples:
- Asking who has/offers X payment solution
- Looking for PSP/acquirer/payment method
- "+1" or "same need" replies to payment requests
- "нужен PSP", "ищу эквайринг", "looking for payment processor"
- "у кого-то есть X", "есть ли среди нас X" — possession questions
- "рассматриваю офферы", "рассматриваем варианты" — evaluating options
- "какие есть решения на рынке" — asking for solutions with implied need
- "присоединяюсь к запросу" or joining an existing request

NO examples:
- Selling or pitching services
- Reviews/reputation checks ("кто работал с X")
- Market questions without personal need ("как дела у X?")
- Hiring, complaints, general chat

Reply with a single word: YES or NO."""

_STAGE1_USER_TEMPLATE = """\
MESSAGE:
{text}

YES or NO?"""


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
        return "  (no prior context)"
    lines = []
    for m in context:
        text = (m.text or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:197] + "..."
        ts = m.date.isoformat() if m.date else "?"
        uname = m.username or "anon"
        mid = m.message_id if m.message_id is not None else "?"
        lines.append(f"  [msg_id={mid} @{uname} {ts}] {text}")
    return "\n".join(lines)


# -------- extractor -------------------------------------------------------


class LeadExtractor:
    """Per-message lead classifier with surrounding-context awareness."""

    def __init__(
        self,
        glossary: list[dict],
        llm: LLMClient,
        config=None,
        fusion_config=None,
        rag_config=None,
        rag_index=None
    ):
        self.glossary_index = GlossaryIndex(glossary)
        self.llm = llm
        self.config = config or LeadExtractionConfig()
        self.fusion = FusionExtractor(fusion_config) if fusion_config else None
        if rag_index is not None:
            self.rag_index = rag_index
        elif rag_config is not None:
            self.rag_index = RagIndex(rag_config)
        else:
            self.rag_index = None

        # Pre-filter wiring. Resolved once: import the function and warm the
        # term dictionary so a missing/empty DB fails loudly here rather than
        # on every message. On failure we disable the pre-filter and fall back
        # to sending every message to the LLM.
        self.prefiltered_count = 0
        self.stage1_filtered_count = 0
        self._prefilter_lock = threading.Lock()
        self._pre_filter = None
        if self.config.pre_filter_enabled:
            try:
                from intent_filter import pre_filter, load_glossary_terms
                terms = load_glossary_terms(self.config.pre_filter_db_path)
                if not terms:
                    raise ValueError("intent_glossary is empty")
                self._pre_filter = pre_filter
            except Exception as exc:  # noqa: BLE001 — never block extraction on this
                from rich.console import Console as _RC
                _RC().print(
                    f"[yellow]Pre-filter disabled ({exc}). "
                    f"All messages will be sent to the LLM."
                )

    def extract_one(
        self,
        target: Message,
        context: list[Message],
    ) -> LeadDecision | None:
        """Classify one message. Returns None if the message was skipped."""

        cfg = self.config
        text = (target.text or "").strip()
        if not text or len(text) < cfg.skip_min_text_length:
            return None
        if target.username and target.username in cfg.excluded_authors:
            return None

        glossary_hits = self.glossary_index.find_in(text, limit=cfg.max_glossary_entries_per_call)

        # Cheap lexical pre-filter: skip the LLM entirely when the message has
        # neither a domain term nor a buyer-intent signal. Decided locally as
        # not-a-lead. ~67% of a real corpus is dropped here at zero LLM cost.
        if self._pre_filter is not None and not self._pre_filter(text, cfg.pre_filter_db_path):
            with self._prefilter_lock:
                self.prefiltered_count += 1
            return LeadDecision(
                message_id=target.message_id,
                timestamp=target.date.isoformat() if target.date else None,
                username=target.username,
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
                rationale="[pre-filtered: no domain term or intent signal — LLM call skipped]",
                glossary_terms_seen=[e["term"] for e in glossary_hits],
                elapsed_ms=0.0,
                raw_llm_response=None,
            )

        # Stage 1: a single cheap YES/NO buyer-intent screen. Cheaper than the
        # full structured prompt, so a NO here lets us skip it entirely and
        # decide is_lead=False locally. A YES (or a Stage 1 failure) falls
        # through to the full v5-short prompt below.
        if cfg.stage1_enabled:
            try:
                self.llm.set_stage("lead_stage1_micro")
                _s1 = self.llm.complete(
                    _STAGE1_SYSTEM,
                    _STAGE1_USER_TEMPLATE.format(text=text),
                    max_tokens=10,
                    temperature=0.0,
                )
                is_yes = "yes" in _s1.text.strip().lower()
            except Exception:  # noqa: BLE001 — never let the screen block extraction
                is_yes = True  # fail open: defer to the full prompt
            if not is_yes:
                with self._prefilter_lock:
                    self.stage1_filtered_count += 1
                return LeadDecision(
                    message_id=target.message_id,
                    timestamp=target.date.isoformat() if target.date else None,
                    username=target.username,
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
                    rationale="[stage-1 filtered: not a buyer-intent message — full LLM call skipped]",
                    glossary_terms_seen=[e["term"] for e in glossary_hits],
                    elapsed_ms=0.0,
                    raw_llm_response=None,
                )

        system = _SYSTEM_TEMPLATE.replace(
            "{lead_definition}", cfg.lead_definition
        )
        # RAG few-shot injection
        few_shot_block = ""
        if self.rag_index is not None:
            try:
                examples = self.rag_index.query(text)
                few_shot_block = format_few_shot(examples)

                # Логи
                snippet = text.replace("\n", " ").strip()[:100]
                logger.info(f"[rag] target: {snippet}")
                for i, ex in enumerate(examples["leads"], 1):
                    logger.info(f"[rag] lead_{i}: {ex.replace(chr(10), ' ').strip()[:100]}")
                for i, ex in enumerate(examples["not_leads"], 1):
                    logger.info(f"[rag] not_lead_{i}: {ex.replace(chr(10), ' ').strip()[:100]}")

            except Exception as e:
                logger.warning(f"[rag] query failed: {e}")

        user = _USER_TEMPLATE.format(
            glossary_block=_format_glossary_block(glossary_hits),
            few_shot_block=few_shot_block,
            context_block=_format_context(context),
            message_id=target.message_id,
            username=target.username or "anon",
            timestamp=target.date.isoformat() if target.date else "?",
            text=text,
        )

        _t0 = time.perf_counter()
        try:
            if self.fusion is not None:
                data, raw = self.fusion.run(system, user)
                if isinstance(data, list):
                    data = data[0] if data else {}
            else:
                self.llm.set_stage("lead_full_prompt")
                resp_data, resp = self.llm.complete_json(system, user, max_tokens=500)
                data, raw = resp_data, resp.text
        except Exception as exc:
            return LeadDecision(
                message_id=target.message_id,
                timestamp=target.date.isoformat() if target.date else None,
                username=target.username,
                text=text,
                is_lead=False,
                confidence=0.0,
                lead_type=None,
                intent="other",
                interest_level="low",
                vertical=[],
                geo=[],
                payment_methods_mentioned=[],
                evidence_quote="",
                rationale=f"[extraction failed: {exc}]",
                glossary_terms_seen=[e["term"] for e in glossary_hits],
                elapsed_ms=(time.perf_counter() - _t0) * 1000.0,
                raw_llm_response=None,
            )

        decision = _build_decision(target, text, data, glossary_hits, raw, cfg=self.config)
        decision.elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        return decision

    def extract_batch(
        self,
        messages: list[Message],
        *,
        sample_message_ids: Iterable[int] | None = None,
        concurrency: int = 4,
        chunk_size: int = 200,
        progress_desc: str = "Leads",
        stream_jsonl_path: str | Path | None = None,
        resume: bool = True,
    ) -> list[LeadDecision]:
        """Classify a batch of messages with bounded memory.

        Processes messages in chunks of ``chunk_size`` so the executor only
        ever holds that many futures in flight at once. Optionally streams
        each completed decision to ``stream_jsonl_path`` as it's produced, so
        a crashed run loses at most the in-flight chunk and can resume.

        Args:
            messages: the full ordered corpus (used to build context windows).
            sample_message_ids: if given, only classify messages whose ids
                are in this set. If None, classify every message.
            concurrency: how many LLM calls in flight per chunk.
            chunk_size: max futures held by the executor at once. Lower =
                less memory, slightly more dispatch overhead. 200 is a sane
                default.
            progress_desc: tqdm label.
            stream_jsonl_path: if given, write each decision as a JSONL line
                immediately on completion. Enables resume.
            resume: if True and ``stream_jsonl_path`` exists, skip message_ids
                already present in that file. Off if False.
        """
        cfg = self.config

        # Build target list once, drop messages we'll definitely skip.
        targets: list[tuple[int, Message]] = []
        for i, m in enumerate(messages):
            if sample_message_ids is not None and m.message_id not in sample_message_ids:
                continue
            text = (m.text or "").strip()
            if not text or len(text) < cfg.skip_min_text_length:
                continue
            if m.username and m.username in cfg.excluded_authors:
                continue
            targets.append((i, m))

        # Resume: which message_ids are already in the JSONL output.
        already_done: set[int] = set()
        if stream_jsonl_path is not None and resume:
            p = Path(stream_jsonl_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        mid = d.get("message_id")
                        if isinstance(mid, int):
                            already_done.add(mid)
                    except json.JSONDecodeError:
                        continue
                if already_done:
                    targets = [t for t in targets if t[1].message_id not in already_done]

        # Open append-mode JSONL stream (so resume keeps prior lines).
        stream_file = None
        if stream_jsonl_path is not None:
            p = Path(stream_jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            stream_file = p.open("a", encoding="utf-8")
            if already_done:
                # User-visible hint about resume.
                from rich.console import Console as _RC
                _RC().print(
                    f"[yellow]Resuming: skipping {len(already_done)} message_ids "
                    f"already in {stream_jsonl_path}"
                )

        def _do(i_msg: tuple[int, Message]) -> LeadDecision | None:
            i, m = i_msg
            ctx_before = messages[max(0, i - cfg.context_messages_before):i]
            ctx_after = messages[i + 1:i + 1 + cfg.context_messages_after]
            ctx = list(ctx_before) + list(ctx_after)
            d = self.extract_one(m, ctx)
            if d is not None:
                # Drop the raw response immediately — we only kept it for
                # ad-hoc debugging and at scale it explodes memory.
                d.raw_llm_response = None
            return d

        decisions: list[LeadDecision] = []
        progress = tqdm(total=len(targets), desc=progress_desc)
        try:
            for chunk_start in range(0, len(targets), chunk_size):
                chunk = targets[chunk_start:chunk_start + chunk_size]
                # Fresh executor per chunk -> futures are released between chunks.
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(_do, im) for im in chunk]
                    for fut in as_completed(futures):
                        progress.update(1)
                        try:
                            d = fut.result()
                        except Exception:
                            continue
                        if d is None:
                            continue
                        decisions.append(d)
                        if stream_file is not None:
                            stream_file.write(
                                json.dumps(d.to_dict(), ensure_ascii=False) + "\n"
                            )
                            stream_file.flush()
                # `futures` and `chunk` go out of scope here; GC reclaims them.
        finally:
            progress.close()
            if stream_file is not None:
                stream_file.close()

        # Preserve corpus order.
        order = {m.message_id: i for i, m in enumerate(messages) if m.message_id is not None}
        decisions.sort(key=lambda d: order.get(d.message_id, 1_000_000))
        return decisions


def _build_decision(
        target: Message,
        text: str,
        data: dict,
        glossary_hits: list[dict],
        raw: str,
        cfg: LeadExtractionConfig | None = None,
    ) -> LeadDecision:

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

    is_lead = bool(data.get("is_lead", False))
    lead_type = data.get("lead_type")
    if isinstance(lead_type, str):
        lead_type = lead_type.strip() or None
    elif lead_type is not None:
        lead_type = str(lead_type)

    evidence_quote = str(data.get("evidence_quote", "")).strip()
    rationale = str(data.get("rationale", "")).strip()

    # Code-level safeguard: an LLM that hallucinated evidence by pulling text
    # from the surrounding context (not the target message) is unreliable.
    # If the quoted evidence isn't substantively present in the message,
    # downgrade the decision. We allow trivial slack (case, whitespace,
    # punctuation) but require a meaningful chunk of the quote to be in
    # the message body itself.
    if is_lead and evidence_quote:
        if cfg.validate_evidence_quote and not _quote_supported_by_text(evidence_quote, text):
            is_lead = False
            rationale = (
                "[downgraded by evidence-quote validator: quoted evidence "
                "not found in target message text] " + rationale
            )

    return LeadDecision(
        message_id=target.message_id,
        timestamp=target.date.isoformat() if target.date else None,
        username=target.username,
        text=text,
        is_lead=is_lead,
        confidence=_bound_conf(data.get("confidence", 0.0)),
        lead_type=lead_type if is_lead else None,
        intent=str(data.get("intent", "other")).strip().lower() or "other",
        interest_level=str(data.get("interest_level", "low")).strip().lower() or "low",
        vertical=_list_str(data.get("vertical")),
        geo=_list_str(data.get("geo")),
        payment_methods_mentioned=_list_str(data.get("payment_methods_mentioned")),
        evidence_quote=evidence_quote,
        rationale=rationale,
        glossary_terms_seen=[e["term"] for e in glossary_hits],
        raw_llm_response=raw,
    )


def _quote_supported_by_text(quote: str, text: str, *, min_overlap_chars: int = 25) -> bool:
    """Return True iff the quoted evidence is meaningfully present in text.

    We don't require the quote to be a literal substring (the LLM often
    elides punctuation, normalises whitespace, etc.). Instead we look for
    a run of consecutive characters from the quote in the text.

    For short quotes (< min_overlap_chars), require an exact-normalised
    substring match. For longer quotes, require that a window of
    ``min_overlap_chars`` consecutive characters appears in the text.
    """
    if not quote or not text:
        return False
    nt = _norm_for_quote_match(text)
    nq = _norm_for_quote_match(quote)
    if not nq or not nt:
        return False
    # Trivial case: full quote in text.
    if nq in nt:
        return True
    # Short quote that didn't match → reject.
    if len(nq) < min_overlap_chars:
        return False
    # Longer quote: try sliding a window of min_overlap_chars.
    for i in range(0, len(nq) - min_overlap_chars + 1):
        chunk = nq[i:i + min_overlap_chars]
        if chunk in nt:
            return True
    return False


def _norm_for_quote_match(s: str) -> str:
    """Lowercase + collapse non-alphanumeric runs for quote-substring checks."""
    import re as _re
    return _re.sub(r"[^a-zA-Z0-9\u0400-\u04FF]+", " ", s.lower()).strip()


# -------- evaluation --------------------------------------------------------


@dataclass
class EvalReport:
    n_total: int
    n_predicted_lead: int
    n_actual_lead: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_predicted_lead": self.n_predicted_lead,
            "n_actual_lead": self.n_actual_lead,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def evaluate(
    decisions: list[LeadDecision],
    ground_truth: dict[int, bool],
) -> EvalReport:
    """Compute precision/recall/F1 against a ground-truth label dict.

    ``ground_truth`` maps message_id -> bool (is_lead). Only messages
    present in both ``decisions`` and ``ground_truth`` count.
    """
    decision_by_id = {d.message_id: d for d in decisions if d.message_id is not None}
    common = [mid for mid in ground_truth if mid in decision_by_id]
    tp = fp = fn = tn = 0
    for mid in common:
        pred = decision_by_id[mid].is_lead
        actual = bool(ground_truth[mid])
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return EvalReport(
        n_total=len(common),
        n_predicted_lead=sum(1 for mid in common if decision_by_id[mid].is_lead),
        n_actual_lead=sum(1 for mid in common if ground_truth[mid]),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )
