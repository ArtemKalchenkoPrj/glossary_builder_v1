"""Stage 3 — in-corpus LLM inference (Tier 1).

For each candidate term, hand the LLM a domain brief, a batch of example
messages where the term appears, and any inline self-definitions we already
mined. Ask for a structured judgement: definition, senses, relevance, confidence.

This is the workhorse stage. ~70% of terms get a high-confidence answer here
without needing external research.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Iterable

from .candidates import Candidate
from .config import InferenceConfig
from .llm import LLMClient
from .self_definitions import SelfDefinition


@dataclass
class Sense:
    definition: str
    context_clues: list[str] = field(default_factory=list)
    example_message_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DraftEntry:
    term: str
    relevant: bool
    senses: list[Sense]
    confidence: float
    reasoning: str
    source_stage: str = "tier1_inference"
    raw_response: str | None = None

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "relevant": self.relevant,
            "senses": [s.to_dict() for s in self.senses],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "source_stage": self.source_stage,
        }


_SYSTEM = """\
You are a domain analyst building a glossary for an automated agent that scans
chat archives looking for sales leads in the payments / iGaming industry.

The agent will use the glossary to understand jargon, acronyms, brand names
and slang it encounters. Bad glossary entries cause bad classifications. Be
strict: if the evidence is weak, say so and lower your confidence.

For each term you analyse, you will:
  1. Decide whether the term is *domain-relevant*. Relevant means: the term
     carries SPECIFIC meaning in payments, iGaming, sales / business
     development, finance, fraud or risk that a non-industry person would
     NOT immediately understand. NOT relevant:
        - generic English words ("Head", "Manager", "Business", "Pay")
          unless they are part of a specific industry compound (e.g. "MoR"
          for Merchant of Record is relevant, "Manager" alone is not)
        - user handles / nicknames (anything that looks like a Telegram
          username, e.g. "whos_fv", "user_123")
        - chat-system / bot artefacts (e.g. "dailysummary", "supersummary_*")
        - command fragments (e.g. "start" from "/start")
  2. Provide one or more *senses* — distinct meanings that disambiguate by
     surrounding context. Most terms have exactly one sense.
  3. For each sense, list a few short *context clues* — words that, when they
     co-occur with the term, signal this particular meaning.
  4. Reference example messages by their ``message_id`` to support your senses.
  5. Score your overall *confidence* in the entry from 0.0 to 1.0, where
     1.0 = certain (multiple examples all clearly support the definition) and
     0.3 = weak (examples are ambiguous or definition is generic).

Output strict JSON. No prose outside the JSON block. ``confidence`` and
``reasoning`` MUST be top-level keys in the JSON, not nested inside senses.\
"""


_USER_TEMPLATE = """\
DOMAIN CONTEXT:
{brief}

TERM: {term}
Surface forms seen in corpus: {surfaces}
Total occurrences in sample: {freq}
Token kind (rule-based hint): {kind}

INLINE SELF-DEFINITIONS FOUND IN CORPUS:
(These are *candidate* hints — some are real, some are false positives.
Use them only if they actually fit the example messages.)
{self_defs}

EXAMPLE MESSAGES (with message_id, sender, text):
{examples}

Return JSON matching this schema exactly:
{{
  "term": "{term}",
  "relevant": <bool>,           // false if noise / off-topic / common word
  "senses": [
    {{
      "definition": "<concise English definition, <=160 chars>",
      "context_clues": ["<word>", "<word>", ...],
      "example_message_ids": [<int>, ...]   // ids supporting this sense
    }}
  ],
  "confidence": <float 0..1>,
  "reasoning": "<one or two sentences, English>"
}}

If the term is not domain-relevant, return relevant=false, an empty senses
array, and explain briefly in reasoning.\
"""


def _format_examples(examples: list[dict], limit: int) -> str:
    lines = []
    for ex in examples[:limit]:
        mid = ex.get("message_id", "?")
        user = ex.get("username") or "anon"
        text = (ex.get("text") or "").replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:277] + "..."
        lines.append(f"  - msg_id={mid} @{user}: {text}")
    return "\n".join(lines) if lines else "  (none)"


def _format_self_defs(self_defs: list[SelfDefinition] | None) -> str:
    if not self_defs:
        return "  (none found)"
    lines = []
    for sd in self_defs:
        snippet = sd.message_text.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        lines.append(
            f"  - [{sd.pattern}] definition={sd.definition!r}\n"
            f"      in: msg_id={sd.message_id} {snippet!r}"
        )
    return "\n".join(lines)


def infer_definition(
    candidate: Candidate,
    self_defs: list[SelfDefinition] | None,
    domain_brief: str,
    llm: LLMClient,
    config: InferenceConfig | None = None,
) -> DraftEntry:
    """Run Tier-1 inference for a single candidate."""
    cfg = config or InferenceConfig()
    user = _USER_TEMPLATE.format(
        brief=domain_brief,
        term=candidate.term,
        surfaces=", ".join(candidate.surface_forms) or candidate.term,
        freq=candidate.frequency,
        kind=candidate.kind,
        self_defs=_format_self_defs(self_defs),
        examples=_format_examples(candidate.example_messages, cfg.examples_per_call),
    )
    data, resp = llm.complete_json(_SYSTEM, user)
    return _parse_draft(candidate.term, data, resp.text)


def _parse_draft(term: str, data: dict, raw_text: str) -> DraftEntry:
    senses_in = data.get("senses") or []
    senses: list[Sense] = []
    nested_conf: float | None = None
    nested_reasoning: str | None = None
    for s in senses_in:
        if not isinstance(s, dict):
            continue
        senses.append(Sense(
            definition=str(s.get("definition", "")).strip(),
            context_clues=[str(c).strip() for c in (s.get("context_clues") or [])],
            example_message_ids=[
                int(i) for i in (s.get("example_message_ids") or [])
                if _is_intish(i)
            ],
        ))
        # Some models (notably gpt-4o-mini) sneak confidence/reasoning
        # *inside* the first sense object. Capture them as a fallback.
        if nested_conf is None and "confidence" in s:
            try:
                nested_conf = float(s["confidence"])
            except (TypeError, ValueError):
                pass
        if nested_reasoning is None and isinstance(s.get("reasoning"), str):
            nested_reasoning = s["reasoning"].strip()

    conf_raw = data.get("confidence", nested_conf)
    try:
        conf_f = float(conf_raw) if conf_raw is not None else 0.0
    except (TypeError, ValueError):
        conf_f = 0.0

    reasoning_raw = data.get("reasoning") or nested_reasoning or ""

    return DraftEntry(
        term=term,
        relevant=bool(data.get("relevant", False)),
        senses=senses,
        confidence=max(0.0, min(1.0, conf_f)),
        reasoning=str(reasoning_raw).strip(),
        raw_response=raw_text,
    )


def _is_intish(v) -> bool:
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False
