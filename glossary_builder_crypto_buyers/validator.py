"""Stage 5 — adversarial validation.

A fresh LLM call, *deliberately framed to attack the draft*. We do not ask
"is this correct?" — we ask "find the strongest reason this is wrong."

If the critique is compelling (per a structured self-rating) we lower the
confidence; if it's not, we leave it alone. Either way the critique is stored
on the entry so any disputed glossary item can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ValidatorConfig
from .inference import DraftEntry, Sense
from .llm import LLMClient


_SYSTEM = """\
You are an adversarial reviewer of glossary entries used by an automated lead-
scoring agent. Your job is NOT to confirm the entry. Your job is to find the
strongest possible reason the entry is wrong, incomplete, or ambiguous.

Be specific. Quote example messages if they contradict the definition. If you
genuinely cannot find an issue, say so plainly.

Distinguish carefully between three independent failure modes:
  1. NOT DOMAIN-SPECIFIC: the term is generic language (in any language) that
     a non-industry speaker would immediately recognise. Examples of NOT
     domain-specific: "manager", "feedback", "обратную связь", "platform",
     "развитие". These belong in a dictionary, not a domain glossary.
  2. WRONGLY ADMITTED noise: bot artefacts, command fragments, user handles,
     truncated tokens (e.g. "Development" extracted from "Business
     Development"), random proper nouns with no domain meaning.
  3. INCOMPLETE: term IS domain-specific and IS correctly admitted, but the
     definition misses a clearly-attested secondary sense. Example: NDA used
     as a placeholder for an unnamed company ("Head of payments NDA, igaming")
     in addition to the legal contract sense.

Mode 3 is NOT a reason to mark the entry as wrongly_relevant — it just means
we need to add senses. Reserve wrongly_relevant for modes 1 and 2.

Output strict JSON only.\
"""


_USER_TEMPLATE = """\
DOMAIN CONTEXT:
{brief}

PROPOSED GLOSSARY ENTRY:
  term: {term}
  relevant: {relevant}
  senses:
{senses}
  reasoning: {reasoning}
  confidence: {confidence}

ORIGINAL EXAMPLE MESSAGES (for verification):
{examples}

Find the strongest reasons this entry could be wrong. Consider all three
failure modes from the system prompt.

Critically, answer this question independently of how good the *definition*
looks:
  > Would someone who speaks the language(s) of these messages but has NO
  > payments / iGaming / fraud / risk industry experience immediately
  > understand this term as it appears in the examples?

If YES (generic language), set domain_specific=false. This overrides any
otherwise-correct definition — the term doesn't belong in a glossary.

POLYSEMY HEURISTIC — read carefully:
A term has *multiple senses* (mode 3) ONLY when the existing definition
cannot explain a significant share of example messages, AND the unexplained
messages share a clear structural pattern. Common signals:
  - The term appears repeatedly in profile-signature contexts
    ("Role at [TERM], vertical"), with no surrounding sentence.
  - The term is followed by a vertical name, geography, or product
    descriptor ("[TERM] PSP", "[TERM] igaming", "[TERM] Light").
  - The term is used in fixed compounds the definition doesn't mention.

When you see these signals, default to MODE 3 (record missing_senses) —
DO NOT default to wrongly_admitted_noise. The term IS domain-specific;
the existing definition is just incomplete.

STRICT — do NOT flag missing_senses when:
  - The existing definition already covers the unexplained examples
    if read charitably.
  - The supposed "second sense" is just a paraphrase, generalisation,
    or instance of the first sense.
  - The supposed "second sense" is "the term used as a placeholder /
    company shorthand in profile signatures" UNLESS the term itself
    is functionally an anonymiser (like NDA). Most brand names, role
    titles, and concepts appearing in signatures (CEO, Head, PSP,
    casino) are NOT polysemous — they're just being used in their
    normal sense in a structured context.
  - The supposed "second sense" is just a more specific modifier-form
    of the first sense (e.g. "white acquiring" is not a separate sense
    of "acquiring", it's the same concept with an adjective).

If you cannot articulate a NEW MEANING that differs from sense 1 by more
than wording, do NOT add a missing_sense. An empty missing_senses array
is the default and correct answer for the vast majority of terms.

Positive worked example #1: `NDA`. A draft defining it as "Non-Disclosure
Agreement" is correct in the legal-document sense. But profile lines like
"Head of payments NDA, igaming" or "bizdev NDA PSP" show a SECOND sense:
NDA used as a placeholder for an anonymised employer ("a company I'm
under NDA with"). The new sense has a different REFERENT (an employer,
not a contract). The right response is:
  - missing_senses: [{{ description: "Placeholder in profile signatures
    meaning 'employer I cannot publicly name due to NDA'", ... }}]

Positive worked example #2: `FTD`. A draft defining it as "First Time
Deposit; the initial deposit by a new player" is correct in the EVENT
sense. But messages like "у кого есть Blik FTD?", "Apple Pay / Google
Pay FTD", "FTD+STD" use it as a CLASSIFICATION of payment methods /
traffic streams — meaning "a payment solution that accepts first-time
depositors as a market segment". Different referent (a market category,
not an event). The right response includes a second sense like:
"Used as a category/qualifier for payment methods that serve first-
time-deposit traffic ('Blik FTD' = a Blik solution accepting FTDs)."

Negative worked example #1: `CEO`. Every example is "Name / CEO / Company"
in a profile signature. There is no second sense — it's always the
chief executive role. Adding "CEO as placeholder for an executive role"
would be a paraphrase of the same referent. The right response is
missing_senses: [].

Negative worked example #2: `эквайринг`. Examples include "white-эквайринг",
"локальный эквайринг", "карточный эквайринг". These are NOT separate
senses — they are the same concept (acquiring) with different adjectives.
The right response is missing_senses: []. (Note the difference from FTD:
"Blik FTD" uses FTD as a NOUN classifier, while "white эквайринг"
uses "white" as an ADJECTIVE modifying эквайринг — different grammar,
different polysemy verdict.)

Return JSON:
{{
  "issue_severity": "none" | "minor" | "major",
  "domain_specific": <bool>,         // false ONLY for generic language (mode 1)
  "wrongly_admitted_noise": <bool>,  // true for bot artefacts / fragments / handles (mode 2)
  "missing_senses": [                // empty unless mode 3 applies
    {{
      "description": "<short, what the sense means in this corpus>",
      "evidence_message_ids": [<int>, ...],
      "context_clues": ["<word>", ...]
    }}
  ],
  "critique": "<one or two sentences explaining the strongest issue, or 'none'>"
}}
"""


def _format_senses(draft: DraftEntry) -> str:
    if not draft.senses:
        return "    (no senses)"
    lines = []
    for i, s in enumerate(draft.senses):
        lines.append(f"    sense {i + 1}: {s.definition}")
        if s.context_clues:
            lines.append(f"      context_clues: {', '.join(s.context_clues)}")
    return "\n".join(lines)


def _format_examples(examples: list[dict], limit: int = 10) -> str:
    lines = []
    for ex in examples[:limit]:
        mid = ex.get("message_id", "?")
        t = (ex.get("text") or "").replace("\n", " ").strip()
        if len(t) > 240:
            t = t[:237] + "..."
        lines.append(f"  - msg_id={mid}: {t}")
    return "\n".join(lines) if lines else "  (none)"


@dataclass
class MissingSense:
    description: str
    evidence_message_ids: list[int]
    context_clues: list[str]

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "evidence_message_ids": self.evidence_message_ids,
            "context_clues": self.context_clues,
        }


@dataclass
class Critique:
    issue_severity: str
    critique: str
    domain_specific: bool
    wrongly_admitted_noise: bool
    missing_senses: list[MissingSense]

    @property
    def needs_enrichment(self) -> bool:
        """True when the entry should be enriched with extra senses instead of dropped."""
        return (
            self.domain_specific
            and not self.wrongly_admitted_noise
            and len(self.missing_senses) > 0
        )

    @property
    def wrongly_relevant(self) -> bool:
        """Legacy shim — true when the entry should be marked relevant=False."""
        return (not self.domain_specific) or self.wrongly_admitted_noise

    def to_dict(self) -> dict:
        return {
            "issue_severity": self.issue_severity,
            "critique": self.critique,
            "domain_specific": self.domain_specific,
            "wrongly_admitted_noise": self.wrongly_admitted_noise,
            "missing_senses": [m.to_dict() for m in self.missing_senses],
            "wrongly_relevant": self.wrongly_relevant,
        }


def _parse_missing(items) -> list[MissingSense]:
    out: list[MissingSense] = []
    for it in (items or []):
        if isinstance(it, str):
            out.append(MissingSense(description=it.strip(), evidence_message_ids=[], context_clues=[]))
            continue
        if not isinstance(it, dict):
            continue
        ids = []
        for i in (it.get("evidence_message_ids") or []):
            try:
                ids.append(int(i))
            except (TypeError, ValueError):
                pass
        out.append(MissingSense(
            description=str(it.get("description", "")).strip(),
            evidence_message_ids=ids,
            context_clues=[str(c).strip() for c in (it.get("context_clues") or [])],
        ))
    return [m for m in out if m.description]


def validate_entry(
    draft: DraftEntry,
    examples: list[dict],
    domain_brief: str,
    llm: LLMClient,
    config: ValidatorConfig | None = None,
) -> tuple[DraftEntry, Critique]:
    """Run the adversarial pass; mutate draft confidence/relevance accordingly."""
    cfg = config or ValidatorConfig()
    if not cfg.enabled:
        return draft, Critique("none", "validator disabled", True, False, [])

    # Deterministic backstop — the LLM consistently accepts generic business
    # vocab if it co-occurs with payment topics. Short-circuit those before
    # spending a token.
    if draft.term.lower().strip() in cfg.generic_vocab_blocklist:
        draft.relevant = False
        draft.confidence = min(draft.confidence, 0.2)
        return draft, Critique(
            issue_severity="major",
            critique="Term is generic business vocabulary, not domain-specific terminology (deterministic blocklist).",
            domain_specific=False,
            wrongly_admitted_noise=False,
            missing_senses=[],
        )

    user = _USER_TEMPLATE.format(
        brief=domain_brief,
        term=draft.term,
        relevant=draft.relevant,
        senses=_format_senses(draft),
        reasoning=draft.reasoning,
        confidence=draft.confidence,
        examples=_format_examples(examples),
    )
    data, _ = llm.complete_json(_SYSTEM, user)
    critique = Critique(
        issue_severity=str(data.get("issue_severity", "none")).lower(),
        critique=str(data.get("critique", "")).strip(),
        domain_specific=bool(data.get("domain_specific", True)),
        wrongly_admitted_noise=bool(data.get("wrongly_admitted_noise", False)),
        missing_senses=_parse_missing(data.get("missing_senses")),
    )

    # Generic vocab or admitted noise -> reject outright.
    if not critique.domain_specific or critique.wrongly_admitted_noise:
        draft.relevant = False
        draft.confidence = min(draft.confidence, 0.3)
        return draft, critique

    # Apply severity penalties only when not rejected outright.
    if critique.issue_severity == "major":
        draft.confidence = max(0.0, draft.confidence - cfg.confidence_penalty)
    elif critique.issue_severity == "minor":
        draft.confidence = max(0.0, draft.confidence - cfg.confidence_penalty / 2)

    return draft, critique


# -------- Polysemy enrichment ----------------------------------------------

_ENRICH_SYSTEM = """\
You draft additional glossary senses for a term that already has at least
one validated sense. A reviewer has flagged that the term has other meanings
in the corpus that the existing entry misses.

For each requested missing sense:
  - Write a concise definition (English, <=160 chars) that captures THIS
    specific usage, not the one the existing entry already covers.
  - List a few short context clues — words that, when co-occurring with the
    term, signal this particular sense.
  - Reference the message_ids that support the sense.

STRICT REJECTION CRITERIA — omit a requested sense if ANY apply:
  - It is a paraphrase or restatement of an existing sense.
  - It is the existing sense narrowed to a specific instance / modifier.
  - It is "the term used as a placeholder in profile signatures" — UNLESS
    the term is functionally an anonymiser (e.g. NDA). Brand names and
    role titles in signatures are NOT polysemous.
  - The example messages do not actually substantiate a distinct meaning.

If after applying these criteria nothing remains, return an empty
``new_senses`` array. That is a valid and common answer.

Output strict JSON only.\
"""


_ENRICH_USER_TEMPLATE = """\
DOMAIN CONTEXT:
{brief}

TERM: {term}

EXISTING SENSES (do not duplicate these):
{existing_senses}

REVIEWER-IDENTIFIED MISSING SENSES TO DRAFT:
{missing}

ALL ORIGINAL EXAMPLE MESSAGES:
{examples}

For each missing sense above, draft a glossary sense. Return JSON:
{{
  "new_senses": [
    {{
      "definition": "<<=160 chars, English>",
      "context_clues": ["<word>", ...],
      "example_message_ids": [<int>, ...]
    }}
  ]
}}

If after re-reading the examples you cannot substantiate a requested missing
sense, omit it from the response. Do not invent senses.
"""


def _format_existing_senses(draft: DraftEntry) -> str:
    if not draft.senses:
        return "  (none)"
    return "\n".join(f"  - {s.definition}" for s in draft.senses)


def _format_missing(missing: list[MissingSense]) -> str:
    if not missing:
        return "  (none)"
    lines = []
    for m in missing:
        lines.append(f"  - {m.description}")
        if m.context_clues:
            lines.append(f"      context_clues: {', '.join(m.context_clues)}")
        if m.evidence_message_ids:
            ids = ", ".join(str(i) for i in m.evidence_message_ids[:10])
            lines.append(f"      evidence_msg_ids: {ids}")
    return "\n".join(lines)


# -------- Semantic sense-distinctness check -------------------------------

_DEDUP_SYSTEM = """\
You are an editor of a domain glossary. Your only job is to decide whether
multiple senses listed for a single term are genuinely DISTINCT meanings or
whether some are PARAPHRASES of others.

DEFAULT TO KEEPING SENSES. The penalty for a false drop (losing a legit
sense) is high; the penalty for a false keep (one redundant sense) is low.
Only drop a sense if you are confident it's a paraphrase, not just related.

KEEP both senses when:
  - They refer to materially different things in the world. Examples:
    * 'merch' = a merchant entity  vs  'merch' = physical promotional goods
    * 'NDA' = legal contract  vs  'NDA' = anonymised-employer placeholder
    * 'FTD' = the deposit event  vs  'FTD' = a category of payment methods
        that serve first-time-deposit traffic ("Blik FTD")
    * 'Apple' = the company  vs  'Apple Pay' = a specific service (these
        would be separate entries normally — keep both senses if one entry
        attempts to cover both)
  - One sense is the dictionary meaning and the other is a domain-specific
    usage. KEEP both.
  - Sense B is a clearly different *referent* — even if related to sense A.
    A category and an instance are different referents. A noun and an
    adjective use are different referents. A process and the artefact of
    the process are different referents.

DROP a sense only when:
  - It explicitly says "shorthand for [definition that matches another sense]"
    or "placeholder for [same as another sense]" — and the example messages
    don't actually show the term used in a structurally different way
    (no profile-signature pattern, no anonymiser usage, etc.).
  - Two senses give the same definition with rephrased wording, referring
    to the same thing.
  - Sense B is a more verbose restatement of sense A without contributing
    new information.

When in doubt, KEEP. Output strict JSON only.
"""


_DEDUP_USER_TEMPLATE = """\
TERM: {term}
SENSES:
{senses_block}

Decide which senses to KEEP. For each sense, output its index and either
"keep" (it is genuinely distinct from all others you keep) or "drop"
(it is a paraphrase of another sense).

Return JSON:
{{
  "kept_sense_indices": [<int>, ...],
  "reasoning": "<one sentence>"
}}

At least one sense must be kept. If all senses are paraphrases, keep the
one with the clearest / most concrete definition.
"""


def _format_senses_for_dedup(senses: list[Sense]) -> str:
    lines = []
    for i, s in enumerate(senses):
        lines.append(f"  [{i}] {s.definition}")
    return "\n".join(lines)


def dedupe_paraphrase_senses(
    validated: list[tuple[DraftEntry, Critique]],
    llm: LLMClient,
) -> list[dict]:
    """For every relevant entry with >1 sense, ask an LLM if senses are
    genuinely distinct or paraphrases of each other. Drop the paraphrases.

    Domain-agnostic — operates purely on sense definitions, no external
    knowledge of THIS chat's specific patterns.

    Returns a list of drop records for the artifact log.
    """
    drops: list[dict] = []
    llm.set_stage("stage5d_sense_dedup")
    for draft, _critique in validated:
        if not draft.relevant or len(draft.senses) < 2:
            continue
        user = _DEDUP_USER_TEMPLATE.format(
            term=draft.term,
            senses_block=_format_senses_for_dedup(draft.senses),
        )
        try:
            data, _ = llm.complete_json(_DEDUP_SYSTEM, user, max_tokens=200)
        except Exception:
            continue
        kept_raw = data.get("kept_sense_indices") or []
        kept: list[int] = []
        for i in kept_raw:
            try:
                idx = int(i)
                if 0 <= idx < len(draft.senses) and idx not in kept:
                    kept.append(idx)
            except (TypeError, ValueError):
                pass
        if not kept:
            continue  # safety: don't strip all senses
        if len(kept) == len(draft.senses):
            continue  # nothing to drop
        # Record the drops, then rebuild senses list.
        for i in range(len(draft.senses)):
            if i not in kept:
                drops.append({
                    "term": draft.term,
                    "dropped_sense_index": i,
                    "kept_sense_indices": kept,
                    "dropped_definition": draft.senses[i].definition,
                    "reasoning": str(data.get("reasoning", "")).strip(),
                })
        draft.senses = [draft.senses[i] for i in kept]
        if draft.source_stage.endswith("polysemy_enrichment") and len(draft.senses) == 1:
            draft.source_stage = "tier1_inference"
    return drops


def enrich_with_missing_senses(
    draft: DraftEntry,
    critique: Critique,
    examples: list[dict],
    domain_brief: str,
    llm: LLMClient,
    confidence_bump: float = 0.10,
) -> DraftEntry:
    """If the critique flagged missing senses, draft and append them.

    Idempotent w.r.t. confidence — bumps it up modestly because the entry
    is now more complete, but capped at 1.0.
    """
    if not critique.needs_enrichment:
        return draft
    user = _ENRICH_USER_TEMPLATE.format(
        brief=domain_brief,
        term=draft.term,
        existing_senses=_format_existing_senses(draft),
        missing=_format_missing(critique.missing_senses),
        examples=_format_examples(examples, limit=20),
    )
    try:
        data, _ = llm.complete_json(_ENRICH_SYSTEM, user)
    except Exception:
        return draft
    new_raw = data.get("new_senses") or []
    added: list[Sense] = []
    for s in new_raw:
        if not isinstance(s, dict):
            continue
        definition = str(s.get("definition", "")).strip()
        if not definition:
            continue
        ids = []
        for i in (s.get("example_message_ids") or []):
            try:
                ids.append(int(i))
            except (TypeError, ValueError):
                pass
        added.append(Sense(
            definition=definition,
            context_clues=[str(c).strip() for c in (s.get("context_clues") or [])],
            example_message_ids=ids,
        ))
    if added:
        draft.senses.extend(added)
        draft.confidence = min(1.0, draft.confidence + confidence_bump)
        draft.source_stage = "tier1_inference+polysemy_enrichment"
    return draft
