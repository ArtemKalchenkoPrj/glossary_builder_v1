"""Stage 1 — rule-based candidate term extraction.

Goal: surface every token in the corpus that *might* be a domain term, without
calling an LLM. The downstream LLM stages will decide what's actually useful.

We're deliberately broad here. False positives are cheap (filtered later);
false negatives are expensive (we'll never know we missed them).
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .config import CandidateConfig
from .loader import Message


# A "token" is a maximal run of letters / digits / a handful of in-word symbols.
# We accept both Latin and Cyrillic letters and the apostrophe / hyphen so that
# forms like ``FTD's`` or ``high-risk`` survive as one unit. We also allow
# a leading digit specifically to catch tokens like ``3DS``, ``2DS``, ``7995``.
_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04FF0-9][A-Za-z\u0400-\u04FF0-9'\-_]*")


# Known MCC codes worth flagging on their own. iGaming uses 7995; gambling-
# adjacent MCCs (6051, 5816, 7993, 7994) also surface in cascade discussions.
_MCC_CODES = {"7995", "6051", "5816", "7993", "7994", "5967", "6010", "6012"}


def _classify(token: str) -> str | None:
    """Return a *kind* label for a token, or None if it should be ignored."""
    if not token:
        return None

    has_latin = any("A" <= c <= "Z" or "a" <= c <= "z" for c in token)
    has_cyrl = any("\u0400" <= c <= "\u04FF" for c in token)
    has_digit = any(c.isdigit() for c in token)
    starts_with_digit = token[0].isdigit()

    # MCC code on its own — surface as a special kind so downstream knows.
    if token.isdigit() and token in _MCC_CODES:
        return "mcc"

    # Mixed-script tokens are almost always domain-relevant (transliterations,
    # Russian-suffixed English terms, e.g. "FTD'шник").
    if has_latin and has_cyrl:
        return "mixed"

    # Digit-prefixed acronyms: 3DS, 2DS, 4DS, etc. Length cap mirrors
    # plain acronyms to avoid grabbing model numbers.
    if starts_with_digit and has_latin and not has_cyrl:
        core = token.rstrip("s").rstrip("'")
        if 3 <= len(core) <= 6 and core[1:].isalpha() and core[1:].isupper():
            return "acronym"
        return None

    if has_latin and not has_cyrl and not has_digit:
        core = token.rstrip("s").rstrip("'")  # drop trivial plural / possessive
        if core.isupper() and 2 <= len(core) <= 6:
            return "acronym"
        if token[0].isupper() and not token.isupper():
            return "proper"  # CamelCase / brand-ish
        if token.islower() and len(token) >= 4:
            return "latin_word"
        return None

    # Latin + digit but not starting with digit (e.g. "P2P", "P2C", "B2B") —
    # treat as acronym if short and otherwise looks like initials.
    if has_latin and has_digit and not has_cyrl:
        if 3 <= len(token) <= 6 and token.isupper():
            return "acronym"
        return None

    # Pure Cyrillic — too noisy at the rule level; we leave Russian domain
    # phrases to the LLM-assisted extraction pass.
    return None


def _normalize(token: str) -> str:
    """Group obvious surface variants together (FTDs / ftd / FTD -> FTD)."""
    core = token.rstrip("s").rstrip("'")
    return core.upper() if _classify(token) == "acronym" else token


@dataclass
class Candidate:
    term: str
    kind: str
    frequency: int
    example_messages: list[dict] = field(default_factory=list)
    surface_forms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "kind": self.kind,
            "frequency": self.frequency,
            "surface_forms": self.surface_forms,
            "example_messages": self.example_messages,
        }


def extract_candidates(
    messages: Iterable[Message],
    config: CandidateConfig | None = None,
    rng_seed: int = 42,
) -> list[Candidate]:
    """Walk the corpus once and emit candidate terms with example messages.

    The output is sorted by frequency descending so downstream stages can
    budget LLM calls by importance.
    """
    cfg = config or CandidateConfig()
    rng = random.Random(rng_seed)

    counts: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    kinds: dict[str, str] = {}

    # Reservoir-sample examples per term so we get representative coverage
    # without storing the entire corpus in memory.
    seen_counts: Counter[str] = Counter()

    for msg in messages:
        if not msg.text:
            continue
        text = msg.text
        tokens_in_msg = set()
        for match in _TOKEN_RE.finditer(text):
            tok = match.group(0)
            kind = _classify(tok)
            if kind is None:
                continue
            norm = _normalize(tok)
            if norm.upper() in cfg.hard_skip:
                continue
            if kind == "latin_word" and len(norm) < cfg.min_latin_word_len:
                continue
            if kind == "acronym" and not (
                cfg.min_acronym_len <= len(norm) <= cfg.max_acronym_len
            ):
                continue
            tokens_in_msg.add((norm, kind, tok))

        for norm, kind, surface_form in tokens_in_msg:
            counts[norm] += 1
            surface[norm][surface_form] += 1
            kinds[norm] = kind

            # Reservoir sample: keep up to max_examples_per_term examples,
            # uniformly sampled from all messages that contain the term.
            seen_counts[norm] += 1
            example = {
                "message_id": msg.message_id,
                "date": msg.date.isoformat() if msg.date else None,
                "username": msg.username,
                "text": text,
            }
            bucket = examples[norm]
            if len(bucket) < cfg.max_examples_per_term:
                bucket.append(example)
            else:
                j = rng.randint(0, seen_counts[norm] - 1)
                if j < cfg.max_examples_per_term:
                    bucket[j] = example

    out: list[Candidate] = []
    for term, freq in counts.most_common():
        if freq < cfg.min_frequency:
            break  # most_common is sorted; everything after is below threshold
        out.append(Candidate(
            term=term,
            kind=kinds[term],
            frequency=freq,
            example_messages=examples[term],
            surface_forms=[s for s, _ in surface[term].most_common(5)],
        ))
    return out


def summarize(candidates: list[Candidate]) -> dict:
    """Quick stats for logging / sanity checks."""
    by_kind: Counter[str] = Counter()
    for c in candidates:
        by_kind[c.kind] += 1
    return {
        "total": len(candidates),
        "by_kind": dict(by_kind),
        "top_20": [
            {"term": c.term, "kind": c.kind, "frequency": c.frequency}
            for c in candidates[:20]
        ],
    }
