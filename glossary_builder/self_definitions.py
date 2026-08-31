"""Stage 2 — mine inline self-definitions from the corpus (Tier 2).

People in technical chats constantly explain English acronyms for each other.
We sweep the archive for canonical patterns like:

    FTD (first time deposit)
    FTD — это первый депозит
    FTD, т.е. первый депозит игрока
    под FTD имею в виду ...
    FTD = first time deposit

A hit here is gold: the definition was written by a domain expert *inside* the
data we already trust. Stage 5's adversarial pass still validates them; this
stage just surfaces the raw evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .candidates import Candidate
from .loader import Message


@dataclass
class SelfDefinition:
    term: str
    definition: str
    pattern: str             # which regex matched, useful for debugging
    message_id: int | None
    message_text: str
    username: str | None

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "definition": self.definition,
            "pattern": self.pattern,
            "message_id": self.message_id,
            "message_text": self.message_text,
            "username": self.username,
        }


# Patterns are ordered from most-to-least specific. Each pattern has a label
# and a regex template. {term} will be substituted with the escaped term;
# group "def" captures the definition.
#
# Tested empirically against Russian payments / iGaming chatter.
_PATTERN_TEMPLATES: list[tuple[str, str]] = [
    # ── Українські (нові) ────────────────────────────────────────────────
    # FTD — це перший депозит
    ("em_dash_tse", r"\b{term}\b\s*[—\-–]\s*це\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD це перший депозит
    ("tse", r"\b{term}\b\s+це\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD, тобто перший депозит
    ("tobto", r"\b{term}\b\s*,?\s*тобто\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD означає ...
    ("oznachaie", r"\b{term}\b\s+означає\s+(?P<def>[^.\n]{{4,140}})"),
    # під FTD маю на увазі ...
    ("pid_maiу", r"під\s+{term}\b\s+маю\s+на\s+увазі\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD розшифровується як ...
    ("rozshyfr", r"\b{term}\b\s+розшифровується\s+як\s+(?P<def>[^.\n]{{4,140}})"),

    # ── Англійські (нові) ────────────────────────────────────────────────
    # FTD means first time deposit
    ("en_means", r"\b{term}\b\s+means\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD stands for first time deposit
    ("en_stands_for", r"\b{term}\b\s+stands\s+for\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD is a/an/the ...
    ("en_is", r"\b{term}\b\s+is\s+(?:a\s+|an\s+|the\s+)?(?P<def>[A-Za-z][^.\n]{{4,140}})"),
    # FTD, i.e. first time deposit
    ("en_ie", r"\b{term}\b\s*,?\s*i\.?e\.?\s+(?P<def>[^.\n]{{4,140}})"),
    # FTD — first time deposit  (em-dash без службового слова)
    ("em_dash_bare", r"\b{term}\b\s*[—–]\s*(?P<def>[A-Za-z\u0400-\u04FF][^.\n]{{4,140}})"),
    # by FTD I mean ...
    ("en_by_i_mean", r"by\s+{term}\b\s+(?:i\s+mean|we\s+mean)\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD (first time deposit)   /   FTD (первый депозит)
    ("parenthetical", r"\b{term}\b\s*\(\s*(?P<def>[^()]{{4,120}}?)\s*\)"),

    # FTD — это первый депозит игрока     (em-dash / hyphen + это)
    ("em_dash_eto", r"\b{term}\b\s*[—\-–]\s*это\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD это первый депозит игрока       (no dash)
    ("eto", r"\b{term}\b\s+это\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD, т.е. первый депозит            (т.е. = "i.e.")
    ("te", r"\b{term}\b\s*,?\s*т\.?\s*е\.?\s+(?P<def>[^.\n]{{4,140}})"),

    # под FTD имею в виду первый депозит  ("by FTD I mean")
    ("pod_imeyu", r"под\s+{term}\b\s+имею\s+в\s+виду\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD расшифровывается как ...        ("FTD stands for ...")
    ("rasshifr", r"\b{term}\b\s+расшифровыва(?:ется|ются)\s+как\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD = first time deposit
    ("equals", r"\b{term}\b\s*=\s*(?P<def>[^.\n]{{4,140}})"),

    # FTD aka first time deposit          (English-style)
    ("aka", r"\b{term}\b\s+(?:a\.?k\.?a\.?|aka)\s+(?P<def>[^.\n]{{4,140}})"),

    # FTD: first time deposit
    ("colon", r"\b{term}\b\s*:\s*(?P<def>[A-Za-z\u0400-\u04FF][^.\n]{{4,140}})"),
]


def _compile_for_term(term: str) -> list[tuple[str, re.Pattern]]:
    escaped = re.escape(term)
    out = []
    for label, tmpl in _PATTERN_TEMPLATES:
        try:
            pattern = re.compile(tmpl.format(term=escaped), re.IGNORECASE)
        except re.error:
            continue
        out.append((label, pattern))
    return out


def _clean_definition(d: str) -> str:
    d = d.strip().strip(".,;:—-– \t")
    # Drop a trailing parenthetical clause if any
    d = re.sub(r"\s*\([^)]*\)\s*$", "", d)
    return d


def mine_self_definitions(
    candidates: Iterable[Candidate],
    messages: Iterable[Message],
    max_per_term: int = 3,
) -> dict[str, list[SelfDefinition]]:
    """For each candidate, find inline definitions inside the message archive.

    Returns a mapping term -> list of SelfDefinition (deduped, capped).
    Terms with no hits don't appear in the output.

    Note: we recompile per term to keep the patterns simple. For 200 candidates
    over ~3k messages this completes in well under a second. If the corpus
    explodes, switch to a single combined regex with alternation.
    """
    candidates_list = list(candidates)
    messages_list = list(messages)

    out: dict[str, list[SelfDefinition]] = {}
    for cand in candidates_list:
        patterns = _compile_for_term(cand.term)
        seen_defs: set[str] = set()
        hits: list[SelfDefinition] = []
        for msg in messages_list:
            if not msg.text:
                continue
            # Cheap fast-fail: skip messages that don't contain the term at all.
            if cand.term.lower() not in msg.text.lower():
                continue
            for label, pat in patterns:
                m = pat.search(msg.text)
                if not m:
                    continue
                definition = _clean_definition(m.group("def"))
                if len(definition) < 4:
                    continue
                key = definition.lower()
                if key in seen_defs:
                    continue
                seen_defs.add(key)
                hits.append(SelfDefinition(
                    term=cand.term,
                    definition=definition,
                    pattern=label,
                    message_id=msg.message_id,
                    message_text=msg.text,
                    username=msg.username,
                ))
                if len(hits) >= max_per_term:
                    break
            if len(hits) >= max_per_term:
                break
        if hits:
            out[cand.term] = hits
    return out
