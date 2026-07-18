"""Stage 1c — curated gazetteer scanner (domain-agnostic).

The scanner matches a list of canonical entries against the corpus
regardless of frequency. The *list* of entries is domain-specific and
lives under ``glossary_builder.domain_data`` — this module is pure logic.

To target a different vertical, supply your own gazetteer list to
``scan(...)`` with the same shape:

    [{"canonical": "Foo", "forms": ["Foo", "foos"], "category": "..."}, ...]

The default ``scan()`` call uses the payments / iGaming gazetteer shipped
with this package for the original use case.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from .candidates import Candidate
from .domain_data import crypto_otc_buyers
from .loader import Message


# The DEFAULT gazetteer lives in domain_data/crypto_otc_buyers.py. Swap that
# module out (or pass gazetteer=... to ``scan``) to retarget the pipeline at
# a different vertical.
_DEFAULT_GAZETTEER = crypto_otc_buyers.gazetteer


def _compile_pattern(form: str) -> re.Pattern:
    """Build a case-insensitive search pattern for a surface form.

    Word-boundary behaviour:
      - For pure Latin/digit/hyphen forms we use look-arounds against
        ASCII alphanumerics on both sides.
      - For forms with non-ASCII chars (Cyrillic, mixed) we fall back to
        substring match — Python's ``\\b`` is letter-defined but mixed
        scripts can produce surprising matches; a substring is safer.
    """
    escaped = re.escape(form)
    if all((c.isascii() and (c.isalnum() or c in "-_.+ /")) for c in form):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _compile_gazetteer(entries: Sequence[dict]):
    return [(e, [_compile_pattern(f) for f in e["forms"]]) for e in entries]


def scan(
    messages: list[Message],
    *,
    gazetteer: Sequence[dict] | None = None,
    min_frequency: int = 1,
    max_examples_per_term: int = 15,
) -> list[Candidate]:
    """Walk the corpus and emit a Candidate per gazetteer entry that hits.

    Args:
        messages: corpus to scan.
        gazetteer: list of canonical entries. Each entry must have
            ``canonical`` (str), ``forms`` (list[str]), ``category`` (str).
            Defaults to the payments/iGaming gazetteer.
        min_frequency: drop entries with fewer than this many matching
            messages. Default 1 — we trust the curated list.
        max_examples_per_term: how many example messages to keep per term.

    Frequencies are counted per message (not per occurrence), so a single
    message that says "Skrill Skrill Skrill" counts once.
    """
    msgs = [m for m in messages if m.text]
    if not msgs:
        return []

    entries = gazetteer if gazetteer is not None else _DEFAULT_GAZETTEER
    compiled = _compile_gazetteer(entries)

    freq: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict]] = defaultdict(list)
    surfaces_seen: dict[str, set] = defaultdict(set)
    category: dict[str, str] = {}

    for msg in msgs:
        text = msg.text
        for entry, patterns in compiled:
            canonical = entry["canonical"]
            hit_surface: str | None = None
            for form, pat in zip(entry["forms"], patterns):
                if pat.search(text):
                    hit_surface = form
                    break
            if hit_surface is None:
                continue
            freq[canonical] += 1
            surfaces_seen[canonical].add(hit_surface)
            category[canonical] = entry["category"]
            if len(examples[canonical]) < max_examples_per_term:
                examples[canonical].append({
                    "message_id": msg.message_id,
                    "date": msg.date.isoformat() if msg.date else None,
                    "username": msg.username,
                    "text": text,
                })

    out: list[Candidate] = []
    for canonical, count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
        if count < min_frequency:
            continue
        out.append(Candidate(
            term=canonical,
            kind=f"gazetteer:{category[canonical]}",
            frequency=count,
            example_messages=examples[canonical],
            surface_forms=list(surfaces_seen[canonical]),
        ))
    return out