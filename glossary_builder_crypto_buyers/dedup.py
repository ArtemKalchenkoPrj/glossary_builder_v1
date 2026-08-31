"""Post-validation deduplication of redundant glossary content.

Two independent dedup steps, both deterministic and domain-agnostic:

1. **Parent-child term pairs** — Stage 1d (brand mining) and the gazetteer
   occasionally produce both a standalone parent token (`Apple`, `Google`)
   and a compound (`Apple Pay`, `Google Pay`). They refer to the same
   concept so the standalone parent is noise.

2. **Redundant senses within one entry** — the LLM enrichment stage
   sometimes adds a "second sense" that paraphrases or narrows sense 1
   without introducing a genuinely different referent. We detect this
   *structurally*:
     - lexical overlap between sense definitions (Jaccard of content words)
     - example-message-id overlap (how much one sense's evidence is
       a subset of another's)

   Both signals are language- and domain-neutral.
"""

from __future__ import annotations

import re
from typing import Iterable

from .inference import DraftEntry, Sense
from .validator import Critique


# A token regex that works across scripts — used only to compute lexical
# overlap between sense definitions. Stop-word-ish words are filtered out
# universally below.
_WORD_RE = re.compile(r"[A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF0-9'\-]*")


# Cross-language function-word stop list. Deliberately conservative — we
# only strip words that carry no semantic content in either language. We do
# NOT add domain-specific stop words here; that would defeat portability.
_SENSE_STOPWORDS = {
    # English function words
    "a", "an", "the", "of", "to", "for", "in", "on", "at", "by", "with",
    "and", "or", "but", "is", "are", "as", "that", "this", "these", "those",
    "it", "its", "be", "been", "from", "into", "via", "such",
    # Russian function words
    "и", "в", "на", "с", "по", "от", "до", "для", "к", "у", "о", "об",
    "из", "за", "над", "под", "при", "без", "это", "тот", "та", "те",
    "или", "но", "а", "не", "ни", "же", "ли", "бы",
    # Definition-y filler that adds no signal
    "term", "refers", "used", "specific", "type", "kind", "context",
    "process", "service", "services",
    # Ukrainian stopwords
    "і", "та", "в", "на", "з", "до", "для", "від", "про",
    "але", "або", "що", "як", "це", "той", "ті", "ці",
    "не", "ні", "же", "би", "об", "за", "під", "при", "без",
    "із", "зі", "над", "між", "через",
}


def _content_words(text: str) -> set[str]:
    """Tokenize a definition into normalised content words for similarity."""
    if not text:
        return set()
    out: set[str] = set()
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in _SENSE_STOPWORDS or len(w) <= 2:
            continue
        out.add(w)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Suffix tokens that strongly imply a compound is the same concept as its
# parent (e.g. "Apple Pay" is the same wallet concept as "Apple").
_COMPOUND_SUFFIXES = {
    "Pay", "Wallet", "Money", "Cash", "Bank", "Bot",
}


# Explicit allow-list of known parent-child pairs to dedupe even if the
# suffix rule doesn't catch them.
_EXPLICIT_PAIRS: set[tuple[str, str]] = {
    ("Apple", "Apple Pay"),
    ("Google", "Google Pay"),
    ("Samsung", "Samsung Pay"),
    ("Mir", "Mir Pay"),
}


def _shorter_is_prefix(shorter: str, longer: str) -> bool:
    """True if shorter is a leading whole-word of longer."""
    if not shorter or not longer or shorter == longer:
        return False
    if len(shorter) >= len(longer):
        return False
    if not longer.lower().startswith(shorter.lower()):
        return False
    next_char = longer[len(shorter):len(shorter) + 1]
    return next_char in (" ", "-", "_", ".")


def _suffix_matches(shorter: str, longer: str) -> bool:
    """True if longer = shorter + ' ' + KNOWN_SUFFIX."""
    if not _shorter_is_prefix(shorter, longer):
        return False
    tail = longer[len(shorter):].strip(" -_.")
    return tail in _COMPOUND_SUFFIXES


def _msg_id_overlap(a_ids: Iterable[int], b_ids: Iterable[int]) -> float:
    a, b = set(a_ids), set(b_ids)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def prune_redundant_senses(
    validated: list[tuple[DraftEntry, Critique]],
    *,
    lexical_jaccard_threshold: float = 0.55,
    example_overlap_threshold: float = 0.80,
) -> list[dict]:
    """For every entry with >1 sense, drop senses that duplicate another.

    A sense ``j`` is dropped if BOTH:
      - its content-word definition has Jaccard overlap >= lexical_threshold
        with some earlier sense ``i``
      - its example_message_ids are mostly a subset (>= overlap_threshold)
        of sense ``i``'s ids

    These two signals together identify paraphrase / generalisation senses
    that share both wording and evidence — i.e. the LLM said the same thing
    twice. Senses that share only one of the two signals are kept.

    Returns a list of drop records for the artifact log. Mutates the
    DraftEntry objects in place.
    """
    drops: list[dict] = []
    for draft, _critique in validated:
        if not draft.relevant or len(draft.senses) < 2:
            continue
        # Precompute content-word sets and id sets per sense.
        defs = [_content_words(s.definition) for s in draft.senses]
        ids = [set(s.example_message_ids) for s in draft.senses]
        keep_flags = [True] * len(draft.senses)

        for j in range(1, len(draft.senses)):
            if not keep_flags[j]:
                continue
            for i in range(j):
                if not keep_flags[i]:
                    continue
                lex = _jaccard(defs[i], defs[j])
                if not ids[j]:
                    # Sense without evidence ids — fall back to lexical only.
                    if lex >= lexical_jaccard_threshold:
                        keep_flags[j] = False
                        drops.append({
                            "term": draft.term,
                            "dropped_sense_index": j,
                            "kept_sense_index": i,
                            "lexical_jaccard": round(lex, 3),
                            "example_overlap": None,
                            "reason": "redundant_sense_no_ids",
                        })
                        break
                    continue
                # Example overlap = how much sense j's evidence is in sense i.
                overlap = len(ids[i] & ids[j]) / len(ids[j])
                if lex >= lexical_jaccard_threshold and overlap >= example_overlap_threshold:
                    keep_flags[j] = False
                    drops.append({
                        "term": draft.term,
                        "dropped_sense_index": j,
                        "kept_sense_index": i,
                        "lexical_jaccard": round(lex, 3),
                        "example_overlap": round(overlap, 3),
                        "reason": "redundant_sense_lexical_and_example",
                    })
                    break

        if not all(keep_flags):
            draft.senses = [s for s, keep in zip(draft.senses, keep_flags) if keep]
            # If we just pruned an enriched second sense, reflect that in source.
            if draft.source_stage.endswith("polysemy_enrichment") and len(draft.senses) == 1:
                draft.source_stage = "tier1_inference"
    return drops


def dedupe_parent_child(
    validated: list[tuple[DraftEntry, Critique]],
    *,
    example_overlap_threshold: float = 0.5,
) -> tuple[list[tuple[DraftEntry, Critique]], list[dict]]:
    """Walk the validated set and demote redundant parent tokens.

    Returns (possibly-mutated list, list of dropped-entry records for the
    artifact log).
    """
    drops: list[dict] = []
    relevant = [(d, c) for d, c in validated if d.relevant]

    # For each (shorter, longer) candidate pair, decide if we drop shorter.
    for d_short, c_short in relevant:
        for d_long, _ in relevant:
            if d_short is d_long or not d_short.relevant:
                continue
            pair_explicit = (d_short.term, d_long.term) in _EXPLICIT_PAIRS
            suffix_match = _suffix_matches(d_short.term, d_long.term)
            if not (pair_explicit or suffix_match):
                continue

            short_ids = {
                mid for s in d_short.senses for mid in s.example_message_ids
            }
            long_ids = {
                mid for s in d_long.senses for mid in s.example_message_ids
            }
            overlap = _msg_id_overlap(short_ids, long_ids)
            if not pair_explicit and overlap < example_overlap_threshold:
                continue

            d_short.relevant = False
            d_short.confidence = min(d_short.confidence, 0.25)
            reason = (
                f"Redundant with longer compound entry {d_long.term!r} "
                f"(example overlap {overlap:.0%})"
            )
            existing = c_short.critique or ""
            c_short.critique = (existing + " " if existing else "") + reason
            drops.append({
                "dropped": d_short.term,
                "in_favour_of": d_long.term,
                "example_overlap": round(overlap, 3),
                "reason": "parent_child_dedup",
            })
            break  # only drop once per shorter term
    return validated, drops
