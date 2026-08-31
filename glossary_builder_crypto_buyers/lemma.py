"""Russian lemmatization + clustering of phrase candidates.

Stage 1b emits phrases as the LLM produced them, often in inflected forms
(`мерчанта`, `мерчантов`, `платежных решений`). Without normalization the
glossary ends up with multiple rows for what is conceptually one term.

We use pymorphy3 to compute a *cluster key* for each phrase — the lemma of
each word joined by a single space. Candidates that share a cluster key get
merged: frequencies summed, example messages unioned, every surface form
recorded in `surface_forms`. The canonical term is whichever surface form
appeared most often.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .candidates import Candidate


_WORD_RE = re.compile(r"[A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF0-9\-']*")

def _get_en_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except ImportError:
        return None
    if not hasattr(_get_en_lemmatizer, "_cached"):
        _get_en_lemmatizer._cached = WordNetLemmatizer()
    return _get_en_lemmatizer._cached

def _get_analyzer():
    """Lazily build a pymorphy3 analyzer (it's ~300ms to construct)."""
    try:
        import pymorphy3
    except ImportError:
        return None
    # Cache on the function object so we only build once per process.
    if not hasattr(_get_analyzer, "_cached"):
        _get_analyzer._cached = pymorphy3.MorphAnalyzer()
    return _get_analyzer._cached


def cluster_key(phrase: str) -> str:
    """Return the cluster key for a phrase — lemmas of all words, joined.

    Latin words pass through lowercased. Cyrillic words go through pymorphy.
    If pymorphy isn't available we fall back to lowercasing the whole phrase,
    which means no clustering happens. That's a safe no-op.
    """
    analyzer = _get_analyzer()
    if analyzer is None:
        return phrase.lower().strip()
    lemmas: list[str] = []
    for tok in _WORD_RE.finditer(phrase):
        w = tok.group(0)
        # Heuristic: skip lemmatization for words with no Cyrillic — pymorphy
        # has English handling but it's noisy.
        if any("\u0400" <= c <= "\u04FF" for c in w):
            try:
                lemmas.append(analyzer.parse(w)[0].normal_form.lower())
            except Exception:
                lemmas.append(w.lower())
        else:
            en_lemmer = _get_en_lemmatizer()
            if en_lemmer:
                lemmas.append(en_lemmer.lemmatize(w.lower()))
            else:
                lemmas.append(w.lower())
    return " ".join(lemmas) if lemmas else phrase.lower().strip()


def cluster_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Collapse candidates that share a cluster key into one canonical entry.

    Frequency is summed, examples and surface forms are unioned. The canonical
    term is the surface form with the highest individual frequency in the
    cluster (which is a reasonable proxy for "the most natural form").
    """
    cands = list(candidates)
    if not cands:
        return []

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        groups[cluster_key(c.term)].append(c)

    merged: list[Candidate] = []
    for key, members in groups.items():
        if len(members) == 1:
            merged.append(members[0])
            continue
        # Pick canonical term — the member with the highest frequency.
        members.sort(key=lambda m: m.frequency, reverse=True)
        canonical = members[0]
        surface_forms: list[str] = []
        seen_forms: set[str] = set()
        seen_msg_ids: set = set()
        examples: list[dict] = []
        total_freq = 0
        kinds: set[str] = set()
        for m in members:
            total_freq += m.frequency
            kinds.add(m.kind)
            for sf in [m.term] + (m.surface_forms or []):
                if sf and sf.lower() not in seen_forms:
                    seen_forms.add(sf.lower())
                    surface_forms.append(sf)
            for ex in m.example_messages or []:
                mid = ex.get("message_id")
                key_id = mid if mid is not None else id(ex)
                if key_id in seen_msg_ids:
                    continue
                seen_msg_ids.add(key_id)
                examples.append(ex)
        # Cap examples to avoid huge prompts downstream.
        examples = examples[:30]
        merged.append(Candidate(
            term=canonical.term,
            kind=canonical.kind if len(kinds) == 1 else "phrase",
            frequency=total_freq,
            example_messages=examples,
            surface_forms=surface_forms,
        ))
    merged.sort(key=lambda c: c.frequency, reverse=True)
    return merged
