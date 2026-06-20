"""Stage 1b — LLM-assisted phrase extraction.

Stage 1 (`candidates.py`) is regex-only and only catches Latin acronyms,
mixed-script tokens, and Latin words. It misses every Russian-only domain
phrase — антифрод, отлив, мерч, карусель, чарджбэк, etc.

This stage covers that gap. We sample batches of messages, ask an LLM to
list notable industry-specific phrases that appear in each batch, then
aggregate the responses across batches. For every unique phrase we count
its actual corpus frequency and pull example messages — same shape as
Stage 1 output, so downstream stages don't care which extractor produced
the candidate.

Cost: ~30 LLM calls × ~1.5k input tokens. On gpt-4o-mini that's pennies.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

from tqdm import tqdm

from .candidates import Candidate
from .config import PhraseConfig
from .domain_data import payments_igaming
from .llm import LLMClient
from .loader import Message


# System prompt template — the {seed_block} placeholder is filled at call
# time with either PhraseConfig.seed_block or the default payments seed.
_SYSTEM_TEMPLATE = """\
You are an information-extraction assistant identifying industry-specific
terminology in chat messages.

For each batch of messages you read, list the *terms and short phrases*
(1-4 words each) that meet ALL of these criteria:

  - The phrase carries SPECIFIC meaning in the target industry.
  - A person without industry experience would NOT immediately understand it.
  - The phrase actually appears (in any inflected form) in at least one of
    the messages you are shown.

{seed_block}

Do NOT include:
  - Short acronyms 4 chars or shorter in any language (PSP, CEO, P2P, KYC,
    FTD, NDA, DM, etc.) — handled by a separate extractor. EXCEPTION: include
    numeric acronyms (3DS, MCC 7995) and multi-word compounds ("high-risk MID",
    "rolling reserve", "approve rate").
  - Everyday conversational words that ANY industry uses — greetings, thanks,
    generic nouns like "company", "information", "question", "feedback":
      Russian: привет, спасибо, компания, информация, вопрос, человек
      Ukrainian: привіт, дякую, компанія, інформація, питання
      English: hello, thanks, company, information, question, feedback
  - DO NOT confuse generic words with domain slang. These look simple but
    ARE domain-specific and MUST be included if they appear:
      мерчант, апрув, трафа, сетлмент, отлив, чарджбэк, каскад, холд,
      мерч, финтех, форекс, крипта, агрегатор, латам
  - Person names and Telegram usernames (@handle).
  - Phrases longer than 4 words or over 60 characters.

Output strict JSON. No prose. Use this schema exactly:
{
  "phrases": [
    {
      "phrase": "<phrase exactly as you would search for it, lower-case, lemma if possible>",
      "language": "ru" | "uk" | "en" | "mixed"
      "why_relevant": "<<=12 words>"
    }
  ]
}

If the batch contains no qualifying phrases, return {"phrases": []}.
"""


_USER_TEMPLATE = """\
DOMAIN CONTEXT:
{brief}

BATCH OF MESSAGES (each starts with msg_id):
{batch}

Return JSON with the qualifying phrases you found in this batch.
"""


def _format_batch(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        text = (m.text or "").replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:397] + "..."
        lines.append(f"[msg_id={m.message_id}] {text}")
    return "\n".join(lines)


def _make_batches(messages: list[Message], batch_size: int, max_batches: int,
                  rng_seed: int) -> list[list[Message]]:
    msgs = [m for m in messages if m.text]
    if not msgs:
        return []
    rng = random.Random(rng_seed)
    # Shuffle then chunk — gives us representative batches without bias toward
    # the start of the corpus.
    rng.shuffle(msgs)
    batches: list[list[Message]] = []
    for i in range(0, len(msgs), batch_size):
        batches.append(msgs[i:i + batch_size])
        if len(batches) >= max_batches:
            break
    return batches


def _normalize_phrase(phrase: str) -> str:
    """Cheap normalization so surface variants converge."""
    if not phrase:
        return ""
    s = unicodedata.normalize("NFKC", phrase).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:!?\"'()[]{}—-")
    return s


def _phrase_appears(phrase: str, text: str) -> bool:
    """Russian-friendly contains-check.

    We deliberately don't anchor on word boundaries — Russian morphology
    means "отлив" should match "отливы", "отливе", etc.
    """
    if not text or not phrase:
        return False
    return phrase in text.lower()


def extract_phrases(
    messages: list[Message],
    llm: LLMClient,
    domain_brief: str,
    config: PhraseConfig | None = None,
    rng_seed: int = 42,
    progress: bool = True,
) -> list[Candidate]:
    """Run LLM phrase extraction over sampled batches, then count + sample.

    Returns Candidate objects compatible with the Stage 1 output, so they
    flow through downstream stages without any extra plumbing.
    """
    cfg = config or PhraseConfig()
    if not cfg.enabled:
        return []

    batches = _make_batches(messages, cfg.messages_per_batch, cfg.max_batches, rng_seed)
    if not batches:
        return []

    llm.set_stage("stage1b_phrases")

    raw_phrases: Counter[str] = Counter()
    languages: dict[str, str] = {}
    why: dict[str, list[str]] = defaultdict(list)

    # Build the system prompt by stitching the seed block — domain-specific
    # examples come from config or default to the payments seed. We use
    # ``replace`` rather than ``format`` because the template contains JSON
    # schema braces that would confuse format-string parsing.
    seed_block = cfg.seed_block if cfg.seed_block is not None else payments_igaming.phrase_seed_block
    system = _SYSTEM_TEMPLATE.replace("{seed_block}", seed_block)

    iterator = tqdm(batches, desc="Stage 1b") if progress else batches
    for batch in iterator:
        user = _USER_TEMPLATE.format(brief=domain_brief, batch=_format_batch(batch))
        try:
            data, _ = llm.complete_json(system, user, max_tokens=900)
        except Exception:
            # One bad batch shouldn't kill the whole stage.
            continue
        for item in (data.get("phrases") or []):
            if not isinstance(item, dict):
                continue
            phrase = _normalize_phrase(item.get("phrase", ""))
            if not (cfg.min_phrase_length <= len(phrase) <= cfg.max_phrase_length):
                continue
            raw_phrases[phrase] += 1
            languages.setdefault(phrase, str(item.get("language") or "").lower())
            note = str(item.get("why_relevant") or "").strip()
            if note and note not in why[phrase]:
                why[phrase].append(note)

    if not raw_phrases:
        return []

    # Backfill real corpus frequencies + example messages for every unique
    # phrase. We do it here rather than trusting the per-batch counts
    # because each batch only sees a slice of the corpus.
    by_phrase_examples: dict[str, list[dict]] = defaultdict(list)
    by_phrase_freq: Counter[str] = Counter()

    msgs_with_text = [m for m in messages if m.text]
    candidates_set = list(raw_phrases.keys())

    for msg in msgs_with_text:
        haystack = msg.text.lower()
        for phrase in candidates_set:
            if phrase in haystack:
                by_phrase_freq[phrase] += 1
                if len(by_phrase_examples[phrase]) < cfg.max_examples_per_phrase:
                    by_phrase_examples[phrase].append({
                        "message_id": msg.message_id,
                        "date": msg.date.isoformat() if msg.date else None,
                        "username": msg.username,
                        "text": msg.text,
                    })

    out: list[Candidate] = []
    for phrase, freq in by_phrase_freq.most_common():
        if freq < cfg.min_frequency:
            continue
        lang = languages.get(phrase, "")
        kind = (
            "russian_phrase" if lang == "ru"
            else "ukrainian_phrase" if lang == "uk"
            else "mixed_phrase" if lang == "mixed"
            else "phrase"
        )
        out.append(Candidate(
            term=phrase,
            kind=kind,
            frequency=freq,
            example_messages=by_phrase_examples[phrase],
            surface_forms=[phrase],
        ))
    return out
