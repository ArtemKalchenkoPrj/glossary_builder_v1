"""Crypto buyer classifier — webhook-side pipeline.

Looks for COMPANIES/businesses buying crypto on a recurring or large-volume
basis (OTC/P2P business buyers), as opposed to private individuals buying a
small one-off amount, sellers, exchanges pitching a two-way service, or
generic market discussion.

Pipeline:

    1. Keyword filter        (local, free)      — keywords.txt
       -> 0 hits: stop, nothing persisted
    2. Anti-keyword filter   (local, free, HARD BLOCK) — anti_keywords.txt
       -> any hit: stop immediately, nothing persisted, no LLM call at all.
          This is a hard reject, not a soft signal: if a message matches an
          anti-pattern (e.g. private/one-off/cash-in-person signals), it is
          rejected even if it also matched keywords.txt.
    3. LLM verification (GLOSSARY_MODEL)  → is_lead true/false — stage3_prompt.txt
       -> if is_lead != true: stop, nothing persisted
    4. LLM field extraction + judge (JUDGE_MODEL) → verdict + company, asset,
       network, pay_asset, pay_method, amount, min, max, confidence, notes
       — stage4_prompt.txt
       -> if verdict != REAL_LEAD: stop, nothing persisted

Keywords, anti-keywords, and both prompts are NOT hardcoded in this module —
they are loaded once at import time from a "vertical" directory, using the
exact same directory contract as vertical_analysis.py (the offline batch
classifier used to tune/tag these same files against a CSV export):

    <vertical_dir>/keywords.txt          required, one regex pattern per line
    <vertical_dir>/anti_keywords.txt     required for this classifier (hard
                                          block is always on) — one regex per
                                          line
    <vertical_dir>/stage3_prompt.txt     required, verify-stage system prompt
    <vertical_dir>/stage4_prompt.txt     required, extract+judge system prompt

Default vertical dir: ./vertical next to this file. Override with the
CRYPTO_BUYER_VERTICAL_DIR env var — e.g. point it at the exact same folder
you pass to `vertical_analysis.py --vertical ...` so the webhook and the
offline batch tool are always testing/running against identical rules.

Blank lines and lines starting with "#" in keywords.txt / anti_keywords.txt
are ignored (same convention as vertical_analysis.py's loader).

No author-context logic (unlike psp_provider_classifier.py) — each message
is classified independently, same as casino_classifier.py / crypto_seller.

IMPORTANT DIFFERENCE FROM psp/casino: this classifier does NOT write to
client_ready_leads. It UPDATEs the same row the primary pipeline already
created in classified_messages_dirty (source_lead_id = that row's id),
filling in crypto-specific columns and flipping is_lead/verdict so the row
surfaces in the manual-review queue. lead_type is set to 'crypto_buyer'
ONLY at the point of persist — i.e. only once verdict == REAL_LEAD has
already been confirmed. If the message isn't a REAL_LEAD (or was hard-
rejected by the anti-keyword filter), nothing is written and lead_type is
left untouched. Manual approval, downstream promotion to client_ready_leads,
etc. are out of scope here.

Entry point: run_crypto_buyer_classifier(...), called from webhook._run().

Requires new columns on classified_messages_dirty:

    ALTER TABLE classified_messages_dirty
        ADD COLUMN asset jsonb,
        ADD COLUMN network jsonb,
        ADD COLUMN pay_asset jsonb,
        ADD COLUMN pay_method jsonb,
        ADD COLUMN amount double precision,
        ADD COLUMN "min" double precision,
        ADD COLUMN "max" double precision,
        ADD COLUMN company text;

    -- existing columns reused: verdict, is_lead, judge_reason, confidence, lead_type
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import asyncpg
import httpx
from json_repair import repair_json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_TABLE_PREFIX = os.getenv("TABLE_PREFIX") or ""

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _api_key() -> str:
    key = os.environ.get("CRYPTO_BUYERS_API_KEY")
    if not key:
        raise RuntimeError(
            "CRYPTO_BUYERS_API_KEY not set in .env — "
            "crypto_buyer_classifier cannot call OpenRouter without it."
        )
    return key


def _verify_model() -> str:
    return os.environ.get("GLOSSARY_MODEL", "openai/gpt-4.1-nano")


def _extract_model() -> str:
    return os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Vertical loading — keywords / anti-keywords / prompts come from files,
# not from hardcoded constants. Same directory contract as
# vertical_analysis.py's _load_vertical(), so both tools can point at the
# same folder and always agree on what "a crypto buyer lead" means.
# ---------------------------------------------------------------------------

_VERTICAL_DIR = Path(
    os.environ.get("CRYPTO_BUYER_VERTICAL_DIR")
    or (Path(__file__).resolve().parent / "vertical")
)

_CRYPTO_BUYER_KEYWORDS: list[str] = []
_COMPILED_KEYWORDS: list[re.Pattern] = []

_ANTI_KEYWORDS: list[str] = []
_COMPILED_ANTI_KEYWORDS: list[re.Pattern] = []

_VERIFY_SYSTEM_PROMPT: str = ""
_EXTRACT_SYSTEM_PROMPT: str = ""


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]


def _load_vertical(vertical_dir: Path) -> None:
    """Load keywords.txt, anti_keywords.txt, stage3_prompt.txt,
    stage4_prompt.txt from `vertical_dir` into the module-level globals.

    Called once at import time. anti_keywords.txt is REQUIRED for this
    classifier (unlike the optional anti-filter in vertical_analysis.py) —
    the hard-block behaviour is always on here, so a missing file is
    treated as a misconfiguration, not "anti-filter disabled".
    """
    global _CRYPTO_BUYER_KEYWORDS, _COMPILED_KEYWORDS
    global _ANTI_KEYWORDS, _COMPILED_ANTI_KEYWORDS
    global _VERIFY_SYSTEM_PROMPT, _EXTRACT_SYSTEM_PROMPT

    kw_path = vertical_dir / "keywords.txt"
    anti_kw_path = vertical_dir / "anti_keywords.txt"
    stage3_path = vertical_dir / "stage3_prompt.txt"
    stage4_path = vertical_dir / "stage4_prompt.txt"

    missing = [p for p in (kw_path, anti_kw_path, stage3_path, stage4_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "crypto_buyer_classifier: missing vertical file(s): "
            + ", ".join(str(p) for p in missing)
            + f" — expected keywords.txt, anti_keywords.txt, stage3_prompt.txt, "
              f"stage4_prompt.txt in '{vertical_dir}'. Set CRYPTO_BUYER_VERTICAL_DIR "
              f"to point at the correct folder if it lives elsewhere."
        )

    keywords = _read_lines(kw_path)
    anti_keywords = _read_lines(anti_kw_path)

    with open(stage3_path, encoding="utf-8") as f:
        stage3_prompt = f.read()
    with open(stage4_path, encoding="utf-8") as f:
        stage4_prompt = f.read()

    _CRYPTO_BUYER_KEYWORDS = keywords
    _COMPILED_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in keywords]
    _ANTI_KEYWORDS = anti_keywords
    _COMPILED_ANTI_KEYWORDS = [re.compile(p, re.IGNORECASE) for p in anti_keywords]
    _VERIFY_SYSTEM_PROMPT = stage3_prompt
    _EXTRACT_SYSTEM_PROMPT = stage4_prompt

    logger.info(
        "crypto_buyer_classifier: vertical loaded from '%s': %d keywords, "
        "%d anti-keywords, stage3 %d chars, stage4 %d chars",
        vertical_dir, len(keywords), len(anti_keywords),
        len(stage3_prompt), len(stage4_prompt),
    )


_load_vertical(_VERTICAL_DIR)


def _keyword_hits(text: str) -> int:
    return sum(1 for pattern in _COMPILED_KEYWORDS if pattern.search(text))


def _anti_keyword_match(text: str) -> Optional[str]:
    """Return the first matching anti-pattern, or None if no anti-pattern
    matched. Returning the pattern (not just True/False) makes debug logs
    useful without having to re-run the regex scan by hand."""
    for pattern in _COMPILED_ANTI_KEYWORDS:
        if pattern.search(text):
            return pattern.pattern
    return None


# ---------------------------------------------------------------------------
# OpenRouter call helper
# ---------------------------------------------------------------------------

async def _call_openrouter(model: str, system: str, user: str, max_tokens: int) -> dict:
    """POST one chat-completion request to OpenRouter, return parsed JSON.

    Uses json_repair to handle malformed JSON responses from the model.
    On any failure returns {"_error": ...} so callers can branch without try/except.
    """
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```json\s*|```", "", raw).strip()
            return json.loads(repair_json(clean))

    except Exception as exc:
        logger.error("crypto_buyer_classifier: OpenRouter call failed (model=%s): %s", model, exc)
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# DB write — UPDATE, not INSERT: this classifier enriches the SAME dirty row
# the primary pipeline already created, it does not create a new one and
# does not touch client_ready_leads.
# ---------------------------------------------------------------------------

_UPDATE_SQL = f"""
UPDATE {_TABLE_PREFIX}classified_messages_dirty
SET
    is_lead      = TRUE,
    verdict      = 'REAL_LEAD',
    judge_reason = $2,
    company      = $3,
    asset        = $4,
    network      = $5,
    pay_asset    = $6,
    pay_method   = $7,
    amount       = $8,
    "min"        = $9,
    "max"        = $10,
    confidence   = $11,
    lead_type    = 'crypto_buyer',
    geo          = $12
WHERE id = $1
RETURNING id;
"""

async def _persist_crypto_buyer_lead(
    *,
    dirty_row_id: Optional[int],
    extracted: dict,
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> None:
    """UPDATE the existing classified_messages_dirty row with crypto fields.

    Requires dirty_row_id (the primary pipeline's db_id) — if it's missing
    (e.g. the primary insert failed) there's nothing to update, so callers
    should not invoke this without a valid id.

    If dry_run=True, prints the row that would have been updated instead
    of touching the database.
    """
    company     = extracted.get("company")
    asset       = extracted.get("asset") or []
    network     = extracted.get("network") or []
    pay_asset   = extracted.get("pay_asset") or []
    pay_method  = extracted.get("pay_method") or []
    geo         = extracted.get("geo") or []
    amount      = extracted.get("amount")
    min_val     = extracted.get("min")
    max_val     = extracted.get("max")
    confidence  = extracted.get("confidence")
    notes       = extracted.get("notes")

    if dry_run:
        print("\n" + "=" * 70)
        print("[crypto_buyer_classifier DRY RUN] UPDATE that would run:")
        print("=" * 70)
        print(f"  WHERE id        : {dirty_row_id}")
        print(f"  is_lead         : TRUE")
        print(f"  verdict         : REAL_LEAD")
        print(f"  lead_type       : crypto_buyer")
        print(f"  judge_reason    : {notes}")
        print(f"  company         : {company}")
        print(f"  asset           : {asset}")
        print(f"  network         : {network}")
        print(f"  pay_asset       : {pay_asset}")
        print(f"  pay_method      : {pay_method}")
        print(f"  geo             : {geo}")
        print(f"  amount          : {amount}")
        print(f"  min             : {min_val}")
        print(f"  max             : {max_val}")
        print(f"  confidence      : {confidence}")
        print("=" * 70 + "\n")
        return

    async with conn_pool.acquire() as conn:
        row = await conn.fetchrow(
            _UPDATE_SQL,
            dirty_row_id,
            notes,
            company,
            json.dumps(asset, ensure_ascii=False),
            json.dumps(network, ensure_ascii=False),
            json.dumps(pay_asset, ensure_ascii=False),
            json.dumps(pay_method, ensure_ascii=False),
            amount,
            min_val,
            max_val,
            confidence,
            json.dumps(geo, ensure_ascii=False),  # $12
        )

    logger.info(
        "crypto_buyer_classifier: updated dirty row id=%s company=%s asset=%s geo=%s",
        row["id"] if row else None, company, asset, geo,
    )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_crypto_buyer_classifier(
    *,
    text: str,
    source_lead_id: Optional[int],
    message_id: Optional[int],
    username: Optional[str],
    conn_pool: asyncpg.Pool,
    dry_run: bool = False,
) -> bool:
    """Run the crypto buyer classifier on a message the primary pipeline
    rejected (is_lead=False or verdict=MISTAKE).

    Unlike psp/casino, this writes back into classified_messages_dirty
    (UPDATE by source_lead_id), not client_ready_leads — crypto leads go
    through the same manual-review flow as the primary pipeline's leads.

    Note: `timestamp` isn't needed here (unlike psp/casino) since we're
    updating an existing row that already has its own msg_timestamp-
    equivalent (`timestamp` column) — nothing new to write.
    """
    text = (text or "").strip()
    if not text:
        logger.debug("crypto_buyer_classifier: empty text, skipping message_id=%s", message_id)
        return False
    if source_lead_id is None:
        logger.warning(
            "crypto_buyer_classifier: no source_lead_id (primary insert failed?), "
            "skipping message_id=%s", message_id
        )
        return False

    logger.debug("crypto_buyer_classifier: start, message_id=%s text=%.50r", message_id, text)

    # --- Step 1: keyword filter -----------------------------------------------
    hits = _keyword_hits(text)
    logger.debug("crypto_buyer_classifier: keyword hits=%d message_id=%s", hits, message_id)
    if hits == 0:
        logger.debug("crypto_buyer_classifier: no keyword hits, skipping message_id=%s", message_id)
        return False

    # --- Step 2: anti-keyword filter (HARD BLOCK, no LLM call) -----------------
    anti_match = _anti_keyword_match(text)
    if anti_match is not None:
        logger.debug(
            "crypto_buyer_classifier: anti-keyword hard block (message_id=%s): pattern=%r",
            message_id, anti_match,
        )
        return False

    # --- Step 3: LLM verification ---------------------------------------------
    logger.debug("crypto_buyer_classifier: calling verify model=%s", _verify_model())

    verify_result = await _call_openrouter(
        model=_verify_model(),
        system=_VERIFY_SYSTEM_PROMPT,
        user=text,
        max_tokens=200,
    )
    logger.debug("crypto_buyer_classifier: verify result=%s", verify_result)

    if verify_result.get("_error"):
        logger.warning(
            "crypto_buyer_classifier: verification failed for message_id=%s: %s",
            message_id, verify_result["_error"],
        )
        return False

    if verify_result.get("is_lead") is not True:
        logger.debug(
            "crypto_buyer_classifier: not a lead (message_id=%s): %s",
            message_id, verify_result.get("reason"),
        )
        return False

    # --- Step 4: LLM field extraction + judge ----------------------------------
    logger.debug("crypto_buyer_classifier: calling extract model=%s", _extract_model())

    extract_result = await _call_openrouter(
        model=_extract_model(),
        system=_EXTRACT_SYSTEM_PROMPT,
        user=text,
        max_tokens=400,
    )
    logger.debug("crypto_buyer_classifier: extract result=%s", extract_result)

    if extract_result.get("_error"):
        logger.warning(
            "crypto_buyer_classifier: extraction failed for message_id=%s: %s",
            message_id, extract_result["_error"],
        )
        return False

    if extract_result.get("verdict") != "REAL_LEAD":
        logger.debug(
            "crypto_buyer_classifier: judge downgraded verdict (message_id=%s): %s",
            message_id, extract_result.get("verdict"),
        )
        return False

    # --- Step 5: persist (UPDATE the dirty row) --------------------------------
    await _persist_crypto_buyer_lead(
        dirty_row_id=source_lead_id,
        extracted=extract_result,
        conn_pool=conn_pool,
        dry_run=dry_run,
    )
    return True