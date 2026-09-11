"""Geo normalization and expansion pipeline.

Pipeline that runs after the classifier extracts a raw geo list:

    Stage 1  load_maps()                — fetch normalization + expansion dicts from DB
    Stage 2  _pre_normalize()           — lowercase + strip every raw value
    Stage 3  normalize_geo()            — map raw values to canonical codes via norm_map;
                                          unknown values first go through Stage 3.5;
                                          unresolvable values are logged to geo_unknown_values
    Stage 3.5 _llm_resolve_geo()        — LLM fallback for values not found in norm_map;
                                          on success, persists new mapping to geo_normalization_map
                                          so future lookups skip the LLM entirely
    Stage 4  expand_geo()               — expand canonical codes to child codes via expansion_map
    Stage 5  apply_payment_geo()        — add geo codes implied by LOCAL payment methods
                                          (e.g. "монобанк" -> ["UA"]), deduplicated.
                                          Sourced from payment_options.local_method_country,
                                          matched via payment_options.aliases.
                                          Runs AFTER expand_geo — payment-derived codes
                                          are NOT themselves expanded.
    Stage 6  log_unknown_payment_methods() — log payment methods not found in any
                                              payment_options.aliases.

Public entry point:
    process_geo(message_id, raw_geo_list, payment_methods, conn) -> list[str]

Intended to be called from webhook._persist() after classification and before
the DB write, replacing result["geo"] with the processed list.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from .config import LLMConfig
from .llm import LLMClient, LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level LLM client for geo resolution.
# Lightweight task — nano model is sufficient.
#
# NOTE: this client forces model="gpt-4.1-nano" which requires OpenAI.
# If your deployment uses Anthropic as the primary provider (ANTHROPIC_API_KEY
# is set but OPENAI_API_KEY is not), _llm_resolve_geo will log an error and
# fall back to "garbage" — no crash, just no LLM resolution. To enable it,
# ensure OPENAI_API_KEY is set alongside your Anthropic key.
# ---------------------------------------------------------------------------

_geo_llm = LLMClient(LLMConfig(model="gpt-4.1-nano", max_tokens=50, temperature=0.0))

_GEO_SYSTEM = """\
You are a geo code resolver for a payments industry database.
You receive a raw geographic value (a country name, region, or abbreviation in any language) \
and a list of valid canonical codes.
Your job: return the single best matching code from the list.

Rules:
- Respond with ONLY the code. No explanation, no punctuation, nothing else.
- If you are not confident, respond with exactly: garbage
- When in doubt, respond with: garbage\
"""


# ---------------------------------------------------------------------------
# Stage 3.5 — LLM fallback resolver
# ---------------------------------------------------------------------------

def _llm_resolve_geo(raw: str, valid_codes: set[str]) -> str:
    """Try to map an unknown raw geo value to a canonical code via LLM.

    Synchronous — blocks the event loop briefly, which is acceptable given
    the low call volume (~500/month).

    Args:
        raw:         The original raw value as extracted from the message.
        valid_codes: Set of all canonical codes currently in geo_normalization_map.

    Returns:
        A code from valid_codes if the LLM resolved it confidently,
        or "garbage" if the value could not be mapped.
    """
    codes_str = ", ".join(sorted(valid_codes))
    user = f"Raw value: {raw}\nValid codes: {codes_str}"

    try:
        resp = _geo_llm.complete(system=_GEO_SYSTEM, user=user)
        candidate = resp.text.strip()

        # Accept if the LLM returned a code from our valid set (case-insensitive).
        candidate_upper = candidate.upper()
        if candidate_upper in valid_codes:
            logger.info(
                "geo_pipeline: LLM resolved %r -> %r",
                raw, candidate_upper,
            )
            return candidate_upper

        # Explicit garbage signal.
        if candidate.lower() == "garbage":
            return "garbage"

        # Anything else (hallucinated code, multi-word response, etc.) → garbage.
        logger.warning(
            "geo_pipeline: LLM returned unrecognised value %r for raw=%r — treating as garbage",
            candidate, raw,
        )
        return "garbage"

    except LLMError as exc:
        logger.error(
            "geo_pipeline: LLM resolve failed for raw=%r: %s — treating as garbage",
            raw, exc,
        )
        return "garbage"


# ---------------------------------------------------------------------------
# Stage 1 — load mappings from DB
# ---------------------------------------------------------------------------

async def load_maps(conn: asyncpg.Connection) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Fetch both mapping tables from the DB and return them as dicts.

    Returns:
        norm_map:      {raw_value_lowercase: normalized_code}
                       e.g. {"росія": "RU", "eu": "EU", "kz": "KZ"}

        expansion_map: {parent_code: [child_code, ...]}
                       e.g. {"EU": ["AT", "BE", "BG", ...], "LATAM": ["BR", "MX", ...]}

    Both dicts are built fresh on every call so changes to the DB
    are picked up without restarting the service.
    """
    norm_rows = await conn.fetch(
        "SELECT raw_value, normalized FROM geo_normalization_map"
    )
    norm_map: dict[str, str] = {row["raw_value"]: row["normalized"] for row in norm_rows}

    exp_rows = await conn.fetch(
        "SELECT parent, child FROM geo_expansion_map"
    )
    expansion_map: dict[str, list[str]] = {}
    for row in exp_rows:
        expansion_map.setdefault(row["parent"], []).append(row["child"])

    return norm_map, expansion_map


# ---------------------------------------------------------------------------
# Stage 2 — pre-normalize raw values
# ---------------------------------------------------------------------------

def _pre_normalize(value: str) -> str:
    """Strip whitespace and lowercase a single raw geo value."""
    return value.strip().lower()


# ---------------------------------------------------------------------------
# Stage 3 — normalize geo list via norm_map (with Stage 3.5 LLM fallback)
# ---------------------------------------------------------------------------

async def normalize_geo(
    geo_list: list[str],
    norm_map: dict[str, str],
    message_id: Optional[int],
    conn: asyncpg.Connection,
) -> list[str]:
    """Map each raw geo value to its canonical code.

    For values not found in norm_map, Stage 3.5 attempts LLM resolution:
    - On success: persists the new mapping to geo_normalization_map (so
      the next identical value hits the DB cache and skips the LLM),
      updates norm_map in-place for the remainder of this call, and
      includes the resolved code in the result.
    - On garbage: logs to geo_unknown_values as before.
    """
    normalized: list[str] = []
    unknown: list[tuple[str, int]] = []

    # Derive valid target codes once — used by the LLM resolver.
    valid_codes: set[str] = set(norm_map.values())

    for raw in geo_list:
        key = _pre_normalize(raw)
        if not key:
            continue

        canonical = norm_map.get(key)
        if canonical:
            normalized.append(canonical)
            continue

        # ---- Stage 3.5: LLM fallback ----
        logger.warning(
            "geo_pipeline: unknown geo value %r in message_id=%s — attempting LLM resolution",
            raw, message_id,
        )

        resolved = _llm_resolve_geo(raw, valid_codes)

        if resolved != "garbage":
            # Persist to DB so this raw_value never reaches the LLM again.
            await conn.execute(
                """
                INSERT INTO geo_normalization_map (raw_value, normalized)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                key, resolved,
            )
            # Update local norm_map so duplicate values in this batch
            # are resolved instantly without another LLM call.
            norm_map[key] = resolved
            valid_codes.add(resolved)  # resolved code was already in set, but safe to re-add
            normalized.append(resolved)
        else:
            logger.warning(
                "geo_pipeline: LLM could not resolve %r in message_id=%s — "
                "logged to geo_unknown_values",
                raw, message_id,
            )
            if message_id is not None:
                unknown.append((raw, message_id))

    if unknown:
        await conn.executemany(
            """
            INSERT INTO geo_unknown_values (raw_value, message_id)
            VALUES ($1, $2)
            ON CONFLICT (raw_value, message_id) DO NOTHING
            """,
            unknown,
        )

    return normalized


# ---------------------------------------------------------------------------
# Stage 4 — expand canonical codes via expansion_map
# ---------------------------------------------------------------------------

def expand_geo(
    geo_list: list[str],
    expansion_map: dict[str, list[str]],
) -> list[str]:
    """Expand each canonical geo code to itself plus any child codes."""
    expanded: list[str] = []
    for code in geo_list:
        expanded.append(code)
        children = expansion_map.get(code)
        if children:
            expanded.extend(children)
    return expanded


# ---------------------------------------------------------------------------
# Stage 5 — derive geo from LOCAL payment methods (payment_options)
# ---------------------------------------------------------------------------

async def load_local_method_geo_map(conn: asyncpg.Connection) -> dict[str, list[str]]:
    """Fetch alias -> local_method_country mapping from payment_options.

    Every alias of every method becomes its own key in the result dict,
    pointing to that method's local_method_country list (already
    normalized ISO codes in the DB). Global methods (empty
    local_method_country) simply map to an empty list — concatenating
    an empty list onto geo_list is a no-op, so no special-casing is
    needed downstream.

    Returns:
        {alias_lowercase: [geo_code, ...]}
        e.g. {"mono": ["UA"], "monobank": ["UA"], "моно": ["UA"], "visa": []}
    """
    rows = await conn.fetch(
        "SELECT aliases, local_method_country FROM payment_options"
    )
    alias_map: dict[str, list[str]] = {}
    for row in rows:
        countries = list(row["local_method_country"] or [])
        for alias in (row["aliases"] or []):
            alias_map[alias.strip().lower()] = countries
    return alias_map


async def apply_payment_geo(
    geo_list: list[str],
    payment_methods: list[str],
    local_method_map: dict[str, list[str]],
) -> list[str]:
    """Append geo codes implied by LOCAL payment methods, deduplicated.

    For each method in payment_methods:
    - Lowercase + strip the method name.
    - Look it up in local_method_map (built from payment_options.aliases).
    - Append every country in its local_method_country list that isn't
      already present in geo_list.

    Global methods (no match, or matched with an empty country list)
    contribute nothing — this is intentional, not an error case.

    Runs AFTER expand_geo: codes added here are already canonical
    and are NOT themselves run through expand_geo.

    Args:
        geo_list:          Geo codes after Stage 3+4 (normalize + expand).
        payment_methods:   Payment method values from the classifier,
                            e.g. ["visa", "монобанк"].
        local_method_map:  Mapping dict loaded by load_local_method_geo_map().

    Returns:
        geo_list with any payment-derived codes appended, no duplicates.
    """
    result = list(geo_list)
    seen = set(result)

    for method in payment_methods:
        key = method.strip().lower()
        for geo_code in local_method_map.get(key, []):
            if geo_code not in seen:
                result.append(geo_code)
                seen.add(geo_code)

    return result


# ---------------------------------------------------------------------------
# Stage 6 — log unknown payment methods
# ---------------------------------------------------------------------------

async def log_unknown_payment_methods(
    payment_methods: list[str],
    message_id: Optional[int],
    conn: asyncpg.Connection,
) -> None:
    """Log payment methods that don't match any alias in payment_options."""
    if not payment_methods or message_id is None:
        return

    unknown = []
    for method in payment_methods:
        key = method.strip().lower()
        row = await conn.fetchrow(
            "SELECT 1 FROM payment_options WHERE $1 = ANY(aliases)",
            key,
        )
        if row is None:
            unknown.append((method, message_id))
            logger.warning(
                "geo_pipeline: unknown payment method %r in message_id=%s",
                method, message_id,
            )

    if unknown:
        await conn.executemany(
            """
            INSERT INTO payment_unknown_values (raw_value, message_id)
            VALUES ($1, $2)
            ON CONFLICT (raw_value, message_id) DO NOTHING
            """,
            unknown,
        )


# ---------------------------------------------------------------------------
# Stage 7 — main entry point
# ---------------------------------------------------------------------------

async def process_geo(
    message_id: Optional[int],
    raw_geo_list: list[str],
    payment_methods: list[str],
    conn: asyncpg.Connection,
) -> list[str]:
    """Run the full geo pipeline for one message.

    Steps:
        1. Load norm_map, expansion_map, local_method_map from the DB.
        2. Normalize each raw value (lowercase + strip).  [inside normalize_geo]
        3. Map to canonical codes; unknown values go through LLM fallback (3.5)
           before being logged to geo_unknown_values.
        4. Expand canonical codes using expansion_map.
        5. Append geo codes implied by LOCAL payment_methods, deduplicated.
        6. Log payment methods not found in any payment_options.aliases.

    Args:
        message_id:      ID of the message; forwarded to normalize_geo for logging.
        raw_geo_list:    Raw geo list from the classifier, e.g. ["KZ", "Росія", "EU"].
        payment_methods: Payment methods from the classifier,
                         e.g. ["visa", "монобанк"]. Pass [] if none.
        conn:            Active asyncpg connection (caller owns the lifecycle).

    Returns:
        Processed geo list ready for the downstream filter step,
        e.g. ["KZ", "RU", "EU", "AT", "BE", "PL", ..., "UA"].
        Returns an empty list if both raw_geo_list and payment_methods
        are empty/None.
    """
    if not raw_geo_list and not payment_methods:
        return []

    # Stage 1 — load mappings fresh from DB on every call.
    norm_map, expansion_map = await load_maps(conn)
    local_method_map = await load_local_method_geo_map(conn)

    # Stages 2 + 3 (+3.5) — pre-normalize, canonicalize, LLM fallback for unknowns.
    canonical = await normalize_geo(raw_geo_list or [], norm_map, message_id, conn)

    # Stage 4 — expand.
    expanded = expand_geo(canonical, expansion_map)

    # Stage 5 — derive extra geo from LOCAL payment methods.
    result = await apply_payment_geo(expanded, payment_methods or [], local_method_map)

    # Stage 6 — log unknown payment methods.
    await log_unknown_payment_methods(payment_methods or [], message_id, conn)

    return result