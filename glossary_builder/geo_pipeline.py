"""Geo normalization and expansion pipeline.

Pipeline that runs after the classifier extracts a raw geo list:

    Stage 1  load_maps()           — fetch normalization + expansion dicts from DB
    Stage 2  _pre_normalize()      — lowercase + strip every raw value
    Stage 3  normalize_geo()       — map raw values to canonical codes via norm_map;
                                     unknown values are logged to geo_unknown_values
    Stage 4  expand_geo()          — expand canonical codes to child codes via expansion_map
    Stage 5  apply_payment_geo()   — add geo codes implied by payment methods
                                     (e.g. "монобанк" -> "UA"), deduplicated.
                                     Runs AFTER expand_geo — payment-derived codes
                                     are NOT themselves expanded.

Public entry point:
    process_geo(message_id, raw_geo_list, payment_methods, conn) -> list[str]

Intended to be called from webhook._persist() after classification and before
the DB write, replacing result["geo"] with the processed list.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


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
    # Normalization map — keys are already lowercase in the DB.
    norm_rows = await conn.fetch(
        "SELECT raw_value, normalized FROM geo_normalization_map"
    )
    norm_map: dict[str, str] = {row["raw_value"]: row["normalized"] for row in norm_rows}

    # Expansion map — group children by parent.
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
    """Strip whitespace and lowercase a single raw geo value.

    Examples:
        "LATAM"  -> "latam"
        " KZ "   -> "kz"
        "Росія"  -> "росія"
    """
    return value.strip().lower()


# ---------------------------------------------------------------------------
# Stage 3 — normalize geo list via norm_map
# ---------------------------------------------------------------------------

async def normalize_geo(
    geo_list: list[str],
    norm_map: dict[str, str],
    message_id: Optional[int],
    conn: asyncpg.Connection,
) -> list[str]:
    """Map each raw geo value to its canonical code.

    For each value in geo_list:
    - Apply _pre_normalize (lowercase + strip).
    - Look up the result in norm_map.
    - Found     → append normalized code to result.
    - Not found → insert (raw_value, message_id) into geo_unknown_values
                  and skip the value (it does not enter the pipeline).

    Args:
        geo_list:   Raw geo values from the classifier, e.g. ["KZ", "Росія", "EU"].
        norm_map:   Normalization dict loaded by load_maps().
        message_id: ID of the message being processed; stored in geo_unknown_values
                    so unknown entries can be reprocessed after the map is updated.
        conn:       Active asyncpg connection.

    Returns:
        List of canonical geo codes, e.g. ["KZ", "RU", "EU"].
    """
    normalized: list[str] = []
    unknown: list[tuple[str, int]] = []  # (raw_value, message_id) pairs to bulk-insert

    for raw in geo_list:
        key = _pre_normalize(raw)
        if not key:
            continue

        canonical = norm_map.get(key)
        if canonical:
            normalized.append(canonical)
        else:
            logger.warning(
                "geo_pipeline: unknown geo value %r in message_id=%s — "
                "logged to geo_unknown_values",
                raw,
                message_id,
            )
            if message_id is not None:
                unknown.append((raw, message_id))

    # Bulk-insert all unknown values in one statement.
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
    """Expand each canonical geo code to itself plus any child codes.

    For each code in geo_list:
    - If the code is a key in expansion_map → keep the code AND append children.
    - Otherwise → keep the code as-is.

    Preserves order: original codes come first, children follow immediately
    after their parent.

    Example:
        geo_list      = ["KZ", "EU", "RU"]
        expansion_map = {"EU": ["AT", "BE", "PL"]}
        result        = ["KZ", "EU", "AT", "BE", "PL", "RU"]

    Args:
        geo_list:       Normalized geo codes from normalize_geo().
        expansion_map:  Expansion dict loaded by load_maps().

    Returns:
        Expanded list of geo codes.
    """
    expanded: list[str] = []
    for code in geo_list:
        expanded.append(code)
        children = expansion_map.get(code)
        if children:
            expanded.extend(children)
    return expanded


# ---------------------------------------------------------------------------
# Stage 5 — derive geo from payment methods
# ---------------------------------------------------------------------------

async def load_payment_geo_map(conn: asyncpg.Connection) -> dict[str, str]:
    """Fetch the payment_method -> geo_code mapping from the DB.

    Returns:
        {payment_method_lowercase: geo_code}
        e.g. {"монобанк": "UA", "qiwi": "RU"}
    """
    rows = await conn.fetch("SELECT payment_method, geo_code FROM payment_geo_map")
    return {row["payment_method"]: row["geo_code"] for row in rows}


async def apply_payment_geo(
    geo_list: list[str],
    payment_methods: list[str],
    payment_geo_map: dict[str, str],
) -> list[str]:
    """Append geo codes implied by payment methods, deduplicated.

    For each method in payment_methods:
    - Lowercase + strip the method name.
    - Look it up in payment_geo_map.
    - Found and not already present in geo_list → append the geo code.

    Runs AFTER expand_geo: codes added here are already canonical
    (e.g. "UA") and are NOT themselves run through expand_geo.

    Args:
        geo_list:         Geo codes after Stage 3+4 (normalize + expand).
        payment_methods:  Payment method values from the classifier,
                           e.g. ["visa", "монобанк"].
        payment_geo_map:  Mapping dict loaded by load_payment_geo_map().

    Returns:
        geo_list with any payment-derived codes appended, no duplicates.
    """
    result = list(geo_list)
    seen = set(result)

    for method in payment_methods:
        key = method.strip().lower()
        geo_code = payment_geo_map.get(key)
        if geo_code and geo_code not in seen:
            result.append(geo_code)
            seen.add(geo_code)

    return result


# ---------------------------------------------------------------------------
# Stage 6 — main entry point
# ---------------------------------------------------------------------------

async def process_geo(
    message_id: Optional[int],
    raw_geo_list: list[str],
    payment_methods: list[str],
    conn: asyncpg.Connection,
) -> list[str]:
    """Run the full geo pipeline for one message.

    Steps:
        1. Load norm_map, expansion_map, payment_geo_map from the DB.
        2. Normalize each raw value (lowercase + strip).  [inside normalize_geo]
        3. Map to canonical codes; log unknowns to geo_unknown_values.
        4. Expand canonical codes using expansion_map.
        5. Append geo codes implied by payment_methods, deduplicated.

    Args:
        message_id:       ID of the message; forwarded to normalize_geo for logging.
        raw_geo_list:      Raw geo list from the classifier, e.g. ["KZ", "Росія", "EU"].
        payment_methods:   Payment methods from the classifier,
                           e.g. ["visa", "монобанк"]. Pass [] if none.
        conn:              Active asyncpg connection (caller owns the lifecycle).

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
    payment_geo_map = await load_payment_geo_map(conn)

    # Stages 2 + 3 — pre-normalize then canonicalize.
    canonical = await normalize_geo(raw_geo_list or [], norm_map, message_id, conn)

    # Stage 4 — expand.
    expanded = expand_geo(canonical, expansion_map)

    # Stage 5 — derive extra geo from payment methods.
    result = await apply_payment_geo(expanded, payment_methods or [], payment_geo_map)

    return result