"""Shared keyword stop-lists, applied BEFORE any LLM call.

One source of truth so every pipeline bans the same scam/recovery vocabulary.

RECOVERY_STOP_WORDS   — "recovery" (fund-recovery / broker-lead scam) and its transliterations.
CHARGEBACK_STOP_WORDS — chargeback variants. Kept separate because in a payment-provider
                        context "no chargebacks" is a legitimate selling point, so not every
                        pipeline wants to ban them.

Matching is lower-cased substring containment (same convention as the pre-filters).
"""

from __future__ import annotations

RECOVERY_STOP_WORDS: list[str] = [
    "recovery",
    "recover",          # recover worldwide / recovers (covers "recovery" too)
    "rekaver", "rekover", "rekavery", "recaver", "recavery", "ricovery",
    "рековери", "рекавери", "рекавер", "рековер", "рекаверь",
    "recovery leads",
    "database leads",
    "recovery chargeback",
]

CHARGEBACK_STOP_WORDS: list[str] = [
    "chardzeback", "chardgeback", "charge back", "chargeback",
    "чардж", "чарджбэк", "чарджбек", "чарджбэки",
]


def contains_recovery(text: str) -> bool:
    """True if text contains any recovery stop word (case-insensitive)."""
    low = (text or "").lower()
    return any(w in low for w in RECOVERY_STOP_WORDS)
