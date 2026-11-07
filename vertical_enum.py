"""Canonical vertical enum — single source of truth for all pipelines.

Every extractor/judge prompt must include VERTICAL_RULES in its system
template so the LLM knows the closed set and the mandatory mappings.
"""

CANONICAL_VERTICALS = frozenset({
    "igaming", "casino", "betting", "forex", "crypto", "adult",
    "nutra", "dating", "trading", "mlm", "subscription",
    "call_centers", "resale", "esports", "other", "unknown",
})

VERTICAL_RULES = """\
CRITICAL — ``vertical`` is a CLOSED ENUM. You MUST output ONLY values from this list:
  igaming, casino, betting, forex, crypto, adult, nutra, dating,
  trading, mlm, subscription, call_centers, resale, esports, other, unknown.
NO other strings are allowed. NEVER copy a raw term from the message — always map it.

Vertical definitions:
  - "igaming"       — online gambling industry as a whole: operators, platforms, aggregators.
                       Includes any generic mention of gambling/гемблинг/gaming in iGaming context.
  - "casino"        — online casino products specifically: slots, table games, live casino, poker.
  - "betting"       — sports betting, bookmakers, odds, betting platforms, sportsbook.
  - "forex"         — currency exchange, FX brokers, currency pairs, MetaTrader.
  - "crypto"        — cryptocurrency, blockchain, Web3, VASP, exchanges, wallets, on/off-ramp.
  - "adult"         — adult content, webcam platforms, adult entertainment.
  - "nutra"         — supplements, weight loss, peptides, pharma, health products.
  - "dating"        — dating apps, matchmaking services.
  - "trading"       — binary options, CFD, financial trading platforms (NOT forex).
  - "mlm"           — multi-level marketing, network marketing, pyramid schemes.
  - "subscription"  — recurring-billing SaaS, apps, digital services (not gambling/dating).
  - "call_centers"  — outbound/inbound phone sales, telemarketing, "кц", "сс", "колл-центр".
  - "resale"        — reselling goods, dropshipping, arbitrage of physical products.
  - "esports"       — competitive gaming (CS2, Dota, LoL), esports betting, киберспорт.
  - "other"         — a vertical IS mentioned but does not match any value above.
  - "unknown"       — NO industry signal at all. Never use if any vertical hint exists.

Mandatory mappings (apply WITHOUT exception):
  "gambling" / "гемблинг" / "гамблинг"           → "igaming"
  "gaming" (in iGaming context)                   → "igaming"
  "mobile gaming"                                 → "igaming"
  "sportsbook" / "беттинг" / "ставки" / "bookmaker" → "betting"
  "poker" / "покер"                               → "casino"
  "fx" / "FX"                                     → "forex"
  "pharma" / "peptides" / "пептиды" / "medicine"  → "nutra"
  "web3" / "Web3" / "VASP" / "crypto/VASP"        → "crypto"
  "drugs"                                         → "nutra"
  "subscriptions"                                 → "subscription"
  "киберспорт" / "cybersport" / "e-sport"         → "esports"

  "high-risk" / "high risk" / "other high-risk"   → do NOT use as vertical (it is a risk category, not an industry).
                                                     Instead pick the actual industry: igaming, casino, betting, forex, crypto, adult, nutra, or "other".
  "digital" / "digital payments" / "digital goods" → "other"
  "finance" / "fintech" / "banking"               → "other"
  "e-commerce" / "ecommerce"                      → "other"
  "SaaS" / "saas" / "apps"                        → "other"
  "insurance" / "car insurance" / "health insurance" → "other"
  "loans" / "loan" / "payday" / "credit card"     → "other"
  "lead gen" / "affiliate" / "affiliate marketing" → "other"
  "PSP" / "psp" / "payments" / "remittance"       → "other"
  "p2p" / "exchange"                              → "other"
  "real estate" / "education" / "hr"              → "other"
  "b2b" / "whitelabel" / "white label"            → "other"
  "streamers" / "content creators"                → "other"
  "prediction market" / "prediction markets"      → "other"
  "binary options"                                → "trading"
  "homeowner" / "local and online services"       → "other"
  "company" / "business transactions"             → "other"
  anything else not in the enum                   → "other"\
"""

# ---------------------------------------------------------------------------
# Code-level normalizer (fallback for when LLM ignores prompt rules)
# ---------------------------------------------------------------------------

VERTICAL_ALIASES: dict[str, str] = {
    # igaming
    "gambling":        "igaming",
    "гемблинг":        "igaming",
    "гамблинг":        "igaming",
    "gaming":          "igaming",
    "mobile gaming":   "igaming",
    "i-gaming":        "igaming",
    "igambling":       "igaming",
    "online gambling": "igaming",
    # casino
    "poker":           "casino",
    "покер":           "casino",
    "slots":           "casino",
    "live casino":     "casino",
    "table games":     "casino",
    # betting
    "sportsbook":      "betting",
    "sports betting":  "betting",
    "беттинг":         "betting",
    "ставки":          "betting",
    "bookmaker":       "betting",
    "sport":           "betting",
    "sports":          "betting",
    # forex
    "fx":              "forex",
    "currency":        "forex",
    "p2p":             "forex",
    "exchange":        "forex",
    # crypto
    "web3":            "crypto",
    "vasp":            "crypto",
    "crypto/vasp":     "crypto",
    "blockchain":      "crypto",
    "cryptocurrency":  "crypto",
    # nutra
    "pharma":          "nutra",
    "peptides":        "nutra",
    "пептиды":         "nutra",
    "medicine":        "nutra",
    "drugs":           "nutra",
    "supplements":     "nutra",
    "health":          "nutra",
    "weight loss":     "nutra",
    # trading
    "binary options":  "trading",
    "cfd":             "trading",
    "options":         "trading",
    # subscription
    "subscriptions":   "subscription",
    "saas":            "subscription",
    # call_centers
    "кц":              "call_centers",
    "сс":              "call_centers",
    "колл-центр":      "call_centers",
    "call center":     "call_centers",
    "telemarketing":   "call_centers",
    # esports
    "киберспорт":      "esports",
    "cybersport":      "esports",
    "e-sport":         "esports",
    "e-sports":        "esports",
    "esport":          "esports",
    # other (risk categories, generic industry labels)
    "high-risk":       "other",
    "high risk":       "other",
    "other high-risk": "other",
    "digital":         "other",
    "digital payments":"other",
    "digital goods":   "other",
    "finance":         "other",
    "fintech":         "other",
    "banking":         "other",
    "e-commerce":      "other",
    "ecommerce":       "other",
    "apps":            "other",
    "insurance":       "other",
    "car insurance":   "other",
    "health insurance":"other",
    "loans":           "other",
    "loan":            "other",
    "payday":          "other",
    "credit card":     "other",
    "lead gen":        "other",
    "affiliate":       "other",
    "affiliate marketing": "other",
    "psp":             "other",
    "payments":        "other",
    "remittance":      "other",
    "real estate":     "other",
    "education":       "other",
    "hr":              "other",
    "b2b":             "other",
    "whitelabel":      "other",
    "white label":     "other",
    "streamers":       "other",
    "content creators":"other",
    "prediction market":  "other",
    "prediction markets": "other",
    "homeowner":       "other",
    "local and online services": "other",
    "company":         "other",
    "business transactions": "other",
}


def normalize_verticals(verticals: list[str]) -> list[str]:
    """Normalize a list of vertical strings to canonical values.

    Unknown values that aren't in CANONICAL_VERTICALS or VERTICAL_ALIASES
    are mapped to "other".
    """
    result = []
    for v in verticals:
        v_lower = v.strip().lower()
        if not v_lower:
            continue
        if v_lower in CANONICAL_VERTICALS:
            normed = v_lower
        else:
            normed = VERTICAL_ALIASES.get(v_lower, "other")
        if normed not in result:
            result.append(normed)
    return result
