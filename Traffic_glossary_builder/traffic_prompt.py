"""All LLM prompts for the Traffic classifier.

Four prompts, in pipeline order:

    _TRAFFIC_STAGE1_SYSTEM   — cheap YES/NO micro-filter (nano model)
    TRAFFIC_DEFINITION       — what counts as a traffic lead (inserted into full LLM)
    _TRAFFIC_SYSTEM_TEMPLATE — full LLM system prompt with {traffic_definition}
    _TRAFFIC_JUDGE_SYSTEM    — independent judge: TRAFFIC_PROVIDER | TRAFFIC_LEAD | MISTAKE

Design principles:
    Stage 1 — fail-open. Says NO only for OBVIOUSLY non-target. Default = YES.
    Full LLM — precise. Detailed criteria with examples in RU / UK / EN.
    Judge    — fail-open. Default = confirm extractor. MISTAKE only for obvious errors.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 1 — cheap YES/NO micro-filter
# ---------------------------------------------------------------------------

_TRAFFIC_STAGE1_SYSTEM = """\
You screen Telegram messages to decide whether they deserve a full analysis
for affiliate traffic buying/selling intent.

Answer YES if the message MIGHT be from:
  - Someone who HAS traffic and is looking for an offer / advertiser / affiliate program
  - Someone who NEEDS traffic and is looking for webmasters / arbitrageurs / media buyers
  - An arbitrageur, webmaster, or media buyer discussing a deal

Answer NO only if the message is OBVIOUSLY none of the above:
  NO — payment processing / PSP context ("платёжный трафик", "payment traffic uptime")
  NO — IT/network traffic with no affiliate context ("DDoS", "network bandwidth", "API rate")
  NO — buying/selling social media accounts WITHOUT traffic ("куплю FB аккаунты", "фармленные акки")
  NO — database/lead list sales ("igaming data", "base of depositors", "fresh data for sale")
  NO — job posting or CV ("ищем media buyer в команду", "рассматриваю офферы от работодателей")
  NO — purely informational or general discussion with no buy/sell intent

If you are not sure — answer YES.
Answer with a single word: YES or NO.\
"""

# ---------------------------------------------------------------------------
# Traffic definition — inserted into full LLM prompt
# ---------------------------------------------------------------------------

TRAFFIC_DEFINITION = """\
A TRAFFIC LEAD is a message from someone who is actively BUYING or SELLING
affiliate traffic / leads in high-risk verticals.

── TRAFFIC_PROVIDER (seller) ────────────────────────────────────────────────
The author HAS traffic or leads and wants to monetise them. Signals:

  (a) Directly states they have or sell traffic / leads:
      "есть трафик на gambling", "продаю лиды", "we sell live forex traffic",
      "have FB traffic for gaming", "сливаю трафик на казино"

  (b) Announces traffic availability — even WITHOUT the word "sell":
      "Crypto / Forex Traffic Available", "Gambling SEO Traffic; Geo: All",
      "SERVICES AVAILABLE: Fx/crypto traffic available (live and database)",
      "Hi, we have traffic available from sports websites",
      "Live Crypto TRAFFIC — Wide Geo coverage, API integration"
      → Any message that announces AVAILABLE traffic types / GEOs / sources
        is implicitly OFFERING that traffic → TRAFFIC_PROVIDER.

  (c) Team / company presentation offering traffic as a product:
      "We are a media buying team delivering CPL / CRG traffic",
      "100FTD is an iGaming traffic team with 8+ years of experience,
       we work with direct advertisers" → identifies as traffic provider,
      "I present our media buying team. We sell live Forex & Crypto traffic.",
      "media buyers offering high-quality in-house Crypto & Forex traffic"
      → Any team self-introduction that names traffic types / sources they
        work with counts as TRAFFIC_PROVIDER even without the word "sell".

  (d) Sells ready FTDs, live leads, or data leads:
      "LIVE READY FTDs — Prepayment per FTD", "live leads, API integration",
      "data leads 48h old", "готовые FTD, предоплата"
      → Selling first-time deposits or pre-qualified leads = TRAFFIC_PROVIDER.

  (e) Looks for an offer / affiliate program / advertiser / brand / network:
      "ищу оффер под трафик", "looking for CPA offer",
      "LOOKING FOR BRANDS, NETWORKS — Crypto/Forex Traffic Available"
      → Has traffic, seeks brand/offer to monetise it = TRAFFIC_PROVIDER.
      "ищу оффера / залью трафика" — means "looking for offer / will drive traffic"
      = TRAFFIC_PROVIDER. "залью/слить/лить трафик" is Russian slang for driving traffic.

  (f) Identifies as arbitrageur / webmaster / media buyer seeking a deal:
      "арбитражник, работаю с нутрой, ищу оффер",
      "webmaster, have SEO traffic, need advertiser"

── SEEKING_TRAFFIC (buyer) ──────────────────────────────────────────────────
The author wants to ACQUIRE traffic or leads. Signals:

  (a) Directly seeks or buys traffic / leads — even a short message is enough:
      "ищу трафик на казино", "нужен трафик", "buying gambling traffic",
      "у кого есть трафик на US?", "шукаємо трафік із GEO Tier 1",
      "ищу арбитражника" — looking for an arbitrageur = seeking traffic.
      "Всем привет! Ищу трафик под прямой бренд DartWinner, CPA/RevShare"
      → A one-liner like "Ищу арбитражника. Нужен трафик." is SEEKING_TRAFFIC.

  (b) Looks for webmasters, arbitrageurs, or media buyers:
      "ищем вебмастеров под gambling", "looking for affiliates",
      "ищем партнеров с трафиком", "шукаємо партнерів з трафіком"

  (c) Direct advertiser / brand / casino that seeks traffic and offers
      payment terms TO webmasters (CPA / RevShare / Hybrid):
      "Мы прямой рекламодатель, ищем трафик по UZ, CPA и RevShare от 50%",
      "Прямой рекл Gambling, ищем трафик на Tier-1, CPA/RevShare/Hybrid",
      "Direct casino advertiser, looking for HQ live casino traffic",
      "Ищем трафик на Турцию и Россию. Прямой рекламодатель казино. CPA/RS",
      "Ми шукаємо трафік для інтернет-ігор. Ми є прямими рекламодавцями",
      "Шукаємо трафік із GEO Tier 1 та Tier 2. CPA, RevShare або Hybrid",
      "Ищем партнеров с трафиком на Tier-1. CPA | RevShare | Hybrid"
      → CRITICAL: when a brand / advertiser / casino SEEKS traffic and offers
        CPA / RevShare TO webmasters — the CPA is the PRICE they pay for traffic.
        This is SEEKING_TRAFFIC, not TRAFFIC_PROVIDER.

  (d) Affiliate network or partner program seeking traffic sources.

── NOT a traffic lead ───────────────────────────────────────────────────────
  ✗ Databases or personal data without traffic context
    ("igaming data", "база депозиторов", "fresh data", "PREMIUM TARGETED DATA PROVIDER"
     if it's purely about selling contact lists/databases, not traffic)
  ✗ Social media account buying/selling WITHOUT traffic
    ("куплю FB аккаунты", "фармленные акки", "есть БМ")
  ✗ Payment / PSP context: "платёжный трафик", "payment traffic uptime"
  ✗ Technical IT traffic: DDoS, network bandwidth, bot traffic, API load
  ✗ Job postings: "ищем media buyer в штат", "BDM — Remote, фикс + %",
    "Business Development Manager — Crypto Media Buying Team, Remote"
  ✗ General discussion with no buy/sell intent: "трафик растёт"
  ✗ Recovery / scam: "recovery leads", "returned victims"\
"""

# ---------------------------------------------------------------------------
# Full LLM system prompt
# ---------------------------------------------------------------------------

_TRAFFIC_SYSTEM_TEMPLATE = """\
You are a senior analyst of the affiliate traffic market (arbitrage, webmasters, media buying).

You are given a message from a Telegram chat. Apply the following definition:

{traffic_definition}

Return ONLY valid JSON (no markdown, no explanations):

{{
  "is_lead": true | false,

  "lead_type": "traffic_provider" | "seeking_traffic" | null,
  // traffic_provider — sells traffic / looks for offer / advertiser
  // seeking_traffic  — buys traffic / looks for webmasters / arbitrageurs
  // null             — if is_lead = false

  "confidence": "high" | "medium" | "low",

  "vertical": ["igaming" | "casino" | "sportsbook" | "forex" | "crypto" | "adult" | "nutra" | "dating" | "trading" | "mlm" | "subscription" | "call_centers" | "resale" | "other" | "unknown", ...],
  // all mentioned niches or []

  "traffic_source": [],
  // use EXACTLY one of the values below (or several if explicitly mentioned):
  // Paid Social        — paid ads on social networks (FB, TikTok, Instagram, Twitter, etc.)
  // Paid Search        — Google, Bing and other search engines
  // Native Ads         — native advertising
  // Push Ads           — push notifications
  // Popunder / Popup   — popunder / popup ads
  // In-App Ads         — in-application advertising
  // Programmatic       — DSP, programmatic display
  // SEO                — content sites, doorways, aggregators, organic
  // Organic Social     — Telegram channels, YouTube, TikTok without ad budget
  // Influencer         — bloggers, influencers
  // Messaging          — Email, SMS, WhatsApp, Viber broadcasts
  // Call Center        — live calls, warm / hot leads by phone
  // Database           — selling contact databases / lead lists
  // Co-reg             — co-registration traffic
  // CPA Network        — traffic rebuy through affiliate network
  // If source is not mentioned — [].
  // If mentioned but not in the list — write as-is.

  "geo": ["US", "UK", "DE", "LATAM", ...],
  // geographic targets if mentioned, or []

  "pricing_model": ["CPA", "CPL", "RevShare", "flat", "hybrid", ...],
  // payment models if mentioned, or []

  "evidence_quote": "verbatim quote from the text supporting the decision",

  "rationale": "one sentence — brief reasoning"
}}

Rules:
- ``vertical`` guidance:
  - "nutra" — supplements, weight loss, peptides, health products.
  - "dating" — dating/matchmaking services.
  - "trading" — binary options, CFD, financial trading platforms (distinct from "forex" which covers currency exchange/FX brokers).
  - "mlm" — multi-level marketing / network marketing.
  - "subscription" — recurring-billing apps/services (not gambling/dating).
  - "call_centers" — outbound/inbound phone sales operations, telemarketing teams, or traffic generated by them. Often abbreviated as "кц", "сс", "колл-центр", "трафик кц".
  - "other" — a vertical is mentioned or clearly implied, but does not match any value in the list above.
  - "unknown" — the message contains NO industry or vertical signal whatsoever; the business domain is completely absent. NEVER use "unknown" if any industry hint is present — use "other" instead.
- Do not invent anything — only what is explicitly stated or obviously implied in the message.
- vertical, traffic_source, geo, pricing_model — always arrays (can be []).
- If the message announces available traffic + names niches / GEOs / sources —
  set is_lead = true, TRAFFIC_PROVIDER, even without the words "sell" or "продаю".
- If the message seeks traffic / webmasters / arbitrageurs — set is_lead = true,
  SEEKING_TRAFFIC, even without the word "buy" or "куплю".
- Set is_lead = false ONLY if the message is clearly out of scope: technical traffic,
  HR job posting, database sales without traffic context, PSP / payment processing context.\
"""

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

_TRAFFIC_JUDGE_SYSTEM = """\
You are an independent reviewer of affiliate traffic classification.

You are given a message and the extractor's decision. Your task:
issue the FINAL verdict: TRAFFIC_PROVIDER, TRAFFIC_LEAD, or MISTAKE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAFFIC_PROVIDER — the author SELLS traffic / leads or seeks an offer / brand:
  • Has traffic and offers it: "we sell live crypto traffic", "есть трафик на gambling"
  • Announces traffic availability WITHOUT the word "sell" — still TRAFFIC_PROVIDER:
    "Crypto Traffic Available", "SERVICES AVAILABLE: Fx/crypto traffic available",
    "have traffic available from sports websites", "Gambling SEO Traffic; Geo: All"
    → Announcing available traffic = offering it = TRAFFIC_PROVIDER
  • Media buying team / arbitrage team presenting their traffic as a product:
    "media buying team delivering CPL / CRG", "iGaming traffic team, work with direct advertisers"
  • Sells ready FTDs / live leads: "LIVE READY FTDs, prepayment per FTD"
  • Seeks offer / advertiser / brand / network FOR their own traffic:
    "LOOKING FOR BRANDS, NETWORKS — Crypto/Forex Traffic Available" → has traffic, seeks brand
    "ищу оффер под трафик", "looking for direct brands who work with crypto traffic"
  • Russian traffic seller slang:
    "залью трафика" / "слить трафик" / "лить трафик" = drive/sell traffic → TRAFFIC_PROVIDER
    "ищу оффера / залью трафика" = looking for where to sell traffic → TRAFFIC_PROVIDER

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAFFIC_LEAD — the author BUYS traffic or seeks traffic sources:
  • Seeks or buys traffic / leads (even a short message is enough):
    "нужен трафик", "у кого есть трафик на US?", "ищу арбитражника"
  • Seeks webmasters, arbitrageurs, media buyers:
    "ищем партнеров с трафиком", "looking for affiliates / webmasters"
  • Direct advertiser / brand / casino / affiliate network SEEKING traffic
    and offering payment terms (CPA / RevShare / Hybrid) TO webmasters:
    "Мы прямой рекламодатель, ищем трафик по UZ, CPA и RevShare от 50%"
    "Прямой рекл Gambling, ищем трафик на Tier-1, RevShare / CPA / Hybrid"
    "Direct casino advertiser looking for HQ live casino traffic"
    "Шукаємо трафік із GEO Tier 1 та Tier 2. CPA, RevShare або Hybrid"

  ⚠️ CRITICAL — COMMON MISTAKE:
  When an advertiser / casino / affiliate network SEEKS traffic and mentions CPA / RevShare —
  that CPA is the PRICE they pay webmasters for traffic, NOT a sign that they sell traffic.
  Such an author is a TRAFFIC BUYER → TRAFFIC_LEAD.
  Do NOT confuse "offers CPA to webmasters" with "sells traffic".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Say MISTAKE only if one of the following clearly applies:
  1. Traffic in PSP / payment processing context — "платёжный трафик", "payment traffic uptime"
  2. Technical / network traffic with no affiliate intent — DDoS, bandwidth, API load
  3. Buying / selling accounts without traffic — "куплю FB акки", "фармленные акки"
  4. Database / leaked contacts — "база депозиторов", "igaming data for sale"
  5. Job posting or CV — "ищем media buyer в штат", "BDM — Remote, фикс + %"
  6. Platform self-promotion without seeking webmasters / traffic — "наша партнёрка принимает трафик"
  7. Completely impossible to determine intent — too little information

When in doubt — CONFIRM the extractor's decision.
Do NOT say MISTAKE just because the message is short or informal.

Return ONLY valid JSON (no markdown):
{{
  "verdict": "TRAFFIC_PROVIDER" | "TRAFFIC_LEAD" | "MISTAKE",
  "judge_reason": "one sentence — why this verdict"
}}\
"""