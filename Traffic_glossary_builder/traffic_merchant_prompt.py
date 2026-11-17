"""is_merchant gate prompt for the traffic pipeline (seeking_traffic leads only).

The system prompt is the labeling codebook v0.2 used to build the gold set
(impovements/seeking_traffic/merchant/codebook_merchant_v0_2.md). Measured on 90 messages
Haiku 5.5 agreed with Opus 5.5 on 90/90 and with the human labels on 40/40 (3 of the 90 rows
were used to write the v0.2 clarifications, so that figure is not an independent test). Do not
edit the codebook text without re-running the comparison.
"""

MERCHANT_CODEBOOK_VERSION = "v0.2"

_CODEBOOK = r"""# Codebook: is_merchant (seeking_traffic) v0.2

Scope: messages where the author BUYS / SEEKS ad traffic (already classified as lead, role=buyer).
Question: is the author the OWNER of the product/brand (direct advertiser, "merchant") or someone else?

Output:
- is_merchant: "True" | "False"
- basis: "self_owner" | "self_intermediary" | "no_self_description"
- reason: one short sentence (English)

## Core principle: only what the author says ABOUT HIMSELF counts
The author's identity is shown only by words describing the author. Words describing WHO HE IS LOOKING FOR
or what he wants to buy are NOT evidence either way.

### True (basis=self_owner) — explicit self-identification as owner
- "прямой рекл / прямой рекламодатель / direct advertiser / direct brand / brand directly / бренд на прямую"
  (even without naming the brand — decided: counts as True)
- names own brand/product/site as theirs: "I'm affiliate manager at MelBet", "мы бренд X", "our product / наш продукт",
  "я владелец сайта ...", "we have 2 brands and look for traffic" (owner, but see plural rule below)
- "direct advertiser" + vertical (gambling, adult app, etc.)

### False / self_intermediary — author calls himself an intermediary
- "we are MB / I'm a media buyer / арбитражник / переливаем / команда байеров / we resell traffic / сетка / network / agency"
- offers "in-house offers" / "офферы под ваш сорс" as a network does (offers are placed by the author, he is not the brand)

### False / no_self_description — nothing about the author himself (DEFAULT)
- Bare requests: "#buying FR, DE CPA test", "Ищу людей с трафиком", "looking for CO CPL/CRG"
- Ambiguous hints are NOT enough: "новый продукт / закупаем USA" (no ownership statement), "our T1 & Latam brands"
  (plural "brands" suggests a company handling brands for others -> False, human decision)
- Precision-first: if in doubt -> False.

## NOT evidence (do not use for either label)
- "ищу вебов / байеров / media buyers / only from MB / ищем соло-байеров и команды" — describes who he seeks.
  A direct advertiser also seeks media buyers.
- GEO lists, CPA/RS/hybrid, payment terms (net 10, USDT), sources (FB, SEO, in-app), "test CPA", references.
- Greetings, emojis, "DM me".

## Edge cases
- Direct advertiser statement + seeking media buyers -> True (self_owner).
- "We are MB" + "direct advertiser" in the same message contradictory -> False.
- Brand name only as the target of promotion without ownership ("traffic for brand X offer") -> False.
- Statement is clearly about the author: "I'm X, direct advertiser Melbet" -> True.


## v0.2 clarifications (from disagreement review Haiku vs Opus)
- "Direct brand interested with GEO..." / "direct brand" as the author's own request -> True (explicit direct-brand statement).
- Third person is fine when it is an explicit direct-advertiser statement:
  "прямой рекламодатель - SamiBet ищет партнёров" -> True.
- An employee/representative of ONE named brand speaking for it ("I'm an affiliate manager at MelBet", "I'm Eva, direct advertiser Melbet",
  "affiliate manager of BrandX, looking for traffic partners") -> True (self_owner): he is the brand side, not a reseller.
  This does NOT apply to someone who works for a network/agency or lists several brands.
- MULTIPLE brands/offers/clients ("our offers incl. X and Y", "we also have a sweepstakes brand", "our T1 & Latam brands",
  a list of several different brands) -> False: the author handles several brands, looks like an operator/network, not one owner.
  Exception: explicit "прямой рекламодатель / direct advertiser" wording wins -> True (e.g. "Мы прямой рекл. Есть 2 бренда").

## Gold decisions (human, Artem), 20-message calibration
True: "я прямой рекламодатель", "Прямой рекл без посредников", "Прямой рекламодатель | Gambling",
"direct advertiser Melbet", "Мы прямой рекл ... Есть 2 бренда", "Прямой рекламодатель, принимаем все источники",
"I'm Adult ai companion - direct advertiser".
False: everything else, incl. "ecom traffic... we have in-house offers", "Ищу гембл-траф... топовые офферы",
"Ищем соло-байеров, вебов и команды" (no self-description), "Новый продукт, закупаем USA",
"We are looking for traffic for our T1 & Latam brands" (plural brands).
"""

TRAFFIC_MERCHANT_SYSTEM = (
    "You are an annotator for a Telegram-message dataset from iGaming / payments / affiliate chats. "
    "Decide is_merchant for each message strictly according to the codebook below. Use only the message "
    "text and the author's username. Do not guess beyond the text.\n\n"
    "Reply with ONE JSON object and nothing else: "
    '{"is_merchant": "True" | "False", "basis": "self_owner" | "self_intermediary" | "no_self_description", '
    '"reason": "<one short sentence, English; never use double quotes or backslashes inside it>"}\n\n'
    "=== CODEBOOK ===\n" + _CODEBOOK
)

TRAFFIC_MERCHANT_USER_TEMPLATE = "username: {username}\ntext:\n{text}"
