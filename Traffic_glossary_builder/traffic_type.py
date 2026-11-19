"""traffic_type: what the author sells or looks for (live traffic / database / infra).

The extractor annotates every message with a ``traffic_type`` list (codebook v1.1,
impovements/seeking_traffic/traffic_type/gold/codebook_tt_v1.md). The list is collapsed with policy P2
and, in ``enforce`` mode, messages that offer ONLY a database or ONLY infrastructure are not leads:
they are never sent to the judge (is_lead=False, verdict=MISTAKE, so nothing is persisted or routed).

Mode (env ``TRAFFIC_TYPE_MODE``):
  off     (default) - the extractor prompt is NOT changed, nothing happens.
  shadow  - the extractor annotates and the result is logged ("[traffic_type] would_drop ..."), no message is dropped.
  enforce - database / infra messages are dropped.

Fail-open: a missing / invalid ``traffic_type`` never drops anything.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

TRAFFIC_TYPE_VALUES = ("live_traffic", "database", "infra", "unclear", "none")
TRAFFIC_TYPE_DROP = ("database", "infra")
_MODES = ("off", "shadow", "enforce")


def traffic_type_mode() -> str:
    """off | shadow | enforce (anything else -> off)."""
    value = os.environ.get("TRAFFIC_TYPE_MODE", "").strip().lower()
    return value if value in _MODES else "off"


# The text below is inserted after the traffic definition. It contains NO braces: the system template is
# a str.format() template where only {traffic_definition} / {vertical_rules} are placeholders.
TRAFFIC_TYPE_BLOCK = '''\
TRAFFIC TYPE (an annotation, independent of is_lead)
Besides the decision above, annotate WHAT the author sells or looks for in the field "traffic_type".
Decide is_lead EXACTLY as defined above; traffic_type does not change that decision.
The goal is an ANTI-FILTER: mark database / infra only on positive evidence. With no evidence either way
the message is most likely live traffic.

"traffic_type" is a non-empty list. Allowed items: "live_traffic", "database", "infra" (any combination, only if the
item is offered or sought AS A PRODUCT; a passing mention does not count, e.g. "we mail your ad to a base of 250 000
chats" is a promotion service = live_traffic, and "a clean base of players" of a brand looking for traffic is not a
database sale), OR ["unclear"] (cannot tell what kind of traffic; no markers either way, e.g. "looking for traffic on BR geo"),
OR ["none"] (not about traffic, databases or infrastructure: payment solutions, recruiting people, job search, black-hat SEO...).

live_traffic = a flow of users / dynamic advertising, regularly, for a result. Strong markers (any one is enough):
- media buying: MB, media buying, медиабаинг; "direct advertiser / direct brand looking for traffic";
- official ad platforms: Google Ads/PPC, Meta/Facebook/FB, TikTok (TT) + business center (БЦ), Telegram Ads, Taboola, Bigo,
  MediaGo, Smadex, DSP, InApp, push / in-page push / popunder (these cannot legally deliver a database);
- SEO / PPC / influencers / bloggers, посевы, арбитраж, "льем трафик" (SEO only when it is a traffic source, not a job title);
- pay-per-result models: CPA, CPL, CPI, RS/RevShare, CRG, hybrid, "по спенду";
- "live traffic", "real time", "lead generation" as a campaign service (targeted advertising, campaign optimization,
  a regular flow), visitors; the word LIVE next to FTD/leads
  ("LIVE READY FTDs", "live leads") = a flow;
- FTD/leads as a REGULAR flow under CPA/CPL; SMS/email mailing as a CAMPAIGN SERVICE for an offer.

database = a static set of contacts / data delivered once. Strong markers (any one is enough):
- "выгрузка" (export) in the context of leads/data; "selling data", "my data", "fresh database", "базы", "папки" (folders of data);
- "Ready FTD" / "Fresh FTD" / "FTD base" WITHOUT live wording, "bank leads", "WhatsApp leads", "phone leads",
  audience segments ("old age", "rich people", "car owners");
- time-stamps of when contacts were collected ("24h to 6 month", "a week / 3 months data");
- leaked or scraped data, bases for SMS / email mailings (the file itself, not the campaign);
- leads sold as a product paid ONCE ("продам лиды"), not a continuous flow;
- leads sold by QUANTITY or in packages ("5K / 10K / 20K leads", "quantity of leads", "lead packages") and/or with a
  DELIVERY FORMAT (Excel / CSV / spreadsheet / CRM, "data fields"), "samples" of leads, "verified / fresh / targeted
  leads" offered as a list, "leads database(s) / numbers"; this includes inbound-call, call-center, B2B/B2C leads.
A "lead" or "FTD" alone decides nothing: a file delivered once = database; a regular flow under CPA/CPL = live_traffic.
The words "leads", "conversions", "ready-to-buy customers", "call center" are NOT live_traffic markers: live_traffic
needs a traffic source / ad platform / campaign service or a pay-per-result model.

infra = infrastructure and tools around ads, not traffic itself, and it must be the MAIN product:
- turnkey / all-in-one infrastructure; cloaking (клоака / клокинг / cloak); ad accounts for sale or rent, proxies,
  anti-detect, VoIP / call-center setup, SIM, OTP solutions, SMS routes / gateways / sender panels, bulk-messaging
  software (Twilio, SMTP, DID/TFN), hosting, streaming APIs.
- Services ATTACHED to an advertising / traffic offer (account setup, support, campaign launch assistance, tracking setup)
  are NOT infra when the same message sells ads/traffic via platforms.

Precedence:
1. Strong database markers (export, time-stamps, ready FTD, "selling data", bank leads, folders) beat CPA/RS wording.
   BUT plain "data" + CPA/CPL with NO such marker is ["unclear"].
2. Strong live markers (official platforms, MB, SEO, influencers, push) beat weak "leads/data" wording.
3. If both a flow and a file are offered as products, list both, e.g. ["live_traffic","database"]
   ("live traffic and databases also available", "Live Traffic & Fresh Database", "traffic/data").
4. "Data traffic" / "live/data" with no further markers: ["unclear"].
5. No markers at all: ["unclear"].
'''

TRAFFIC_TYPE_FIELD = '''  "traffic_type": ["live_traffic" | "database" | "infra" | "unclear" | "none", ...],
  // WHAT is offered or sought, see the TRAFFIC TYPE block above. Independent of is_lead.

'''

_ANCHOR_DEF = "{traffic_definition}"
_ANCHOR_FIELD = '  "confidence": "high" | "medium" | "low",'


def apply_traffic_type_to_template(template: str) -> str:
    """Insert the traffic_type block and JSON field into the extractor system template.

    Returns the template unchanged if the anchors are not found (never breaks the extractor)."""
    if _ANCHOR_DEF not in template or _ANCHOR_FIELD not in template:
        return template
    out = template.replace(_ANCHOR_DEF, _ANCHOR_DEF + "\n\n" + TRAFFIC_TYPE_BLOCK, 1)
    return out.replace(_ANCHOR_FIELD, TRAFFIC_TYPE_FIELD + _ANCHOR_FIELD, 1)


def normalize_traffic_types(value) -> list[str]:
    """Extractor output -> list of known values (unknown items dropped). Empty list = no usable annotation."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        item = str(item).strip().lower()
        if item in TRAFFIC_TYPE_VALUES and item not in out:
            out.append(item)
    return out


def collapse_traffic_type(types: Iterable[str]) -> Optional[str]:
    """Policy P2: a list containing live_traffic stays in the live feed; otherwise database > infra.

    none -> none; empty / only unclear -> unclear; returns None for an empty list (no annotation)."""
    t = set(types or [])
    if not t:
        return None
    if "none" in t:
        return "none"
    if t == {"unclear"}:
        return "unclear"
    if "live_traffic" in t:
        return "live_traffic"
    if "database" in t:
        return "database"
    if "infra" in t:
        return "infra"
    return "unclear"
