"""Affiliate traffic domain data.

Two pieces:
  * ``gazetteer`` — canonical terms of the affiliate traffic market:
    niches, traffic sources, pricing models, roles, affiliate networks.
    Used by the gazetteer scanner to match terms regardless of frequency.
  * ``phrase_seed_block`` — rotating-categories prompt fragment that
    seeds Stage 1b's LLM-assisted phrase extractor with examples of
    jargon to hunt for (RU / UK / EN).

Both are content, not logic. Swap this module to retarget the pipeline
at a different vertical. Do NOT edit the pipeline internals.
"""

from __future__ import annotations


gazetteer: list[dict] = [

    # ── Ніші / вертикалі ──────────────────────────────────────────────────────
    {"canonical": "gambling",  "forms": ["gambling", "гемблинг", "гамблинг"], "category": "niche"},
    {"canonical": "casino",    "forms": ["casino", "казино", "казіно"],        "category": "niche"},
    {"canonical": "betting",   "forms": ["betting", "беттинг", "ставки"],      "category": "niche"},
    {"canonical": "igaming",   "forms": ["igaming", "iGaming", "i-gaming"],    "category": "niche"},
    {"canonical": "crypto",    "forms": ["crypto", "крипто", "криптовалюта"],  "category": "niche"},
    {"canonical": "forex",     "forms": ["forex", "форекс", "FX"],             "category": "niche"},
    {"canonical": "nutra",     "forms": ["nutra", "нутра", "нутра-"],          "category": "niche"},
    {"canonical": "adult",     "forms": ["adult", "адалт", "18+"],             "category": "niche"},
    {"canonical": "dating",    "forms": ["dating", "дейтинг"],                 "category": "niche"},
    {"canonical": "sweepstakes", "forms": ["sweepstakes", "свипстейкс", "sweeps"], "category": "niche"},
    {"canonical": "e-commerce",  "forms": ["ecom", "е-ком", "e-commerce"],     "category": "niche"},
    {"canonical": "binary",    "forms": ["binary", "бинарные опционы"],        "category": "niche"},

    # ── Джерела трафіку ───────────────────────────────────────────────────────
    {"canonical": "Facebook Ads",  "forms": ["Facebook", "FB", "ФБ", "фейсбук", "Facebook Ads"], "category": "traffic_source"},
    {"canonical": "TikTok Ads",    "forms": ["TikTok", "тикток", "TikTok Ads"],                   "category": "traffic_source"},
    {"canonical": "Google Ads",    "forms": ["Google Ads", "Google", "гугл", "UAC"],              "category": "traffic_source"},
    {"canonical": "Instagram",     "forms": ["Instagram", "инстаграм", "инста"],                   "category": "traffic_source"},
    {"canonical": "YouTube",       "forms": ["YouTube", "ютуб", "YT"],                             "category": "traffic_source"},
    {"canonical": "Telegram",      "forms": ["Telegram", "телеграм", "TG", "ТГ"],                  "category": "traffic_source"},
    {"canonical": "push traffic",  "forms": ["push", "пуш", "push traffic", "пуш-трафик"],        "category": "traffic_source"},
    {"canonical": "native ads",    "forms": ["native", "нативка", "нативная", "native ads"],       "category": "traffic_source"},
    {"canonical": "popunder",      "forms": ["pop", "popunder", "popup", "поп-андер"],             "category": "traffic_source"},
    {"canonical": "SEO",           "forms": ["SEO", "сео", "organic", "органика"],                 "category": "traffic_source"},
    {"canonical": "in-app",        "forms": ["in-app", "inapp", "in app"],                         "category": "traffic_source"},
    {"canonical": "email marketing", "forms": ["email", "email рассылка", "мейл"],                 "category": "traffic_source"},
    {"canonical": "SMS marketing", "forms": ["SMS", "смс рассылка", "SMS blast"],                  "category": "traffic_source"},
    {"canonical": "DSP",           "forms": ["DSP", "programmatic", "программатик"],               "category": "traffic_source"},
    {"canonical": "Snapchat Ads",  "forms": ["Snapchat", "снапчат"],                               "category": "traffic_source"},
    {"canonical": "Twitter/X Ads", "forms": ["Twitter", "X Ads", "твиттер"],                       "category": "traffic_source"},

    # ── Моделі оплати ─────────────────────────────────────────────────────────
    {"canonical": "CPA",      "forms": ["CPA", "КПА", "cost per action"],       "category": "pricing_model"},
    {"canonical": "CPL",      "forms": ["CPL", "cost per lead"],                 "category": "pricing_model"},
    {"canonical": "RevShare", "forms": ["RevShare", "ревшара", "rev share", "revenue share", "RS"], "category": "pricing_model"},
    {"canonical": "FTD",      "forms": ["FTD", "ФТД", "first time deposit", "первый депозит"], "category": "pricing_model"},
    {"canonical": "CPM",      "forms": ["CPM", "cost per mille"],                "category": "pricing_model"},
    {"canonical": "CPC",      "forms": ["CPC", "cost per click"],                "category": "pricing_model"},
    {"canonical": "hybrid",   "forms": ["hybrid", "гибрид", "hybrid deal"],     "category": "pricing_model"},
    {"canonical": "flat",     "forms": ["flat", "фикс", "flat fee", "фиксированная"], "category": "pricing_model"},
    {"canonical": "baseline", "forms": ["baseline", "базовая ставка"],           "category": "pricing_model"},

    # ── Ролі учасників ────────────────────────────────────────────────────────
    {"canonical": "арбитражник",  "forms": ["арбитражник", "арбитражники", "арбитражнік"], "category": "role"},
    {"canonical": "вебмастер",    "forms": ["вебмастер", "веб-мастер", "webmaster"],        "category": "role"},
    {"canonical": "медиабаер",    "forms": ["медиабаер", "медиабайер", "медіабаєр", "media buyer", "медиабай"], "category": "role"},
    {"canonical": "арбитраж",     "forms": ["арбитраж", "арбітраж", "affiliate arbitrage"], "category": "role"},
    {"canonical": "тимлид",       "forms": ["тимлид", "тімлід", "team lead"],               "category": "role"},
    {"canonical": "affiliate",    "forms": ["affiliate", "аффилиат", "афіліат", "партнёр"], "category": "role"},

    # ── Партнерки / мережі ────────────────────────────────────────────────────
    {"canonical": "CPA Network",  "forms": ["CPA network", "партнерка", "партнёрка", "партнерська мережа", "affiliate network"], "category": "affiliate_network"},
    {"canonical": "Admitad",      "forms": ["Admitad"],                          "category": "affiliate_network"},
    {"canonical": "Affise",       "forms": ["Affise"],                           "category": "affiliate_network"},
    {"canonical": "ClickDealer",  "forms": ["ClickDealer"],                      "category": "affiliate_network"},
    {"canonical": "Leadbit",      "forms": ["Leadbit"],                          "category": "affiliate_network"},
    {"canonical": "Yellana",      "forms": ["Yellana"],                          "category": "affiliate_network"},
    {"canonical": "MyLead",       "forms": ["MyLead"],                           "category": "affiliate_network"},
    {"canonical": "TrafficStars", "forms": ["TrafficStars", "Traffic Stars"],    "category": "affiliate_network"},
    {"canonical": "PropellerAds", "forms": ["PropellerAds", "Propeller Ads"],   "category": "affiliate_network"},

    # ── Трафіковий жаргон ─────────────────────────────────────────────────────
    {"canonical": "лить трафик",  "forms": ["лить", "льём", "льет", "лити", "заливать", "залив", "слив", "отлив"], "category": "traffic_slang"},
    {"canonical": "оффер",        "forms": ["оффер", "офер", "оффера", "офферы", "offer"],    "category": "traffic_slang"},
    {"canonical": "лид",          "forms": ["лид", "лиды", "лід", "ліди", "lead", "leads"],   "category": "traffic_slang"},
    {"canonical": "лидген",       "forms": ["лидген", "лідген", "лидогенерация", "lead gen", "lead generation"], "category": "traffic_slang"},
    {"canonical": "конверт",      "forms": ["конверт", "конверсия", "конверсія", "conversion", "CR"], "category": "traffic_slang"},
    {"canonical": "апрув",        "forms": ["апрув", "аппрув", "апрув рейт", "approve rate"],  "category": "traffic_slang"},
    {"canonical": "крео",         "forms": ["крео", "креатив", "крео", "creative", "creatives"], "category": "traffic_slang"},
    {"canonical": "трафик",       "forms": ["трафик", "трафік", "traffic"],                    "category": "traffic_slang"},
    {"canonical": "ГЕО",          "forms": ["ГЕО", "гео", "GEO", "геолокация"],               "category": "traffic_slang"},
    {"canonical": "профит",       "forms": ["профит", "профіт", "profit", "профитный"],       "category": "traffic_slang"},
    {"canonical": "ROI",          "forms": ["ROI", "ROAS", "окупаемость"],                     "category": "traffic_slang"},
    {"canonical": "аккаунт",      "forms": ["акк", "аккаунт", "акаунт", "account", "BM"],     "category": "traffic_slang"},
    {"canonical": "баинг",        "forms": ["баинг", "байинг", "buying", "медиабай"],          "category": "traffic_slang"},
    {"canonical": "клоакинг",     "forms": ["клоакинг", "клоака", "cloaking", "клоак"],        "category": "traffic_slang"},
    {"canonical": "белая страница", "forms": ["белая", "вайт", "white page", "whitepage"],    "category": "traffic_slang"},
    {"canonical": "прелэндинг",   "forms": ["прелэнд", "прелендинг", "pre-lander", "prelander", "прела"], "category": "traffic_slang"},
    {"canonical": "RON",          "forms": ["RON", "run of network"],                          "category": "traffic_slang"},
    {"canonical": "таргет",       "forms": ["таргет", "таргетинг", "targeting", "targeted"],  "category": "traffic_slang"},
    {"canonical": "бурж",         "forms": ["бурж", "буржунет", "tier-1", "tier1", "Tier 1"], "category": "traffic_slang"},

    # ── Технічний стек арбітражника ───────────────────────────────────────────
    {"canonical": "трекер",  "forms": ["трекер", "tracker", "трекинг", "Keitaro", "Binom", "Voluum"], "category": "tech_stack"},
    {"canonical": "spy tool", "forms": ["spy", "спай", "AdSpy", "BigSpy", "AdPlexity"], "category": "tech_stack"},
    {"canonical": "антидетект", "forms": ["антидетект", "anti-detect", "Dolphin", "Octo Browser", "AdsPower"], "category": "tech_stack"},
    {"canonical": "прокси",  "forms": ["прокси", "proxy", "резидентные прокси", "residential proxy"], "category": "tech_stack"},
]


phrase_seed_block: str = """\
The chat contains messages in Russian, Ukrainian, and English — often mixed.
Extract affiliate-traffic industry terms in ANY of these languages.

Categories of terms to catch (extract anything from this domain that appears):

  Seller-intent phrases (арбитражник / вебмастер — у кого ЕСТЬ трафик):
    Russian:
      "есть трафик на гемблинг" (have gambling traffic),
      "продаю трафик" (selling traffic),
      "ищу оффер под FB трафик" (looking for offer for FB traffic),
      "сливаю на казино" (driving to casino),
      "лью трафик" (driving traffic),
      "гоню трафик" (pushing traffic),
      "арбитражник, работаю с нутрой" (arbitrageur, work with nutra),
      "вебмастер, есть гемблинг трафик" (webmaster, have gambling traffic),
      "ищу партнерку под крипто" (looking for affiliate program for crypto),
      "отлив на беттинг" (driving to betting),
      "нужен оффер под адалт трафик" (need offer for adult traffic)
    Ukrainian:
      "є трафік на казино" (have casino traffic),
      "шукаю оффер під гемблінг" (looking for gambling offer),
      "продаю ліди" (selling leads)
    English:
      "have gambling traffic", "selling traffic", "looking for offer",
      "need advertiser for my traffic", "have FB traffic for gaming",
      "webmaster seeking CPA deal", "media buyer looking for offer"

  Buyer-intent phrases (рекламодатель / партнёрка — кто ИЩЕТ трафик):
    Russian:
      "ищу вебмастеров под гемблинг" (looking for webmasters for gambling),
      "нужны арбитражники" (need arbitrageurs),
      "принимаю трафик на казино" (accepting casino traffic),
      "ищем медиабаеров" (looking for media buyers),
      "набираем в команду" (recruiting to team),
      "работаем с вебмастерами" (work with webmasters),
      "CPA оффер на нутру" (CPA offer for nutra),
      "куплю трафик на форекс" (buying forex traffic)
    English:
      "looking for webmasters", "seeking affiliates", "buying traffic",
      "CPA offer for gambling", "RevShare deal", "need media buyers",
      "affiliate program for crypto", "partner with us"

  Pricing models and deal structures:
    Russian:
      "работаю по CPA" (work on CPA basis),
      "ревшара 40%" (40% revenue share),
      "оплата за FTD" (payment per first deposit),
      "CPL схема" (CPL scheme),
      "гибрид CPA+RevShare" (CPA+RevShare hybrid),
      "фикс за лид" (flat per lead)
    English:
      "CPA deal", "RevShare offer", "FTD payout", "hybrid model",
      "flat rate", "CPL deal", "cost per acquisition"

  Traffic sources (when mentioned in deal context):
    Russian:
      "FB трафик" (Facebook traffic),
      "гугл трафик" (Google traffic),
      "SEO органика" (SEO organic),
      "пуш трафик" (push traffic),
      "нативная реклама" (native ads),
      "тикток трафик" (TikTok traffic),
      "in-app трафик" (in-app traffic),
      "телеграм канал" (Telegram channel traffic)
    English:
      "Facebook traffic", "Google Ads traffic", "TikTok traffic",
      "push traffic", "native traffic", "SEO traffic", "organic traffic",
      "pop traffic", "in-app traffic", "email traffic"

  Affiliate / arbitrage slang (Russian / mixed):
    "лить" / "льем" / "залить" (to drive traffic),
    "оффер" (offer from advertiser),
    "лид" / "лиды" (lead/s),
    "лидген" (lead generation),
    "крео" (creative / banner),
    "ГЕО" (geographic target),
    "апрув" / "апрув рейт" (approval rate),
    "конверт" (conversion),
    "ROI" / "ROAS" (return on investment),
    "клоакинг" (cloaking),
    "прелэнд" / "прелендинг" (pre-lander page),
    "белая страница" (white page),
    "антидетект" (anti-detect browser),
    "трекер" (traffic tracker — Keitaro, Binom),
    "бурж" (tier-1 / Western markets),
    "баинг" / "медиабай" (media buying),
    "тимлид" (team lead of buyers),
    "таргет" (paid targeting)

  Negatives to skip — do NOT extract these patterns:
    "платежный трафик" / "payment traffic" (PSP context, not affiliate),
    "транзакционный трафик" (transaction traffic — payments),
    "сетевой трафик" (network/IT traffic),
    "DDoS" (attack traffic),
    "куплю FB аккаунты" (account purchasing — different vertical),
    "продаю базу" / "igaming data" (database sales — not traffic)
"""
