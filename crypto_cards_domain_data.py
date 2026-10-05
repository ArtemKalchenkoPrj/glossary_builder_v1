"""Crypto Cards domain data.

Two pieces:
  * ``gazetteer`` — canonical terms of the crypto card market:
    card types, crypto currencies, KYC/compliance terms, card service
    terminology, known provider brands, roles, and card-specific slang.
    Used by the gazetteer scanner to match terms regardless of frequency.
  * ``phrase_seed_block`` — rotating-categories prompt fragment that
    seeds Stage 1b's LLM-assisted phrase extractor with examples of
    jargon to hunt for (RU / EN).

Both are content, not logic. Swap this module to retarget the pipeline
at a different vertical. Do NOT edit the pipeline internals.
"""

from __future__ import annotations


gazetteer: list[dict] = [

    # ── Card types ────────────────────────────────────────────────────────────
    {"canonical": "crypto card",      "forms": ["crypto card", "crypto cards", "криптокарта", "крипто-карта", "крипто карта"], "category": "card_type"},
    {"canonical": "virtual card",     "forms": ["virtual card", "virtual cards", "виртуальная карта", "віртуальна картка", "vcc"],    "category": "card_type"},
    {"canonical": "physical card",    "forms": ["physical card", "physical cards", "физическая карта", "пластик", "plastic card"],    "category": "card_type"},
    {"canonical": "prepaid card",     "forms": ["prepaid card", "prepaid", "предоплаченная карта", "prepayd"],                        "category": "card_type"},
    {"canonical": "reloadable card",  "forms": ["reloadable card", "reloadable crypto card", "reloadable", "пополняемая карта"],      "category": "card_type"},
    {"canonical": "anonymous card",   "forms": ["anonymous card", "анонимная карта", "анонімна картка", "anonymous crypto card"],     "category": "card_type"},
    {"canonical": "зп карты",         "forms": ["зп карты", "зп-карты", "зарплатные карты", "salary card", "salary cards", "payroll card"], "category": "card_type"},
    {"canonical": "corporate card",   "forms": ["corporate card", "корпоративная карта", "корпоративні картки", "b2b card"],         "category": "card_type"},

    # ── Crypto currencies (as card funding source) ────────────────────────────
    {"canonical": "USDT",  "forms": ["USDT", "tether", "юсдт"],          "category": "currency"},
    {"canonical": "USDC",  "forms": ["USDC", "usd coin"],                 "category": "currency"},
    {"canonical": "BTC",   "forms": ["BTC", "bitcoin", "биткоин", "биток"], "category": "currency"},
    {"canonical": "ETH",   "forms": ["ETH", "ethereum", "эфир", "ефір"], "category": "currency"},
    {"canonical": "TRX",   "forms": ["TRX", "tron", "трон"],              "category": "currency"},
    {"canonical": "XRP",   "forms": ["XRP", "ripple"],                    "category": "currency"},
    {"canonical": "LTC",   "forms": ["LTC", "litecoin", "лайткоин"],      "category": "currency"},

    # ── KYC / compliance terms ─────────────────────────────────────────────────
    {"canonical": "no KYC",    "forms": ["no kyc", "nokyc", "no-kyc", "без kуc", "без kyc", "без верификации", "без верифікації", "без верифiкацiї"], "category": "kyc"},
    {"canonical": "anonymous", "forms": ["anonymous", "анонимный", "анонімний", "100% anonymous"],                                "category": "kyc"},
    {"canonical": "no verification", "forms": ["no verification", "без проверки", "без документов", "no docs"],                   "category": "kyc"},
    {"canonical": "SOF",       "forms": ["SOF", "source of funds", "источник средств", "структурирование SOF"],                   "category": "kyc"},
    {"canonical": "CRS",       "forms": ["CRS", "common reporting standard", "уход от CRS"],                                      "category": "kyc"},
    {"canonical": "AML",       "forms": ["AML", "anti-money laundering", "антиотмывочный"],                                       "category": "kyc"},

    # ── Card service / technical terms ────────────────────────────────────────
    {"canonical": "card issuing",  "forms": ["card issuing", "card issuer", "issuing", "выпуск карт", "эмиссия карт", "емісія карток"], "category": "card_service"},
    {"canonical": "BIN",           "forms": ["BIN", "BINs", "multiple bins", "бин", "бины", "bank identification number"],             "category": "card_service"},
    {"canonical": "BIN sponsor",   "forms": ["bin sponsor", "bin sponsorship", "BIN спонсорство", "бин спонсор"],                      "category": "card_service"},
    {"canonical": "white-label",   "forms": ["white label", "white-label", "white-label card", "вайтлейбл", "white label card solution"], "category": "card_service"},
    {"canonical": "card program",  "forms": ["card program", "card programme", "карточная программа", "картковa програма"],            "category": "card_service"},
    {"canonical": "API delivery",  "forms": ["api delivery", "api integration", "delivery api", "апи интеграция", "API"],              "category": "card_service"},
    {"canonical": "Apple Pay",     "forms": ["apple pay", "эппл пей", "apple pay integration"],                                        "category": "card_service"},
    {"canonical": "Google Pay",    "forms": ["google pay", "гугл пей", "google pay integration"],                                      "category": "card_service"},
    {"canonical": "issuing countries", "forms": ["issuing countries", "issuing country", "страны выпуска"],                            "category": "card_service"},
    {"canonical": "card management app", "forms": ["management app", "card app", "приложение для карт"],                              "category": "card_service"},
    {"canonical": "load from wallet",  "forms": ["loaded from wallet", "load from wallet", "load from crypto", "загрузка с кошелька", "пополнение с кошелька", "загружается с"], "category": "card_service"},

    # ── Business model / deal terms ────────────────────────────────────────────
    {"canonical": "resell",    "forms": ["resell", "reseller", "ресейл", "ресейлер", "ресейлить", "перепродажа карт"], "category": "deal_type"},
    {"canonical": "B2B",       "forms": ["b2b", "B2B", "бизнес для бизнеса", "корпоративный клиент"],                "category": "deal_type"},
    {"canonical": "OTC",       "forms": ["OTC", "over the counter", "внебиржевой", "ОТС"],                            "category": "deal_type"},
    {"canonical": "partner program", "forms": ["partner program", "партнёрская программа", "партнерка по картам"],    "category": "deal_type"},
    {"canonical": "wholesale", "forms": ["wholesale", "оптовая поставка", "оптом", "bulk cards"],                    "category": "deal_type"},
    {"canonical": "iGaming VIP", "forms": ["igaming vip", "igaming vip cards", "vip cards", "vip игроки"],          "category": "deal_type"},

    # ── Known provider brands (seeded from real chat examples) ───────────────
    {"canonical": "LEVEL",        "forms": ["LEVEL", "level cards", "level nokyc"],   "category": "provider_brand"},
    {"canonical": "CLUBCARD",     "forms": ["CLUBCARD", "club card nokyc"],            "category": "provider_brand"},
    {"canonical": "NOTAMPAY",     "forms": ["NOTAMPAY", "notam pay"],                 "category": "provider_brand"},
    {"canonical": "Arquen Finance", "forms": ["Arquen", "Arquen Finance"],             "category": "provider_brand"},

    # ── Roles ─────────────────────────────────────────────────────────────────
    {"canonical": "card provider", "forms": ["card provider", "card issuer", "провайдер карт", "поставщик карт", "постачальник карток"], "category": "role"},
    {"canonical": "card seeker",   "forms": ["ищу карту", "looking for card", "нужна карта", "нужны карты", "need card"],               "category": "role"},
    {"canonical": "CBDO",          "forms": ["CBDO", "chief business development"],                                                      "category": "role"},
    {"canonical": "media buyer",   "forms": ["media buyer", "медиабаер", "медиабайер", "медіабаєр"],                                     "category": "role"},

    # ── Card slang (RU) ───────────────────────────────────────────────────────
    {"canonical": "пластик под крипту", "forms": ["пластик под крипту", "пластик с крипты", "карта под крипту", "карта с криптой"], "category": "slang"},
    {"canonical": "крипто-пластик",     "forms": ["крипто пластик", "крипто-пластик", "crypto plastic"],                            "category": "slang"},
    {"canonical": "карта для крипты",   "forms": ["карта для крипты", "карта для криптовалюты", "card for crypto"],                 "category": "slang"},
    {"canonical": "выводить на карту",  "forms": ["вывести на карту крипту", "вывод крипты на карту", "withdraw crypto to card"],   "category": "slang"},
    {"canonical": "тратить USDT",       "forms": ["тратить usdt", "тратить крипту", "spend usdt", "spend crypto"],                  "category": "slang"},
]


phrase_seed_block: str = """\
The chat contains messages in Russian and English — often mixed.
Extract crypto card industry terms in ANY of these languages.

Categories of terms to catch (extract anything from this domain that appears):

  Provider-intent phrases (someone who ISSUES or SUPPLIES crypto cards):
    Russian:
      "выпускаем криптокарты" (we issue crypto cards),
      "занимаемся выдачей карт" (we issue cards),
      "эмиссия карт под крипту" (card issuance loaded with crypto),
      "карта загружается с USDT" (card loaded from USDT),
      "выпуск карт без KYC" (card issuance without KYC),
      "анонимные карты для бизнеса" (anonymous cards for business),
      "зп карты под крипту" (payroll cards loaded with crypto),
      "виртуальные карты с пополнением USDT" (virtual cards with USDT top-up),
      "белый лейбл карточное решение" (white-label card solution),
      "карты для медиабаинга" (cards for media buying),
      "работаем с b2b, выдаём карты" (work with B2B, issue cards),
      "реализуем криптокарты" (we supply crypto cards)
    English:
      "reloadable crypto cards", "virtual cards funded by USDT",
      "anonymous card no KYC", "card issuing for B2B",
      "white-label crypto card solution", "cards loaded from wallet",
      "physical and virtual cards in crypto", "multiple BINs available",
      "card program for iGaming", "API card delivery",
      "we issue crypto cards", "nokyc card provider"

  Seeker-intent phrases (someone who NEEDS crypto cards):
    Russian:
      "ищу криптокарту" (looking for a crypto card),
      "нужна анонимная карта" (need an anonymous card),
      "нужны провайдеры крипто карт" (need crypto card providers),
      "готовы ресейлить крипто карты" (ready to resell crypto cards),
      "ищем поставщика карт под крипту" (looking for crypto card supplier),
      "хочу карту для трат USDT" (want a card to spend USDT),
      "нужен IBAN и крипто карта" (need IBAN and crypto card),
      "интересует партнёрка по картам" (interested in card partnership),
      "ищу карту без верификации" (looking for card without verification),
      "ищу решение для оплаты криптой" (looking for crypto payment solution)
    English:
      "looking for anonymous crypto card", "need a no-KYC card",
      "looking for crypto card solution", "DM if you have crypto cards",
      "need virtual card funded by crypto", "looking for card provider",
      "want to resell crypto cards", "where to get nokyc card",
      "looking for card to spend USDT/BTC", "need B2B crypto card"

  Card product and technical terms:
    Russian:
      "карта без KYC" (card without KYC),
      "пополнение с кошелька" (top-up from wallet),
      "криптокарта для медиабаинга" (crypto card for media buying),
      "выпуск пластика" (card plastic issuance),
      "интеграция Apple Pay / Google Pay" (Apple/Google Pay integration),
      "эмиссионный банк" (issuing bank),
      "BIN-спонсор" (BIN sponsor)
    English:
      "BIN", "multiple BINs", "BIN sponsor", "card issuing",
      "API delivery", "white label card", "card program",
      "Apple Pay integration", "Google Pay", "issuing countries",
      "USDT funding", "USDC funding", "loaded from wallet",
      "management app", "card top-up", "anonymous prepaid card",
      "reloadable card", "no-KYC card", "nokyc",
      "iGaming VIP cards", "salary cards in crypto", "OTC cards",
      "SOF structuring", "CRS avoidance"

  Negatives to skip — do NOT extract these patterns:
    "вывод на карту от биржи" (P2P exchanger withdrawing to external card),
    "покупка USDT через СБП" (buying crypto via bank transfer — P2P),
    "p2p обмен" / "трейдер p2p" (P2P crypto trading, not card issuing),
    "flash USDT" / "flash BTC" (crypto scam tools, not cards),
    "payment gateway" / "платёжный шлюз" (PSP/gateway without card context),
    "эквайринг" (card acquiring for merchants — different from issuing),
    "chargeback" (fraud recovery, not cards),
    "обычная корпоративная карта" (ordinary corporate card, no crypto marker),
    "карта лояльности" (loyalty card — not a payment card),
    "вакансия" / "job opening" (recruitment posts)
"""
