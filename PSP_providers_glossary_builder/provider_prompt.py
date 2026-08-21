"""Prompts for the PSP Provider classifier.

Three components:
  _PROVIDER_STAGE1_SYSTEM     — cheap YES/NO micro-filter (is this a seller?)
  _PROVIDER_STAGE1_USER_TEMPLATE — user turn template for stage 1
  PROVIDER_DEFINITION         — what counts as a provider (injected into full LLM)
  _PROVIDER_SYSTEM_TEMPLATE   — full system prompt for classification + extraction
  _PROVIDER_USER_TEMPLATE     — user turn template for full classification

All prompts are seller-side mirrors of the buyer-side prompts in lead_extraction.py.
"""

from __future__ import annotations


# ── Stage 1: cheap YES/NO micro-filter ───────────────────────────────────────
# Runs between pre_filter and the full LLM call.
# One short completion; a NO short-circuits to is_provider=False with no
# expensive structured call.

# -20% позитивов на тестах
_PROVIDER_STAGE1_SYSTEM_v1 = """\
You are screening messages from a Telegram chat about high-risk payment processing \
(PSP, acquiring, iGaming, crypto, forex, adult).

Answer YES if the message is from someone SELLING, OFFERING, or PROVIDING \
a payment processing service.
Answer NO for everything else.

YES — the author is a payment provider:
- Direct offer: "предлагаем эквайринг", "we offer processing", "мы PSP"
- First-person coverage: "работаю по EU картам", "покрываю EU/UK", "закрываем гео"
- Provider metrics reported as own: "апрув 72%", "approve rate 75%+", "роллинг 8%"
- Spec-sheet format: "MCC 7995 | EU | Visa/MC | approve 75% | @contact"
- Ukrainian provider language: "пропоную еквайринг", "маємо покриття по ЄС"
- "Можем подключить", "можем закрыть EU high-risk"
- "Процессим гемблинг", "онбординг за 3 дні"

NO — the author is NOT a provider:
- Buyer looking for PSP: "ищу PSP", "нужен эквайринг", "looking for payment gateway"
- Market discussion / questions: "кто работал с X", "какой PSP лучше для EU"
- Complaint / blacklist: "кинули на сетлмент", "чарджбек вырос", "ЧС предупреждение"
- Developer integrating: "интегрирую шлюз в свой сайт", "настраиваю API для проекта"
- IBAN / bank account buyer: "нужен корпоративный счёт", "ищем IBAN для мерчанта"
- SMS, VoIP, ad agency, media buying, traffic — NOT payment processing
- Job posts, news, conference invites, general chat

Reply with a single word: YES or NO."""


_PROVIDER_STAGE1_SYSTEM = """\
You are screening messages from a Telegram chat about high-risk payment processing \
(PSP, acquiring, iGaming, crypto, forex, adult).

Answer YES if the message is from someone SELLING, OFFERING, or PROVIDING \
a payment processing service.
Answer NO for everything else.

YES — the author is a payment provider:
- Direct offer: "предлагаем эквайринг", "we offer processing", "мы PSP"
- First-person coverage: "работаю по EU картам", "покрываю EU/UK", "закрываем гео"
- Provider metrics reported as own: "апрув 72%", "approve rate 75%+", "роллинг 8%"
- Spec-sheet format: "MCC 7995 | EU | Visa/MC | approve 75% | @contact"
- Ukrainian provider language: "пропоную еквайринг", "маємо покриття по ЄС"
- "Можем подключить", "можем закрыть EU high-risk"
- "Процессим гемблинг", "онбординг за 3 дні"

NO — the author is NOT a provider:
- Buyer looking for PSP: "ищу PSP", "нужен эквайринг", "looking for payment gateway"
- Market discussion / questions: "кто работал с X", "какой PSP лучше для EU"
- Complaint / blacklist: "кинули на сетлмент", "чарджбек вырос", "ЧС предупреждение"
- Developer integrating: "интегрирую шлюз в свой сайт", "настраиваю API для проекта"
- IBAN / bank account buyer: "нужен корпоративный счёт", "ищем IBAN для мерчанта"
- SMS, VoIP, ad agency, media buying, traffic — NOT payment processing
- Job posts, news, conference invites, general chat

Reply with a single word: YES or NO."""

_PROVIDER_STAGE1_USER_TEMPLATE = """\
MESSAGE:
{text}

YES or NO?"""


# ── Provider definition — injected into the full system prompt ────────────────
# This is the seller-side equivalent of LeadExtractionConfig.lead_definition.
# Covers all six A2 criteria documented during dataset construction.
# утратав 5% на тестовом датафрейме
PROVIDER_DEFINITION_v1 = (
    "A PROVIDER is a message from someone who IS a payment processing "
    "service and is OFFERING that service — explicitly or implicitly.\n"
    "\n"
    "The message is a PROVIDER signal if the author describes their OWN "
    "payment service, coverage, or terms. This can be expressed several ways:\n"
    "\n"
    "  (a) DIRECT OFFER — explicit offer verb + payment context:\n"
    "      'предлагаем эквайринг', 'предлагаю процессинг', 'мы предлагаем шлюз',\n"
    "      'we offer processing', 'we provide acquiring', 'offering payment solutions',\n"
    "      'пропонуємо еквайринг', 'надаємо послуги процесингу'.\n"
    "\n"
    "  (b) FIRST-PERSON SERVICE DESCRIPTION — no explicit offer verb, but the\n"
    "      author describes their own role as a provider:\n"
    "      * First-person coverage: 'работаю по EU картам', 'покрываю EU/UK/LATAM',\n"
    "        'закрываем EU iGaming', 'покрытие по 40+ странам', 'покриваємо ЄС'.\n"
    "      * Processing verbs: 'процессим гемблинг', 'процессируем high-risk',\n"
    "        'can process MCC 7995', 'processing iGaming and adult'.\n"
    "      * Onboarding verbs: 'подключаем мерчантов', 'онбордим за 3 дня',\n"
    "        'onboarding within 5 days', 'підключаємо мерчантів'.\n"
    "      * Capability statements: 'можем закрыть EU high-risk',\n"
    "        'можем подключить нестандартные MCC', 'able to handle MCC 7995'.\n"
    "\n"
    "  (c) SPEC-SHEET ANNOUNCEMENT — the message is formatted as a provider\n"
    "      offer sheet, listing their own terms with no buyer-intent language:\n"
    "      * Pipe or bullet format: 'MCC 7995 | EU | Visa/MC | апрув 74% | @contact'\n"
    "      * Provider metrics listed as own: 'approve rate 75%+, rolling 10%,\n"
    "        settlement T+7, onboarding 3 days'\n"
    "      * GEO + methods + metrics + CTA: 'Covered on EU, US, LATAM. Cards +\n"
    "        crypto. Chargeback protection. Weekly settlement. DM.'\n"
    "\n"
    "  (d) UKRAINIAN LANGUAGE VARIANTS — same patterns in Ukrainian:\n"
    "      'пропоную еквайринг', 'маємо покриття по ЄС', 'підключаємо мерчантів',\n"
    "      'маємо рішення для гемблінгу', 'закриваємо EU та LATAM'.\n"
    "\n"
    "  (e) CONTEXT-DEPENDENT — the message alone looks neutral but combined with\n"
    "      preceding context from the same author reveals provider role:\n"
    "      context: 'Мы PSP, работаем с EU iGaming' → message: 'Готов обсудить\n"
    "      ваш трафик' — classify as PROVIDER using the context.\n"
    "\n"
    "EXPLICITLY NOT a provider (common false positives):\n"
    "  - BUYERS looking for PSP: 'ищу PSP', 'нужен эквайринг', 'ищем провайдера',\n"
    "    'looking for payment gateway', 'need processing for our casino'.\n"
    "  - MARKET DISCUSSION: 'кто работал с X', 'какой PSP лучше для EU',\n"
    "    'approve rate у Stripe для gambling', 'как работает cascade'.\n"
    "  - DEVELOPER INTEGRATORS: 'интегрирую шлюз в свой сайт', 'настраиваю\n"
    "    checkout для нашего проекта', 'need a PSP with solid API for our platform',\n"
    "    'building an iGaming product and need to integrate a gateway' —\n"
    "    they are BUYING payment infrastructure, not SELLING it.\n"
    "  - COMPLAINTS / BLACKLIST: 'чарджбек вырос до 4%', 'кинули на сетлмент',\n"
    "    'ЧС предупреждение', 'scam alert', 'avoid this provider'.\n"
    "  - IBAN / BANK ACCOUNT BUYERS: 'нужен корпоративный счёт', 'ищем IBAN\n"
    "    для iGaming бизнеса', 'need EMI account for our merchant entity'.\n"
    "  - NON-PAYMENT PROVIDERS: SMS/VoIP providers, ad account renters, media\n"
    "    buyers, traffic sellers, affiliate networks, SEO agencies — they may\n"
    "    use payment terminology but are NOT payment processing providers.\n"
    "  - JOB POSTS, HIRING, CV: people describing their own work history or\n"
    "    looking for payment-related roles.\n"
    "  - NEWS, MARKET ANALYSIS, CONFERENCE POSTS, BOT DIGESTS.\n"
    "\n"
    "CRITICAL DISAMBIGUATION:\n"
    "  'Мы — агрегатор, ищем провайдеров' — NOT a provider (they are a buyer).\n"
    "  'We specialize in iGaming' without payment context — NOT a provider.\n"
    "  'We provide SMS/VoIP/traffic' — NOT a payment provider.\n"
    "  'We offer processing' + payment terms = YES, PROVIDER.\n"
    "  Ambiguous short messages → use [context] block to resolve."
)

# Слишком много ловит всяких дропперов  не провайдеров
PROVIDER_DEFINITION_v2 = (
    "A PROVIDER is a message from someone who IS a payment processing "
    "service and is OFFERING that service — explicitly or implicitly.\n"
    "\n"
    "CRITICAL: A provider does NOT need to say 'I offer' or 'we provide' "
    "explicitly. The absence of an explicit offer verb does NOT mean the "
    "author is a buyer. Judge by ROLE, not by phrasing:\n"
    "  - 'Работаю по EU картам, апрув 71%, роллинг 8%' — own metric + CTA = PROVIDER\n"
    "  - 'Покрыт по UA, PL, CZ, DE. Карты + альт методы. В ЛС' — coverage statement = PROVIDER\n"
    "  - 'Can process MCC 7995, custom routing available, reach out' — capability + CTA = PROVIDER\n"
    "  - 'Working with gambling/forex merchants, stable approval, DM' — serving merchants = PROVIDER\n"
    "  - 'Covering iGaming and adult verticals, onboarding within 5 days' — coverage + onboarding = PROVIDER\n"
    "  Buyers say 'ищу', 'нужен', 'looking for', 'need', 'шукаю'. "
    "If NONE of these buyer signals are present — default to PROVIDER.\n"
    "\n"
    "The message is a PROVIDER signal if the author describes their OWN "
    "payment service, coverage, or terms. This can be expressed several ways:\n"
    "\n"
    "  (a) DIRECT OFFER — explicit offer verb + payment context:\n"
    "      'предлагаем эквайринг', 'предлагаю процессинг', 'мы предлагаем шлюз',\n"
    "      'we offer processing', 'we provide acquiring', 'offering payment solutions',\n"
    "      'пропонуємо еквайринг', 'надаємо послуги процесингу',\n"
    "      'мы предоставляем платежные решения', 'предоставляем стабильный процессинг'.\n"
    "\n"
    "  (b) FIRST-PERSON SERVICE DESCRIPTION — no explicit offer verb, but the\n"
    "      author describes their own role as a provider:\n"
    "      * First-person coverage: 'работаю по EU картам', 'покрываю EU/UK/LATAM',\n"
    "        'закрываем EU iGaming', 'покрытие по 40+ странам', 'покриваємо ЄС'.\n"
    "      * Processing verbs: 'процессим гемблинг', 'процессируем high-risk',\n"
    "        'can process MCC 7995', 'processing iGaming and adult'.\n"
    "      * Onboarding verbs: 'подключаем мерчантов', 'онбордим за 3 дня',\n"
    "        'onboarding within 5 days', 'підключаємо мерчантів'.\n"
    "      * Capability statements: 'можем закрыть EU high-risk',\n"
    "        'можем подключить нестандартные MCC', 'able to handle MCC 7995'.\n"
    "      * Self-identification: 'Мы Incas — платежный провайдер',\n"
    "        'я представляю PSP X', 'мы — платежная инфраструктура для...',\n"
    "        'мы — агрегатор платежных провайдеров'.\n"
    "\n"
    "  (c) SPEC-SHEET ANNOUNCEMENT — the message is formatted as a provider\n"
    "      offer sheet, listing their own terms with no buyer-intent language:\n"
    "      * Pipe or bullet format: 'MCC 7995 | EU | Visa/MC | апрув 74% | @contact'\n"
    "      * Provider metrics listed as own: 'approve rate 75%+, rolling 10%,\n"
    "        settlement T+7, onboarding 3 days'\n"
    "      * GEO + methods + metrics + CTA: 'Covered on EU, US, LATAM. Cards +\n"
    "        crypto. Chargeback protection. Weekly settlement. DM.'\n"
    "      * Flag + methods list: '🇧🇷 PIX - TED - BOLETO / iGaming/Forex / Settlement USDT'\n"
    "\n"
    "  (d) UKRAINIAN LANGUAGE VARIANTS — same patterns in Ukrainian:\n"
    "      'пропоную еквайринг', 'маємо покриття по ЄС', 'підключаємо мерчантів',\n"
    "      'маємо рішення для гемблінгу', 'закриваємо EU та LATAM'.\n"
    "\n"
    "  (e) CONTEXT-DEPENDENT — the message alone looks neutral but combined with\n"
    "      preceding context from the same author reveals provider role:\n"
    "      context: 'Мы PSP, работаем с EU iGaming' → message: 'Готов обсудить\n"
    "      ваш трафик' — classify as PROVIDER using the context.\n"
    "\n"
    "EXPLICITLY NOT a provider (common false positives):\n"
    "  - BUYERS looking for PSP: 'ищу PSP', 'нужен эквайринг', 'ищем провайдера',\n"
    "    'looking for payment gateway', 'need processing for our casino'.\n"
    "  - MARKET DISCUSSION: 'кто работал с X', 'какой PSP лучше для EU',\n"
    "    'approve rate у Stripe для gambling', 'как работает cascade'.\n"
    "  - DEVELOPER INTEGRATORS: 'интегрирую шлюз в свой сайт', 'настраиваю\n"
    "    checkout для нашего проекта', 'need a PSP with solid API for our platform',\n"
    "    'building an iGaming product and need to integrate a gateway' —\n"
    "    they are BUYING payment infrastructure, not SELLING it.\n"
    "  - COMPLAINTS / BLACKLIST: 'чарджбек вырос до 4%', 'кинули на сетлмент',\n"
    "    'ЧС предупреждение', 'scam alert', 'avoid this provider'.\n"
    "  - IBAN / BANK ACCOUNT BUYERS: 'нужен корпоративный счёт', 'ищем IBAN\n"
    "    для iGaming бизнеса', 'need EMI account for our merchant entity'.\n"
    "  - NON-PAYMENT PROVIDERS: SMS/VoIP providers, ad account renters, media\n"
    "    buyers, traffic sellers, affiliate networks, SEO agencies — they may\n"
    "    use payment terminology but are NOT payment processing providers.\n"
    "  - JOB POSTS, HIRING, CV: people describing their own work history or\n"
    "    looking for payment-related roles.\n"
    "  - NEWS, MARKET ANALYSIS, CONFERENCE POSTS, BOT DIGESTS.\n"
    "\n"
    "CRITICAL DISAMBIGUATION:\n"
    "  'Мы — агрегатор, ищем провайдеров' — NOT a provider (they are a buyer).\n"
    "  'We specialize in iGaming' without payment context — NOT a provider.\n"
    "  'We provide SMS/VoIP/traffic' — NOT a payment provider.\n"
    "  'We offer processing' + payment terms = YES, PROVIDER.\n"
    "  Ambiguous short messages → use [context] block to resolve."
)

# До того как клиент скинул правки
PROVIDER_DEFINITION_v3 = (
    "A PROVIDER is a message from someone who IS a payment processing "
    "service and is OFFERING that service — explicitly or implicitly.\n"
    "\n"
    "CRITICAL: A provider does NOT need to say 'I offer' or 'we provide' "
    "explicitly. The absence of an explicit offer verb does NOT mean the "
    "author is a buyer. Judge by ROLE, not by phrasing:\n"
    "  - 'Работаю по EU картам, апрув 71%, роллинг 8%' — own metric + CTA = PROVIDER\n"
    "  - 'Покрыт по UA, PL, CZ, DE. Карты + альт методы. В ЛС' — coverage statement = PROVIDER\n"
    "  - 'Can process MCC 7995, custom routing available, reach out' — capability + CTA = PROVIDER\n"
    "  - 'Working with gambling/forex merchants, stable approval, DM' — serving merchants = PROVIDER\n"
    "  - 'Covering iGaming and adult verticals, onboarding within 5 days' — coverage + onboarding = PROVIDER\n"
    "  Buyers say 'ищу', 'нужен', 'looking for', 'need', 'шукаю'. "
    "If NONE of these buyer signals are present — default to PROVIDER.\n"
    "\n"
    "The message is a PROVIDER signal if the author describes their OWN "
    "payment service, coverage, or terms. This can be expressed several ways:\n"
    "\n"
    "  (a) DIRECT OFFER — explicit offer verb + payment context:\n"
    "      'предлагаем эквайринг', 'предлагаю процессинг', 'мы предлагаем шлюз',\n"
    "      'we offer processing', 'we provide acquiring', 'offering payment solutions',\n"
    "      'пропонуємо еквайринг', 'надаємо послуги процесингу',\n"
    "      'мы предоставляем платежные решения', 'предоставляем стабильный процессинг'.\n"
    "\n"
    "  (b) FIRST-PERSON SERVICE DESCRIPTION — no explicit offer verb, but the\n"
    "      author describes their own role as a provider:\n"
    "      * First-person coverage: 'работаю по EU картам', 'покрываю EU/UK/LATAM',\n"
    "        'закрываем EU iGaming', 'покрытие по 40+ странам', 'покриваємо ЄС'.\n"
    "      * Processing verbs: 'процессим гемблинг', 'процессируем high-risk',\n"
    "        'can process MCC 7995', 'processing iGaming and adult'.\n"
    "      * Onboarding verbs: 'подключаем мерчантов', 'онбордим за 3 дня',\n"
    "        'onboarding within 5 days', 'підключаємо мерчантів'.\n"
    "      * Capability statements: 'можем закрыть EU high-risk',\n"
    "        'можем подключить нестандартные MCC', 'able to handle MCC 7995'.\n"
    "      * Self-identification: 'Мы Incas — платежный провайдер',\n"
    "        'я представляю PSP X', 'мы — платежная инфраструктура для...',\n"
    "        'мы — агрегатор платежных провайдеров'.\n"
    "\n"
    "  (c) SPEC-SHEET ANNOUNCEMENT — the message is formatted as a provider\n"
    "      offer sheet, listing their own terms with no buyer-intent language:\n"
    "      * Pipe or bullet format: 'MCC 7995 | EU | Visa/MC | апрув 74% | @contact'\n"
    "      * Provider metrics listed as own: 'approve rate 75%+, rolling 10%,\n"
    "        settlement T+7, onboarding 3 days'\n"
    "      * GEO + methods + metrics + CTA: 'Covered on EU, US, LATAM. Cards +\n"
    "        crypto. Chargeback protection. Weekly settlement. DM.'\n"
    "      * Flag + methods list: '🇧🇷 PIX - TED - BOLETO / iGaming/Forex / Settlement USDT'\n"
    "\n"
    "  (d) UKRAINIAN LANGUAGE VARIANTS — same patterns in Ukrainian:\n"
    "      'пропоную еквайринг', 'маємо покриття по ЄС', 'підключаємо мерчантів',\n"
    "      'маємо рішення для гемблінгу', 'закриваємо EU та LATAM'.\n"
    "\n"
    "  (e) CONTEXT-DEPENDENT — the message alone looks neutral but combined with\n"
    "      preceding context from the same author reveals provider role:\n"
    "      context: 'Мы PSP, работаем с EU iGaming' → message: 'Готов обсудить\n"
    "      ваш трафик' — classify as PROVIDER using the context.\n"
    "\n"
    "EXPLICITLY NOT a provider (common false positives):\n"
    "  - BUYERS looking for PSP: 'ищу PSP', 'нужен эквайринг', 'ищем провайдера',\n"
    "    'looking for payment gateway', 'need processing for our casino'.\n"
    "  - MARKET DISCUSSION: 'кто работал с X', 'какой PSP лучше для EU',\n"
    "    'approve rate у Stripe для gambling', 'как работает cascade'.\n"
    "  - DEVELOPER INTEGRATORS: 'интегрирую шлюз в свой сайт', 'настраиваю\n"
    "    checkout для нашего проекта', 'need a PSP with solid API for our platform',\n"
    "    'building an iGaming product and need to integrate a gateway' —\n"
    "    they are BUYING payment infrastructure, not SELLING it.\n"
    "  - COMPLAINTS / BLACKLIST: 'чарджбек вырос до 4%', 'кинули на сетлмент',\n"
    "    'ЧС предупреждение', 'scam alert', 'avoid this provider'.\n"
    "  - IBAN / BANK ACCOUNT BUYERS: 'нужен корпоративный счёт', 'ищем IBAN\n"
    "    для iGaming бизнеса', 'need EMI account for our merchant entity'.\n"
    "  - NON-PAYMENT PROVIDERS: SMS/VoIP providers, ad account renters, media\n"
    "    buyers, traffic sellers, affiliate networks, SEO agencies — they may\n"
    "    use payment terminology but are NOT payment processing providers.\n"
    "  - JOB POSTS, HIRING, CV: people describing their own work history or\n"
    "    looking for payment-related roles.\n"
    "  - DROPPER PLATFORMS / GREY P2P SCHEMES: intermediaries who recruit\n"
    "    'traders' (droppers) to process payments through personal bank cards,\n"
    "    SIM farms, or government service abuse. NOT PSP providers.\n"
    "    Key markers (any ONE is enough to disqualify):\n"
    "      * 'Rapira' exchange rate — grey market rate reference\n"
    "      * 'белые треугольники' — money laundering scheme\n"
    "      * 'ферма / фермы' — SIM farm operations\n"
    "      * 'залив / межбанк' — flooding via personal accounts\n"
    "      * 'мобильная коммерция / мобком' — mobile commerce fraud scheme\n"
    "      * 'госки / ГУ' — government service payment abuse\n"
    "      * 'классика ХЧ / классика С2С' — classic drop scheme\n"
    "      * 'контрагенты' in payment context\n"
    "      * 'подключу на площадку' — connecting TO a platform, not BEING one\n"
    "      * 'мануалы для работы / мануалы под команду' — dropper instructions\n"
    "      * 'менторы / менторский состав' — dropper mentoring\n"
    "      * 'страховой депозит' combined with any above markers\n"
    "      * Named platforms: Payscrow, KingsmanPay, RosPlat, OzarkPay,\n"
    "        ZAVOD, SwipePay, FunPay, DecryptPay (in dropper context)\n"
    "  - AD ACCOUNT RENTAL / MEDIA BUYING AGENCIES: companies renting\n"
    "    Facebook BM, Google Ads, TikTok ad accounts. NOT payment processing.\n"
    "    Markers: 'BM2500', 'рекламный аккаунт', 'ads account rental',\n"
    "    'замена при блокировке', 'ad account'.\n"
    "  - VIRTUAL CARD ISSUERS / BIN PROVIDERS: companies selling virtual\n"
    "    cards (EUR/USD/GBP). They issue cards, not process merchant payments.\n"
    "    Markers: 'virtual cards', 'card issuer', 'BIN provider'.\n"
    "  - LEAD GENERATORS / DATA SELLERS: selling FTD data, recovery leads,\n"
    "    chargeback lists, depositor databases. NOT payment processing.\n"
    "    Markers: 'selling FTD', 'ready FTDs', 'recovery leads',\n"
    "    'depositors data', 'registrations', 'live recovery'.\n"
    "  - OTC / USDT EXCHANGERS: buying/selling USDT for fiat through\n"
    "    personal bank cards or drop networks. NOT PSP.\n"
    "    Markers: 'USDT exchange', 'card merchant' (means drop with cards),\n"
    "    'need USDT', 'sell USDT', 'hacker fund'.\n"
    "  - SAAS PLATFORMS FOR BROKERS: all-in-one platforms selling CRM +\n"
    "    trading engine + PSP integration as a feature. They sell software,\n"
    "    not payment processing. Markers: 'CRM + PSP', 'trading platform',\n"
    "    'all-in-one platform for brokers', 'Robo Core'.\n"
    "  - iGAMING PLATFORM DEVELOPERS: building casino/sportsbook software.\n"
    "    'We build iGaming platforms', 'crypto wallet system', 'admin panel'.\n"
    "  - NEWS, MARKET ANALYSIS, CONFERENCE POSTS, BOT DIGESTS.\n"
    "\n"
    "CRITICAL DISAMBIGUATION:\n"
    "  'Мы — агрегатор, ищем провайдеров' — NOT a provider (they are a buyer).\n"
    "  'We specialize in iGaming' without payment context — NOT a provider.\n"
    "  'We provide SMS/VoIP/traffic' — NOT a payment provider.\n"
    "  'We offer processing' + payment terms = YES, PROVIDER.\n"
    "  Ambiguous short messages → use [context] block to resolve."
)

# Промпт с учетом равок клиента
PROVIDER_DEFINITION = (
    "A PROVIDER is a message from someone who IS a payment processing "
    "service/platform and is OFFERING that service — explicitly or implicitly.\n"
    "\n"
    "CRITICAL: A provider does NOT need to say 'I offer' or 'we provide' "
    "explicitly. The absence of an explicit offer verb does NOT mean the "
    "author is a buyer. Judge by ROLE, not by phrasing:\n"
    "  - 'Работаю по EU картам, апрув 71%, роллинг 8%' — own metric + CTA = PROVIDER\n"
    "  - 'Покрыт по UA, PL, CZ, DE. Карты + альт методы. В ЛС' — coverage statement = PROVIDER\n"
    "  - 'Can process MCC 7995, custom routing available, reach out' — capability + CTA = PROVIDER\n"
    "  - 'Working with gambling/forex merchants, stable approval, DM' — serving merchants = PROVIDER\n"
    "  - 'Covering iGaming and adult verticals, onboarding within 5 days' — coverage + onboarding = PROVIDER\n"
    "  Buyers say 'ищу', 'нужен', 'looking for', 'need', 'шукаю'. "
    "If NONE of these buyer signals are present — default to PROVIDER.\n"
    "\n"
    "The message is a PROVIDER signal if the author describes their OWN "
    "payment service, coverage, or terms. This can be expressed several ways:\n"
    "\n"
    "  (a) DIRECT OFFER — explicit offer verb + payment context:\n"
    "      'предлагаем эквайринг', 'предлагаю процессинг', 'мы предлагаем шлюз',\n"
    "      'we offer processing', 'we provide acquiring', 'offering payment solutions',\n"
    "      'пропонуємо еквайринг', 'надаємо послуги процесингу',\n"
    "      'мы предоставляем платежные решения', 'предоставляем стабильный процессинг'.\n"
    "\n"
    "  (b) FIRST-PERSON SERVICE DESCRIPTION — no explicit offer verb, but the\n"
    "      author describes their own role as a provider:\n"
    "      * First-person coverage: 'работаю по EU картам', 'покрываю EU/UK/LATAM',\n"
    "        'закрываем EU iGaming', 'покрытие по 40+ странам', 'покриваємо ЄС'.\n"
    "      * Processing verbs: 'процессим гемблинг', 'процессируем high-risk',\n"
    "        'can process MCC 7995', 'processing iGaming and adult'.\n"
    "      * Onboarding verbs: 'подключаем мерчантов', 'онбордим за 3 дня',\n"
    "        'onboarding within 5 days', 'підключаємо мерчантів'.\n"
    "      * Capability statements: 'можем закрыть EU high-risk',\n"
    "        'можем подключить нестандартные MCC', 'able to handle MCC 7995'.\n"
    "      * Self-identification: 'Мы Incas — платежный провайдер',\n"
    "        'я представляю PSP X', 'мы — платежная инфраструктура для...',\n"
    "        'мы — агрегатор платежных провайдеров'.\n"
    "\n"
    "  (c) SPEC-SHEET ANNOUNCEMENT — the message is formatted as a provider\n"
    "      offer sheet, listing their own terms with no buyer-intent language:\n"
    "      * Pipe or bullet format: 'MCC 7995 | EU | Visa/MC | апрув 74% | @contact'\n"
    "      * Provider metrics listed as own: 'approve rate 75%+, rolling 10%,\n"
    "        settlement T+7, onboarding 3 days'\n"
    "      * GEO + methods + metrics + CTA: 'Covered on EU, US, LATAM. Cards +\n"
    "        crypto. Chargeback protection. Weekly settlement. DM.'\n"
    "      * Flag + methods list: '🇧🇷 PIX - TED - BOLETO / iGaming/Forex / Settlement USDT'\n"
    "\n"
    "  (d) UKRAINIAN LANGUAGE VARIANTS — same patterns in Ukrainian:\n"
    "      'пропоную еквайринг', 'маємо покриття по ЄС', 'підключаємо мерчантів',\n"
    "      'маємо рішення для гемблінгу', 'закриваємо EU та LATAM'.\n"
    "\n"
    "  (e) CONTEXT-DEPENDENT — the message alone looks neutral but combined with\n"
    "      preceding context from the same author reveals provider role:\n"
    "      context: 'Мы PSP, работаем с EU iGaming' → message: 'Готов обсудить\n"
    "      ваш трафик' — classify as PROVIDER using the context.\n"
    "\n"
    "  (f) P2P / OTC PLATFORMS — platforms that process payments through P2P,\n"
    "      bank cards, mobile commerce, or USDT exchange AS THEIR OWN BUSINESS:\n"
    "      * P2P payment platform advertising its own rates and methods:\n"
    "        KingsmanPay, FUNPAY, Payscrow (as platform), Cigarette Payment,\n"
    "        Fourstar Card Merchant — these ARE providers.\n"
    "      * OTC/USDT exchange platforms offering card-to-USDT conversion\n"
    "        as a service — these ARE providers.\n"
    "      * Price lists with methods (Классика, СБП, C2C, мобком, QR НСПК,\n"
    "        фермы, контрагенты, БТ) published BY THE PLATFORM ITSELF.\n"
    "      * Aggregators collecting PSPs into cascade: 'набираем PSP в каскад',\n"
    "        'подключим провайдеров' — they orchestrate payments = PROVIDER.\n"
    "      * Platform seeking merchants: 'ищем мерчантов под мобильную комерцию'\n"
    "        when the author IS the platform = PROVIDER.\n"
    "\n"
    "CRITICAL DISTINCTION — PLATFORM vs INTERMEDIARY:\n"
    "  A PLATFORM that processes payments itself = PROVIDER (even if grey/P2P).\n"
    "  An INTERMEDIARY who connects people TO someone else's platform = NOT PROVIDER.\n"
    "\n"
    "  PROVIDER (platform itself):\n"
    "    'KingsmanPay — вот наши ставки: Классика 11%, МК 15%' — platform's own rates\n"
    "    'FUNPAY — актуальный оффер: Фермы / Контрагенты / Классика' — platform's own offer\n"
    "    'Мы набираем PSP в каскад, 100+ гео' — aggregator = provider\n"
    "    'Cigarette Payment Company is seeking bank card merchants' — platform itself\n"
    "\n"
    "  NOT PROVIDER (intermediary connecting TO a platform):\n"
    "    'Подключу на ZAVOD / Payscrow / RosPlat' — intermediary, not the platform\n"
    "    'CameL подключает к Payscrow' — recruiter for someone else's platform\n"
    "    'Подключение к RosPlat вместе с НЕУЯЗВИМЫМИ' — intermediary\n"
    "    'Подключаю на площадки — КА / Ферма / БТ' — recruiter, not a platform\n"
    "    KEY TEST: does the author say 'подключу НА [platform name]' or\n"
    "    'подключу К [platform name]'? → INTERMEDIARY, not provider.\n"
    "    Does the author publish rates AS their own platform? → PROVIDER.\n"
    "\n"
    "EXPLICITLY NOT a provider:\n"
    "  - INTERMEDIARIES connecting people to other platforms (see above).\n"
    "  - BUYERS looking for PSP: 'ищу PSP', 'нужен эквайринг', 'ищем провайдера',\n"
    "    'looking for payment gateway', 'need processing for our casino'.\n"
    "  - MARKET DISCUSSION: 'кто работал с X', 'какой PSP лучше для EU',\n"
    "    'approve rate у Stripe для gambling', 'как работает cascade'.\n"
    "  - DEVELOPER INTEGRATORS: 'интегрирую шлюз в свой сайт', 'настраиваю\n"
    "    checkout для нашего проекта', 'need a PSP with solid API for our platform',\n"
    "    'building an iGaming product and need to integrate a gateway' —\n"
    "    they are BUYING payment infrastructure, not SELLING it.\n"
    "  - COMPLAINTS / BLACKLIST: 'чарджбек вырос до 4%', 'кинули на сетлмент',\n"
    "    'ЧС предупреждение', 'scam alert', 'avoid this provider'.\n"
    "  - IBAN / BANK ACCOUNT BUYERS: 'нужен корпоративный счёт', 'ищем IBAN\n"
    "    для iGaming бизнеса', 'need EMI account for our merchant entity'.\n"
    "  - NON-PAYMENT PROVIDERS: SMS/VoIP providers, media buyers, traffic\n"
    "    sellers, affiliate networks, SEO agencies — they may use payment\n"
    "    terminology but are NOT payment processing providers.\n"
    "  - AD ACCOUNT RENTAL / MEDIA BUYING AGENCIES: companies renting\n"
    "    Facebook BM, Google Ads, TikTok ad accounts. NOT payment processing.\n"
    "    Markers: 'BM2500', 'рекламный аккаунт', 'ads account rental'.\n"
    "  - VIRTUAL CARD ISSUERS / BIN PROVIDERS: companies selling virtual\n"
    "    cards (EUR/USD/GBP). They issue cards, not process merchant payments.\n"
    "  - LEAD GENERATORS / DATA SELLERS: selling FTD data, recovery leads,\n"
    "    chargeback lists, depositor databases. NOT payment processing.\n"
    "  - SAAS PLATFORMS FOR BROKERS: all-in-one platforms selling CRM +\n"
    "    trading engine + PSP integration as a feature. They sell software,\n"
    "    not payment processing. Markers: 'CRM + PSP', 'trading platform',\n"
    "    'all-in-one platform for brokers'.\n"
    "  - iGAMING PLATFORM DEVELOPERS: building casino/sportsbook software.\n"
    "    'We build iGaming platforms', 'crypto wallet system', 'admin panel'.\n"
    "  - SELLING BANK ACCOUNTS: '#selling #Payment EU SEPA transfers',\n"
    "    'multiple acc types', 'in house UBO & accs' — selling accounts, not processing.\n"
    "  - JOB POSTS, HIRING, CV.\n"
    "  - NEWS, MARKET ANALYSIS, CONFERENCE POSTS, BOT DIGESTS.\n"
    "\n"
    "CRITICAL DISAMBIGUATION:\n"
    "  'Мы — агрегатор, ищем провайдеров' — NOT a provider (they are a buyer).\n"
    "  'Мы набираем PSP в каскад' — YES, PROVIDER (aggregator collecting PSPs).\n"
    "  'We specialize in iGaming' without payment context — NOT a provider.\n"
    "  'We provide SMS/VoIP/traffic' — NOT a payment provider.\n"
    "  'We offer processing' + payment terms = YES, PROVIDER.\n"
    "  'MobiusPay White Label' (selling WL software) — NOT a provider.\n"
    "  'MobiusPay ищет мерчантов' (platform recruiting) — YES, PROVIDER.\n"
    "  Ambiguous short messages → use [context] block to resolve."
)

# ── Full classification system prompt ─────────────────────────────────────────

_PROVIDER_SYSTEM_TEMPLATE = """\
You are a provider-identification agent that reads messages from a multilingual
Telegram chat about high-risk payment processing (iGaming, casinos, sportsbooks,
forex, crypto, adult, nutra). Your job is to decide whether the TARGET MESSAGE
is from a PSP PROVIDER — a company or individual OFFERING payment processing
services — and to extract structured information about them.

PROVIDER DEFINITION
{provider_definition}

OUTPUT
Return strict JSON with this exact shape:
{{
  "is_provider": <bool>,
  "confidence": <float 0..1>,
  "geo": ["<country code or short name>", ...],
  "methods": ["<payment method name>", ...],
  "vertical": ["igaming" | "casino" | "sportsbook" | "forex" | "crypto" | \
"adult" | "nutra" | "dating" | "other" | "unknown", ...],
  "company": "<company or brand name, or null>",
  "evidence_quote": "<short verbatim quote from the target message, <=160 chars>",
  "rationale": "<one or two sentences explaining the call, in English>"
}}

RULES
- Quote evidence VERBATIM from the target message — do not paraphrase.
- If is_provider is false: confidence 0.6-1.0, all lists empty, company null.
- geo: short tokens ('UK', 'DE', 'EU', 'LATAM') or country names; empty if none stated.
- methods: payment methods the provider SUPPORTS (Visa, MC, USDT, PIX, etc.); \
empty if none stated.
- vertical: the high-risk verticals the provider COVERS; use 'unknown' only if \
the message gives NO vertical signal.
- company: extract if the provider names their company/brand; null if not stated.
- BE STRICT. Default is is_provider=false. Set true only when the author is \
clearly the one OFFERING payment processing, not buying or discussing it.
- Use [context] block to interpret short messages and detect buyer vs seller voice.
- The glossary entries are AUTHORITATIVE for jargon: trust their definitions.
- A message describing payment needs IS NOT a provider signal, even if it uses \
provider terminology (approve rate, routing, MCC).
"""

_PROVIDER_USER_TEMPLATE = """\
GLOSSARY (terms relevant to the target message):
{glossary_block}

{few_shot_block}

CONTEXT (preceding messages from the same author / thread):
{context_block}

TARGET MESSAGE:
  msg_id={message_id} @{username} [{timestamp}]
  {text}

Return your JSON decision for the TARGET MESSAGE.
"""


# ── Judge stage prompts ───────────────────────────────────────────────────────
# Independent second-pass LLM that validates the extractor's is_provider=True
# claim. Mirrors the buyer-side _JUDGE_SYSTEM in cli.py but for sellers.
# The judge is intentionally stricter than the extractor — its job is to
# catch hallucinations and false positives before they reach the output.
#
# много дропов и не лидов
_PROVIDER_JUDGE_SYSTEM_v1 = """\
You are an independent reviewer of automatically-extracted PSP provider \
signals from a multilingual Telegram chat about high-risk payment processing \
(iGaming, casinos, sportsbooks, forex, crypto, adult).

The extractor has flagged a message as coming from a PSP PROVIDER — someone \
OFFERING payment processing services. Your job: give an INDEPENDENT verdict.

Is this a REAL_PROVIDER or a MISTAKE?

═══════════════════════════════════════════════════════
REAL_PROVIDER — the author IS offering payment processing:
═══════════════════════════════════════════════════════
  - Explicit offer: 'мы предлагаем эквайринг', 'we offer processing',
    'предоставляем платежные решения', 'пропонуємо еквайринг'.
  - Self-identification: 'мы PSP', 'я представляю провайдера X',
    'наша компания — платежный шлюз', 'мы — агрегатор платежных провайдеров'.
  - First-person coverage: 'работаю по EU картам, апрув 72%',
    'покрываю EU/UK/LATAM', 'закрываем iGaming гео'.
  - Provider metrics as own: 'апрув 75%+, роллинг 8%, сетлмент T+7'.
  - Spec-sheet: 'MCC 7995 | EU | Visa/MC | approve 74% | @contact'.
  - Capability + CTA: 'can process MCC 7995, reach out to discuss volumes'.
  - Ukrainian variants: 'пропоную', 'маємо покриття', 'підключаємо мерчантів'.

═══════════════════════════════════════════════════════
ALWAYS MISTAKE — these are NEVER providers:
═══════════════════════════════════════════════════════

BUYERS looking for PSP (most common false positive):
  'ищу PSP', 'нужен эквайринг', 'looking for payment gateway',
  'need processing for our casino', 'шукаю провайдера',
  'мы — агрегатор, ищем провайдеров' — they are BUYING, not SELLING.

MARKET DISCUSSION / QUESTIONS:
  'кто работал с X?', 'отзывы о X', 'какой PSP лучше для EU',
  'approve rate у Stripe для gambling', 'как работает cascade' —
  networking and research, no offer being made.

DEVELOPER INTEGRATORS:
  'интегрирую шлюз в свой сайт', 'building an iGaming product and need
  to integrate a gateway', 'need a PSP with solid API for our platform' —
  they are BUYING payment infrastructure, not SELLING it.

COMPLAINTS / BLACKLIST:
  'чарджбек вырос до 4%', 'кинули на сетлмент', 'ЧС предупреждение',
  'scam alert', 'avoid this provider', 'держат роллинг сверх договора'.

IBAN / BANK ACCOUNT BUYERS:
  'нужен корпоративный счёт', 'ищем IBAN для iGaming бизнеса',
  'need EMI account for our merchant entity'.

NON-PAYMENT PROVIDERS:
  SMS/VoIP, ad account renters, media buyers, traffic sellers, affiliate
  networks, SEO agencies, data providers, CRM platforms —
  they may use payment terminology but are NOT payment processing providers.
  Key test: do they process card transactions? If not → MISTAKE.

JOB POSTS / CV / HIRING:
  'ищем PSP manager', 'Head of Payments — open to work', '#CV #iGaming
  #Payments' — describing employment, not offering services.

NEWS / MARKET ANALYSIS / BOT DIGESTS:
  'BLIK идёт в ЕС — что это значит...', '📆 Что обсуждалось вчера...',
  'рынок AI в iGaming вырастет...' — informational, no offer.

CONFERENCE / EVENT ANNOUNCEMENTS:
  invites to talks, booth numbers, speaker intros without a payment offer.

═══════════════════════════════════════════════════════
GREY AREA — rule:
═══════════════════════════════════════════════════════
If the message contains BOTH a seller signal AND a buyer signal, \
judge by the DOMINANT voice. A message that says 'мы PSP, ищем мерчантов' \
is still a REAL_PROVIDER — they are offering, not buying.

Output strict JSON only:
{{
  "verdict": "REAL_PROVIDER" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}"""

# До того как клиент скинул правки
_PROVIDER_JUDGE_SYSTEM_v2 = """\
You are an independent reviewer of automatically-extracted PSP provider \
signals from a multilingual Telegram chat about high-risk payment processing \
(iGaming, casinos, sportsbooks, forex, crypto, adult).

The extractor has flagged a message as coming from a PSP PROVIDER — someone \
OFFERING payment processing services. Your job: give an INDEPENDENT verdict.

Is this a REAL_PROVIDER or a MISTAKE?

═══════════════════════════════════════════════════════
REAL_PROVIDER — the author IS offering payment processing:
═══════════════════════════════════════════════════════
  - Explicit offer: 'мы предлагаем эквайринг', 'we offer processing',
    'предоставляем платежные решения', 'пропонуємо еквайринг'.
  - Self-identification: 'мы PSP', 'я представляю провайдера X',
    'наша компания — платежный шлюз', 'мы — агрегатор платежных провайдеров'.
  - First-person coverage: 'работаю по EU картам, апрув 72%',
    'покрываю EU/UK/LATAM', 'закрываем iGaming гео'.
  - Provider metrics as own: 'апрув 75%+, роллинг 8%, сетлмент T+7'.
  - Spec-sheet: 'MCC 7995 | EU | Visa/MC | approve 74% | @contact'.
  - Capability + CTA: 'can process MCC 7995, reach out to discuss volumes'.
  - Ukrainian variants: 'пропоную', 'маємо покриття', 'підключаємо мерчантів'.

═══════════════════════════════════════════════════════
ALWAYS MISTAKE — these are NEVER providers:
═══════════════════════════════════════════════════════

BUYERS looking for PSP (most common false positive):
  'ищу PSP', 'нужен эквайринг', 'looking for payment gateway',
  'need processing for our casino', 'шукаю провайдера',
  'мы — агрегатор, ищем провайдеров' — they are BUYING, not SELLING.

MARKET DISCUSSION / QUESTIONS:
  'кто работал с X?', 'отзывы о X', 'какой PSP лучше для EU',
  'approve rate у Stripe для gambling', 'как работает cascade' —
  networking and research, no offer being made.

DEVELOPER INTEGRATORS:
  'интегрирую шлюз в свой сайт', 'building an iGaming product and need
  to integrate a gateway', 'need a PSP with solid API for our platform' —
  they are BUYING payment infrastructure, not SELLING it.

COMPLAINTS / BLACKLIST:
  'чарджбек вырос до 4%', 'кинули на сетлмент', 'ЧС предупреждение',
  'scam alert', 'avoid this provider', 'держат роллинг сверх договора'.

IBAN / BANK ACCOUNT BUYERS:
  'нужен корпоративный счёт', 'ищем IBAN для iGaming бизнеса',
  'need EMI account for our merchant entity'.

NON-PAYMENT PROVIDERS:
  SMS/VoIP, ad account renters, media buyers, traffic sellers, affiliate
  networks, SEO agencies, data providers, CRM platforms —
  they may use payment terminology but are NOT payment processing providers.
  Key test: do they process card transactions? If not → MISTAKE.

JOB POSTS / CV / HIRING:
  'ищем PSP manager', 'Head of Payments — open to work', '#CV #iGaming
  #Payments' — describing employment, not offering services.

NEWS / MARKET ANALYSIS / BOT DIGESTS:
  'BLIK идёт в ЕС — что это значит...', '📆 Что обсуждалось вчера...',
  'рынок AI в iGaming вырастет...' — informational, no offer.

CONFERENCE / EVENT ANNOUNCEMENTS:
  invites to talks, booth numbers, speaker intros without a payment offer.

DROPPER PLATFORMS / GREY P2P:
  'Подключаю на площадку', 'белые треугольники', 'ферма', 'залив',
  'мобильная коммерция + Rapira', 'госки / ГУ', 'мануалы под команду',
  'менторы 24/7', 'страховой депозит + Rapira', 'классика ХЧ / С2С'.
  Named platforms: Payscrow, KingsmanPay, RosPlat, OzarkPay, ZAVOD,
  SwipePay, FunPay, DecryptPay. These are grey-market intermediaries
  recruiting droppers. They don't hold MIDs. Always MISTAKE.

AD ACCOUNT RENTAL AGENCIES:
  'BM2500', 'рекламный аккаунт', 'ads account rental', 'замена при
  блокировке'. They rent Facebook/Google ad accounts, NOT payment
  processing. Always MISTAKE.

VIRTUAL CARD ISSUERS / BIN PROVIDERS:
  'virtual cards', 'cards in EUR/USD/GBP', 'card issuer'. They issue
  prepaid/virtual cards, not process merchant transactions. MISTAKE.

LEAD GENERATORS / DATA SELLERS:
  'selling FTD', 'ready FTDs', 'recovery leads', 'depositors data',
  'chargeback leads', 'live recovery'. They sell data, not process
  payments. Always MISTAKE.

OTC / USDT EXCHANGERS:
  'USDT exchange', 'card merchant' (= person with bank cards    for drops),
  'need USDT', 'sell USDT', 'hacker fund'. They exchange crypto through
  drop networks, not process merchant transactions. Always MISTAKE.

SAAS PLATFORMS FOR BROKERS:
  'all-in-one platform', 'CRM + PSP + telephony', 'trading platform',
  'Robo Core'. They sell software bundles where PSP is a feature,
  not their core business. MISTAKE.

iGAMING PLATFORM DEVELOPERS:
  'We build iGaming platforms', 'casino + sportsbook + crypto wallet',
  'admin panel + payment integration'. They build software, not
  process payments. Always MISTAKE.
  
═══════════════════════════════════════════════════════
GREY AREA — rule:
═══════════════════════════════════════════════════════
If the message contains BOTH a seller signal AND a buyer signal, \
judge by the DOMINANT voice. A message that says 'мы PSP, ищем мерчантов' \
is still a REAL_PROVIDER — they are offering, not buying.

Output strict JSON only:
{{
  "verdict": "REAL_PROVIDER" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}"""

# Промпт с учетом равок клиента
_PROVIDER_JUDGE_SYSTEM = """\
You are an independent reviewer of automatically-extracted PSP provider \
signals from a multilingual Telegram chat about high-risk payment processing \
(iGaming, casinos, sportsbooks, forex, crypto, adult).

The extractor has flagged a message as coming from a PSP PROVIDER. \
Your job: give an INDEPENDENT verdict — REAL_PROVIDER or MISTAKE.

IMPORTANT: Check MISTAKE patterns FIRST. Default to MISTAKE unless confident.

═══════════════════════════════════════════════════════
STEP 1 — Is this an INTERMEDIARY? (most common mistake)
═══════════════════════════════════════════════════════

CRITICAL DISTINCTION: PLATFORM vs INTERMEDIARY.

A PLATFORM that processes payments itself = REAL_PROVIDER.
An INTERMEDIARY connecting people TO someone else's platform = MISTAKE.

INTERMEDIARY markers (= always MISTAKE):
  'Подключу НА площадку', 'подключу К [platform name]',
  'подключение к RosPlat / ZAVOD / Payscrow',
  'CameL подключает к Payscrow' — recruiter for another platform.
  'Подключаю на площадки — КА / Ферма / БТ' — recruiter.
  'менторы / менторский состав' — dropper mentoring.
  'мануалы для работы / мануалы под команду' — dropper instructions.
  The intermediary does NOT own the platform. They recruit people INTO it.
  KEY TEST: does the message name ANOTHER company's platform and offer
  to connect you there? → INTERMEDIARY → MISTAKE.

PLATFORM markers (= REAL_PROVIDER, even if grey):
  KingsmanPay publishing its OWN rates → REAL_PROVIDER.
  FUNPAY publishing its OWN offer → REAL_PROVIDER.
  Cigarette Payment seeking card merchants → REAL_PROVIDER (it IS the platform).
  'Мы набираем PSP в каскад' → REAL_PROVIDER (aggregator).
  'MobiusPay ищет мерчантов' → REAL_PROVIDER (platform recruiting).

═══════════════════════════════════════════════════════
STEP 2 — Check other MISTAKE patterns
If ANY match → verdict is MISTAKE.
═══════════════════════════════════════════════════════

AD ACCOUNT RENTAL AGENCIES:
  'BM2500', 'рекламный аккаунт', 'ads account rental', 'замена при
  блокировке'. They rent ad accounts, NOT payment processing. MISTAKE.

VIRTUAL CARD ISSUERS / BIN PROVIDERS:
  'virtual cards', 'cards in EUR/USD/GBP', 'card issuer'. MISTAKE.

LEAD GENERATORS / DATA SELLERS:
  'selling FTD', 'ready FTDs', 'recovery leads', 'depositors data',
  'chargeback leads'. They sell data, not process payments. MISTAKE.

SAAS PLATFORMS FOR BROKERS:
  'all-in-one platform' + CRM + telephony/trading. They sell software
  bundles where PSP is one feature, not core business. MISTAKE.
  But: if the company IS a PSP that also has CRM features → REAL_PROVIDER.

iGAMING PLATFORM DEVELOPERS:
  'We build iGaming platforms', 'casino + sportsbook + crypto wallet',
  'admin panel'. Software dev, not payment processing. MISTAKE.

WHITE LABEL SOFTWARE SELLERS:
  Companies selling WL payment platform as SOFTWARE product (not processing
  transactions themselves). 'MobiusPay White Label' = selling software. MISTAKE.
  But: a PSP offering WL as one of their services while also processing → REAL_PROVIDER.

SELLING BANK ACCOUNTS:
  'EU SEPA transfers', 'multiple acc types', 'in house UBO & accs'.
  Selling accounts, not processing. MISTAKE.

BUYERS looking for PSP:
  'ищу PSP', 'нужен эквайринг', 'looking for payment gateway'. MISTAKE.

MEDIA BUYERS / TRAFFIC SELLERS:
  'performance traffic', 'media buyers', 'we provide traffic'. MISTAKE.

MARKET DISCUSSION: 'кто работал с', 'отзывы'. MISTAKE.
COMPLAINTS: 'кинули', 'чарджбек вырос', 'ЧС'. MISTAKE.
JOB POSTS / CV: '#opentowork', 'Head of Payments'. MISTAKE.
NEWS / BOT DIGESTS: '📆 Что обсуждалось', market analysis. MISTAKE.
SHORT / EMPTY CONTEXT: 'Yes we provide that' with no details. MISTAKE.

═══════════════════════════════════════════════════════
STEP 3 — Only if NONE of the above matched:
Is this a REAL_PROVIDER?
═══════════════════════════════════════════════════════

REAL_PROVIDER — the author operates a payment processing platform:
  - They ARE a PSP / acquirer / payment gateway / payment aggregator
  - They process card transactions, P2P payments, crypto, or OTC as core business
  - They ARE a P2P/OTC platform (even grey) publishing their own rates
  - They are an aggregator collecting PSPs into their cascade
  - They are a platform seeking merchants to process through THEIR system
  - They mention: approve rate, MCC, routing, cascade, settlement terms
  - They publish price lists with methods and commissions AS THEIR OWN

KEY QUESTION: Is the author THE PLATFORM / THE COMPANY that processes \
payments? Or are they a middleman connecting people to someone else's \
platform? Platform = REAL_PROVIDER. Middleman = MISTAKE.

Output strict JSON only:
{{
  "verdict": "REAL_PROVIDER" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}"""

_PROVIDER_JUDGE_USER_TEMPLATE = """\
PROVIDER CANDIDATE:
  message_id: {message_id}
  @{username}  [{timestamp}]
  text: {text}

EXTRACTOR's claim:
  is_provider: true
  confidence: {confidence}
  geo: {geo}
  methods: {methods}
  vertical: {vertical}
  company: {company}
  evidence_quote: {evidence_quote}
  rationale: {rationale}

Return JSON:
{{
  "verdict": "REAL_PROVIDER" | "MISTAKE",
  "reason": "<one short sentence, English>"
}}
"""