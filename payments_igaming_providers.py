"""Payments / iGaming / high-risk processing domain data — PROVIDER SIDE.

Two pieces:
  * ``gazetteer`` — canonical payment-method / rail / brand entries.
    Identical to the buyer-side module: the terms themselves don't change,
    only the context in which providers mention them.
  * ``phrase_seed_block`` — prompt fragment that seeds Stage 1b's LLM-assisted
    phrase extractor with examples of SELLER-SIDE jargon to hunt for.
    This is the key difference from payments_igaming.py: buyer-intent phrases
    are replaced with provider-intent phrases.

Replace this module wholesale when retargeting; do NOT edit pipeline internals.
"""

from __future__ import annotations


gazetteer: list[dict] = [
    {"canonical": "Monobank", "forms": ["Monobank", "Монобанк", "Mono"], "category": "apm_ua"},
    {"canonical": "PrivatBank", "forms": ["PrivatBank", "Приватбанк", "Privat24"], "category": "apm_ua"},
    {"canonical": "Oschadbank", "forms": ["Oschadbank", "Ощадбанк"], "category": "apm_ua"},
    {"canonical": "Portmone", "forms": ["Portmone"], "category": "apm_ua"},
    {"canonical": "LiqPay", "forms": ["LiqPay", "Ліквей"], "category": "apm_ua"},
    {"canonical": "WayForPay", "forms": ["WayForPay", "Way for Pay"], "category": "apm_ua"},
    {"canonical": "EasyPay", "forms": ["EasyPay", "Ізіпей"], "category": "apm_ua"},

    # ---- Cards & networks ----
    {"canonical": "Visa", "forms": ["Visa"], "category": "card_scheme"},
    {"canonical": "Mastercard", "forms": ["Mastercard", "MasterCard", "MC", "Master Card"], "category": "card_scheme"},
    {"canonical": "American Express", "forms": ["AMEX", "American Express", "Amex"], "category": "card_scheme"},
    {"canonical": "Discover", "forms": ["Discover"], "category": "card_scheme"},
    {"canonical": "JCB", "forms": ["JCB"], "category": "card_scheme"},
    {"canonical": "UnionPay", "forms": ["UnionPay", "Union Pay", "CUP"], "category": "card_scheme"},
    {"canonical": "Mir", "forms": ["Mir Pay", "МИР", "Мир"], "category": "card_scheme"},
    {"canonical": "Maestro", "forms": ["Maestro"], "category": "card_scheme"},

    # ---- Wallets / e-money ----
    {"canonical": "Skrill", "forms": ["Skrill"], "category": "ewallet"},
    {"canonical": "Neteller", "forms": ["Neteller"], "category": "ewallet"},
    {"canonical": "ecoPayz", "forms": ["ecoPayz", "EcoPayz", "Eco Payz"], "category": "ewallet"},
    {"canonical": "MuchBetter", "forms": ["MuchBetter", "Much Better"], "category": "ewallet"},
    {"canonical": "MiFinity", "forms": ["MiFinity", "Mifinity"], "category": "ewallet"},
    {"canonical": "AstroPay", "forms": ["AstroPay", "Astro Pay"], "category": "ewallet"},
    {"canonical": "Jeton", "forms": ["Jeton"], "category": "ewallet"},
    {"canonical": "EcoVoucher", "forms": ["EcoVoucher", "Eco Voucher"], "category": "ewallet"},
    {"canonical": "Paysafecard", "forms": ["Paysafecard", "Paysafe Card", "PSC"], "category": "ewallet"},
    {"canonical": "PayPal", "forms": ["PayPal", "Paypal"], "category": "ewallet"},
    {"canonical": "Webmoney", "forms": ["Webmoney", "WebMoney"], "category": "ewallet"},
    {"canonical": "Piastrix", "forms": ["Piastrix"], "category": "ewallet"},
    {"canonical": "FK Wallet", "forms": ["FK Wallet", "FKWallet"], "category": "ewallet"},
    {"canonical": "ЮMoney", "forms": ["ЮMoney", "Yoomoney", "Yandex Money", "ЯндексДеньги"], "category": "ewallet"},
    {"canonical": "Qiwi", "forms": ["Qiwi", "QIWI", "Киви"], "category": "ewallet"},
    {"canonical": "Wero", "forms": ["wero", "Wero"], "category": "ewallet"},

    # ---- Bank redirect / open banking APMs ----
    {"canonical": "iDeal", "forms": ["iDeal", "IDeal"], "category": "bank_redirect"},
    {"canonical": "Trustly", "forms": ["Trustly"], "category": "bank_redirect"},
    {"canonical": "Sofort", "forms": ["Sofort", "SOFORT"], "category": "bank_redirect"},
    {"canonical": "GiroPay", "forms": ["GiroPay", "Giro Pay"], "category": "bank_redirect"},
    {"canonical": "EPS", "forms": ["EPS"], "category": "bank_redirect"},
    {"canonical": "MultiBanco", "forms": ["MultiBanco", "Multi Banco"], "category": "bank_redirect"},
    {"canonical": "Bancontact", "forms": ["Bancontact"], "category": "bank_redirect"},
    {"canonical": "Klarna", "forms": ["Klarna"], "category": "bnpl"},

    # ---- Local rails / APMs by region ----
    {"canonical": "Blik", "forms": ["Blik", "BLIK"], "category": "apm_pl"},
    {"canonical": "Swish", "forms": ["Swish"], "category": "apm_se"},
    {"canonical": "Vipps", "forms": ["Vipps"], "category": "apm_no"},
    {"canonical": "PIX", "forms": ["PIX", "Pix"], "category": "apm_br"},
    {"canonical": "Boleto", "forms": ["Boleto"], "category": "apm_br"},
    {"canonical": "OXXO", "forms": ["OXXO"], "category": "apm_mx"},
    {"canonical": "SPEI", "forms": ["SPEI"], "category": "apm_mx"},
    {"canonical": "MercadoPago", "forms": ["MercadoPago", "Mercado Pago"], "category": "apm_latam"},
    {"canonical": "INTERAC", "forms": ["INTERAC", "Interac"], "category": "apm_ca"},
    {"canonical": "Aircash", "forms": ["Aircash", "Air Cash"], "category": "apm_hr"},
    {"canonical": "KEKS Pay", "forms": ["KEKS Pay", "KEKS"], "category": "apm_hr"},
    {"canonical": "CorvusPay", "forms": ["CorvusPay", "Corvus Pay"], "category": "apm_hr"},
    {"canonical": "Havale", "forms": ["Havale", "хавале", "Хавале"], "category": "apm_tr"},
    {"canonical": "Papara", "forms": ["Papara"], "category": "apm_tr"},
    {"canonical": "СБП", "forms": ["СБП", "SBP", "FPS Russia"], "category": "apm_ru"},
    {"canonical": "Сбербанк", "forms": ["Сбербанк", "Sberbank", "Сбер"], "category": "apm_ru"},

    # ---- Cards / Mobile / Asian wallets ----
    {"canonical": "Apple Pay", "forms": ["Apple Pay", "ApplePay"], "category": "mobile_wallet"},
    {"canonical": "Google Pay", "forms": ["Google Pay", "GooglePay", "G Pay", "GPay"], "category": "mobile_wallet"},
    {"canonical": "Samsung Pay", "forms": ["Samsung Pay", "SamsungPay"], "category": "mobile_wallet"},
    {"canonical": "WeChat Pay", "forms": ["WeChat Pay", "WeChatPay", "WeChat"], "category": "apm_cn"},
    {"canonical": "Alipay", "forms": ["Alipay", "AliPay"], "category": "apm_cn"},
    {"canonical": "GCash", "forms": ["GCash"], "category": "apm_ph"},
    {"canonical": "GrabPay", "forms": ["GrabPay", "Grab Pay"], "category": "apm_sg"},
    {"canonical": "UPI", "forms": ["UPI"], "category": "apm_in"},
    {"canonical": "Paytm", "forms": ["Paytm"], "category": "apm_in"},
    {"canonical": "PhonePe", "forms": ["PhonePe", "Phone Pe"], "category": "apm_in"},
    {"canonical": "Razorpay", "forms": ["Razorpay"], "category": "psp_in"},
    {"canonical": "Bkash", "forms": ["Bkash", "bKash"], "category": "apm_bd"},

    # ---- Africa ----
    {"canonical": "M-Pesa", "forms": ["M-Pesa", "MPesa", "Mpesa"], "category": "apm_africa"},
    {"canonical": "MTN MoMo", "forms": ["MTN MoMo", "MoMo"], "category": "apm_africa"},
    {"canonical": "Vodacom", "forms": ["Vodacom"], "category": "apm_africa"},
    {"canonical": "Flutterwave", "forms": ["Flutterwave"], "category": "psp_africa"},
    {"canonical": "Paystack", "forms": ["Paystack"], "category": "psp_africa"},

    # ---- Crypto rails ----
    {"canonical": "USDT", "forms": ["USDT", "Tether"], "category": "crypto_rail"},
    {"canonical": "USDC", "forms": ["USDC", "USD Coin"], "category": "crypto_rail"},
    {"canonical": "BTC", "forms": ["BTC", "Bitcoin"], "category": "crypto_rail"},
    {"canonical": "ETH", "forms": ["ETH", "Ethereum"], "category": "crypto_rail"},
    {"canonical": "TRC-20", "forms": ["TRC-20", "TRC20", "TRC 20"], "category": "crypto_network"},
    {"canonical": "ERC-20", "forms": ["ERC-20", "ERC20", "ERC 20"], "category": "crypto_network"},
    {"canonical": "BEP-20", "forms": ["BEP-20", "BEP20", "BEP 20"], "category": "crypto_network"},

    # ---- Adult-industry PSPs ----
    {"canonical": "CCBill", "forms": ["CCBill", "CCbill", "Ccbill"], "category": "psp_adult"},
    {"canonical": "Segpay", "forms": ["Segpay", "SegPay", "Seg Pay"], "category": "psp_adult"},

    # ---- Industry terms & networks ----
    {"canonical": "ЦУПИС", "forms": ["ЦУПИС", "ZUPIS"], "category": "infra_ru"},
    {"canonical": "FasterPayments", "forms": ["Faster Payments", "FasterPayments"], "category": "apm_uk"},
    {"canonical": "OnRamp", "forms": ["on-ramp", "onramp", "on ramp"], "category": "crypto_flow"},
    {"canonical": "OffRamp", "forms": ["off-ramp", "offramp", "off ramp"], "category": "crypto_flow"},

    # ---- Processing-stack vocabulary (English) ----
    {"canonical": "3DS", "forms": ["3DS", "3-D Secure", "3D Secure"], "category": "stack"},
    {"canonical": "MoR", "forms": ["MoR", "Merchant of Record"], "category": "stack"},
    {"canonical": "rolling reserve", "forms": ["rolling reserve", "роллинг резерв"], "category": "stack"},
    {"canonical": "chargeback", "forms": ["chargeback", "charge-back", "чарджбэк", "чарджбек"], "category": "stack"},
    {"canonical": "approve rate", "forms": ["approve rate", "approval rate", "аппрув рейт"], "category": "stack"},
    {"canonical": "push-to-card", "forms": ["push-to-card", "push to card", "P2C", "OCT"], "category": "stack"},
    {"canonical": "multi-MID", "forms": ["multi-MID", "multi MID", "multimid"], "category": "stack"},
    {"canonical": "cascade", "forms": ["cascade", "каскад"], "category": "stack"},

    # ---- Verticals (English) ----
    {"canonical": "iGaming", "forms": ["iGaming", "igaming"], "category": "vertical"},
    {"canonical": "sportsbook", "forms": ["sportsbook", "sports book", "bookmaker"], "category": "vertical"},
    {"canonical": "forex", "forms": ["forex", "FX"], "category": "vertical"},
    {"canonical": "sweepstakes", "forms": ["sweepstakes", "sweeps"], "category": "vertical"},
    {"canonical": "nutra", "forms": ["nutra"], "category": "vertical"},
    {"canonical": "adult", "forms": ["adult", "эдалт"], "category": "vertical"},
    {"canonical": "dating", "forms": ["dating"], "category": "vertical"},
    {"canonical": "open banking", "forms": ["open banking", "Open Banking"], "category": "stack"},
    {"canonical": "acquiring", "forms": ["acquiring", "acquirer", "acquiror"], "category": "stack"},
    {"canonical": "payment orchestration", "forms": ["payment orchestration", "orchestrator"], "category": "stack"},
    {"canonical": "merchant of record", "forms": ["merchant of record", "MoR"], "category": "stack"},
]


phrase_seed_block: str = """\
The chat may contain messages in Russian, Ukrainian, and English — often
mixed within a single message. Extract industry-specific terms in ANY of
these languages.

We are looking for terminology used by PSP PROVIDERS — companies and
individuals who OFFER payment processing services. Focus on seller-side
language: how a provider describes their own product, coverage, and terms.

Categories of terms we explicitly want you to catch:

  Ukrainian seller-side phrases (provider context):
    пропоную еквайринг (I offer acquiring),
    маємо покриття (we have coverage),
    підключаємо мерчантів (we onboard merchants),
    покриваємо ЄС / EU (we cover EU),
    маємо рішення для гемблінгу (we have a solution for gambling),
    апрув X% (approve rate as a provider metric),
    роутинг / маршрутизація (routing — provider's own setup),
    каскадування (cascading — provider's own setup),
    онбординг за N днів (onboarding in N days),
    сетлмент / розрахунок T+N (settlement terms provider sets),
    роллінг / резервування (rolling reserve — provider's terms),
    пишіть в лс / стукайте в лс (CTA from provider)

  English seller-intent phrases (PSP provider context):
    we offer processing, we provide acquiring, covering EU geo,
    approve rate 75%+, onboarding in 3 days, rolling reserve 10%,
    T+7 settlement, we process high-risk, can handle iGaming,
    our gateway, our processing, our solution, our platform,
    we cover, DM for details, reach out to discuss volumes,
    high-risk acquiring available, multi-MID routing,
    custom cascade setup, fast merchant onboarding,
    processing iGaming and adult, covering LATAM and EU,
    able to handle MCC 7995, high approval rates

  Russian seller-side phrases (provider context):
    работаю по EU картам (I work EU cards — provider's own),
    покрываю / покрытие по (I cover / coverage by — provider's GEO claim),
    закрываем гео / закрываем EU (we cover — provider's offering),
    процессим / процессируем (we process — provider's verb),
    подключаем / подключим мерчантов (we onboard merchants),
    онбордим / онбординг за N дней (we onboard in N days),
    апрув X% (approve rate — provider's own metric),
    роутинг настраиваем / наш роутинг (our routing setup),
    каскад / каскадирование (our cascade),
    роллинг X% / наш роллинг (rolling reserve — provider's terms),
    сетлмент T+N / сетлмент раз в неделю (settlement terms),
    пишите в лс / стучите в лс (CTA — contact for services),
    специализируемся на (we specialise in — provider's vertical),
    можем закрыть / можем подключить (we can cover / connect),
    наш шлюз / наш гейтвей / наш процессинг (our gateway / processing)

  Processing-stack vocabulary (same terms, provider context):
    multi-MID, routing, cascade, approve rate, 3DS, push-to-card,
    on-ramp, off-ramp, MoR (merchant of record), MID, MDR,
    KYB, AML, PEP, sanctions screening, IBAN, SEPA,
    P2C (person-to-card), P2P (peer-to-peer), MCC, OCT,
    rolling reserve, set-up fee, white label, WL,
    high-risk MID, FTD-ratio, settlement, onboarding

  iGaming / high-risk verticals (provider covers these):
    iGaming, gambling, betting, sportsbook, forex, adult, nutra,
    dating, sweepstakes, crypto, high-risk, quasi-ecom

  Provider-specific announcement formats:
    MCC 7995 | EU | Visa/MC | 75% approve — pipe-format spec sheets,
    GEO + methods + metrics + contact — provider announcement pattern,
    covered on EU US LATAM — geographic coverage statement
"""
