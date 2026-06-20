"""Payments / iGaming / high-risk processing domain data.

Two pieces:
  * ``gazetteer`` — a list of canonical payment-method / rail / brand
    entries that the gazetteer scanner (``glossary_builder.gazetteer``)
    matches against the corpus regardless of frequency.
  * ``phrase_seed_block`` — the rotating-categories prompt fragment that
    seeds Stage 1b's LLM-assisted Russian-phrase extractor with examples
    of the kinds of jargon to hunt for.

Both are content, not logic. They are the obvious things to swap when
retargeting this pipeline at a different vertical (e.g. crypto OTC,
adtech, DeFi). Replace this module wholesale; do NOT edit the pipeline
internals.
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

Categories of terms we explicitly want you to catch (these are common in this
community — if any appears in the batch, extract it):

    Ukrainian slang & abbreviations:
    першак / перший депозит (first-time deposit),
    каскад / каскадування (cascading),
    чорний список (blacklist),
    чарджбек (chargeback), повернення (refund),
    еквайринг / еквайєр (acquiring),
    онбординг (merchant onboarding),
    трафік / лити трафік (affiliate traffic),
    шахрайство / скам (fraudulent PSP),
    виплата / виплати (payout),
    розрахунок / сетлмент (settlement),
    ризик / високий ризик (high-risk),
    платіжне рішення (payment solution),
    агрегатор / оркестратор (payment orchestrator)

  English buyer-intent phrases (iGaming payments context):
    seeking PSP, looking for acquirer, need payment solution,
    anyone offer, does anyone have, open to offers,
    first-time deposit traffic, high-risk processing,
    payment gateway needed, merchant account needed,
    rolling reserve negotiable, low MDR, high approve rate,
    white label cashier, plug-and-play integration
    
  Russian slang & abbreviations:
    оффер / офферы (commercial proposal from a PSP),
    первичка (card processing for first-time deposits),
    каскад / каскадирование (cascading failed transactions through MIDs),
    ЧС (chat-internal blacklist of bad PSPs),
    чб / чарджбэк (chargeback), рефанд (refund),
    ХЧ (high-risk, abbreviated Russian),
    гэмбла / гембет / гемблинг (gambling),
    эквайр / эквайер (acquirer, distinct from эквайринг),
    онбординг / бордиться (merchant onboarding),
    РР (rolling reserve), трансгран (cross-border),
    квази-эком (quasi-ecommerce — masking gambling as retail),
    рекурент / рекуррентные (recurring billing),
    латам (LatAm), еком / ecom (e-commerce vertical),
    скам (fraudulent PSP), морозиться (ghosting after non-payment),
    отлив / лить трафик / лить (driving affiliate traffic),
    дроп (drop accounts / mules),
    антифрод (anti-fraud system), карусель (transaction routing),
    конверт / конверсия (conversion / approval rate),
    оркестратор / агрегатор (payment orchestrator),
    номинал / номиналы (nominee directors), мерч (merchandise OR merchant),
    мисскод (MCC miscoded transactions),
    выплата / выплаты (payout), сэтл / сеттл / сеттлмент (settlement)

  English/mixed payments stack:
    multi-MID, routing, cascade, approve rate, 3DS, push-to-card,
    on-ramp, off-ramp, MoR (merchant of record), MID, MDR,
    KYB, AML, PEP, sanctions screening, IBAN, SEPA,
    P2C (person-to-card), P2P (peer-to-peer), MCC, OCT,
    rolling reserve, set-up fee, white label, WL,
    high-risk MID, FTD-ratio

  iGaming/affiliate vocabulary:
    bookmaker, sportsbook, casino, igaming, gambling, betting,
    affiliate, mediabuy, GEO, vertical, KAM, sweepstakes,
    forex, crypto, adult, nutra, dating, edalt

  Chat-institutional terms (specific to this group):
    ЧС (blacklist), Red / Orange (blacklist tiers), Shittest (admin command),
    trusted (vetted PSP for FTDs), вторичка (secondary / repeat-deposit traffic)
"""
