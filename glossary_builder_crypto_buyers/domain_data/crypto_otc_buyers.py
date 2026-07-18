"""Crypto OTC / P2P — B2B regular-volume buyers domain data.

Two pieces:
  * ``gazetteer`` — a list of canonical crypto-asset / network entries that
    the gazetteer scanner (``glossary_builder.gazetteer``) matches against
    the corpus regardless of frequency.
  * ``phrase_seed_block`` — the rotating-categories prompt fragment that
    seeds Stage 1b's LLM-assisted phrase extractor with examples of the
    kinds of jargon to hunt for.

Both are content, not logic. Retargeted from ``payments_igaming.py``.

Target vertical: B2B / regular-volume crypto buyers in a mixed P2P/OTC
Telegram chat (buyers AND sellers post in the same stream). We are looking
for COMPANY/BUSINESS buyers seeking a crypto supplier on a recurring basis
or in large volume — NOT private one-off buyers, NOT sellers, NOT
two-sided exchangers, NOT trading/DeFi/mining discussion.

Decision log (see project checklist):
  - No separate gazetteer additions for OTC-desk brands / business payment
    rails — the existing crypto_rail/crypto_network base (carried over
    unchanged from payments_igaming.py) is considered sufficient. Business/
    volume vocabulary lives in phrase_seed_block + intent_filter.py's
    INTENT_SIGNALS instead, not here.
"""

from __future__ import annotations


gazetteer: list[dict] = [
    # ---- Crypto assets ----
    {"canonical": "USDT", "forms": ["USDT", "Tether"], "category": "crypto_rail"},
    {"canonical": "USDC", "forms": ["USDC", "USD Coin"], "category": "crypto_rail"},
    {"canonical": "BTC", "forms": ["BTC", "Bitcoin"], "category": "crypto_rail"},
    {"canonical": "ETH", "forms": ["ETH", "Ethereum"], "category": "crypto_rail"},

    # ---- Networks ----
    {"canonical": "TRC-20", "forms": ["TRC-20", "TRC20", "TRC 20"], "category": "crypto_network"},
    {"canonical": "ERC-20", "forms": ["ERC-20", "ERC20", "ERC 20"], "category": "crypto_network"},
    {"canonical": "BEP-20", "forms": ["BEP-20", "BEP20", "BEP 20"], "category": "crypto_network"},
]


phrase_seed_block: str = """\
The chat may contain messages in Russian, Ukrainian, and English — often
mixed within a single message. Extract industry-specific terms in ANY of
these languages.

This is a mixed P2P/OTC crypto chat: buyers AND sellers post in the same
stream. We care about BUSINESS buyers seeking a crypto supplier on a
recurring basis or in large volume — not private one-off buyers, not
sellers, not two-sided exchangers, not trading/DeFi/mining chatter.

Categories of terms we explicitly want you to catch (these are common in
this community — if any appears in the batch, extract it):

  Business / recurring-volume framing (the core signal for this vertical):
    ежедневный объём, ежедневная закупка, постоянная основа,
    долгосрочное партнёрство, долгосрочный поставщик, регулярные объёмы,
    оптовая закупка, закупка крипты, постоянный поставщик,
    long-term partner, long-term supplier, daily volume, recurring supply,
    upfront supply, operational need, game funds, gaming company,
    our company needs, corporate account, business volumes

  Ukrainian business/volume framing:
    щоденний обсяг, постійна основа, довгостроковий партнер,
    оптова закупівля, регулярні обсяги, постачальник криптовалюти,
    компанія шукає постачальника

  OTC / P2P crypto trade vocabulary:
    ОТС / OTC сделка, ОТС деск, P2P сделка, мейкер / тейкер,
    контрагент, эскроу, гарант сделки, объём сделки, лимит сделки,
    комиссия за обмен, спред, курс покупки, курс продажи,
    сеттлмент крипта, ончейн / оффчейн расчёт, холодный кошелёк,
    горячий кошелёк, верификация контрагента, KYC контрагента

  Buy-side signals (what we want) vs sell-side signals (what we do NOT
  want — extract both so downstream stages can tell them apart):
    куплю крипту, куплю USDT, ищу продавца, нужен поставщик USDT,
    покупаем крипту оптом, закупаем ежедневно
    [buy-side]
    продам USDT, продаю крипту, меняю USDT на рубли, ищу покупателя,
    отдам крипту
    [sell-side — NOT relevant to this vertical]

  Payment rails used to settle the crypto purchase (business context):
    SWIFT, SEPA, Wise, Revolut, банковский перевод, безналичный расчёт,
    корпоративный счёт, инвойс, оплата по счёту, Binance Pay

  Chat-institutional / P2P market terms (specific to this kind of group):
    ЧС (blacklist of unreliable counterparties), гарант (escrow guarantor),
    trusted seller / trusted buyer, объявление (posted buy/sell listing),
    +1 в поиске (thread-joining buyer signal)
"""
