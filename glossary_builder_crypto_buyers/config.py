"""Pipeline configuration.

Most knobs live here so the system can be tuned without touching the
business logic in other modules. Anything that affects cost or quality
is here and documented inline.

The defaults are tuned for the original PSP / iGaming use case; see the
README's "Retargeting" section for what to swap when you point the
pipeline at a different chat or vertical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


@dataclass
class CandidateConfig:
    """Stage 1 — rule-based candidate extraction."""

    min_frequency: int = 3
    """A token must appear at least this many times across the corpus."""

    min_acronym_len: int = 2
    max_acronym_len: int = 6
    """Latin-uppercase acronym length bounds, e.g. FTD, KYC, MCC."""

    min_latin_word_len: int = 4
    """Lowercase Latin words shorter than this are usually noise (`a`, `of`, `to`)."""

    max_examples_per_term: int = 25
    """Examples kept per candidate for downstream LLM stages."""

    # Tokens we *never* want as candidates regardless of frequency.
    # Keep this list short — the LLM stage is much better than us at filtering.
    hard_skip: set[str] = field(default_factory=lambda: {
        # Ukrainian/Russian noise
        "ТОВ", "ФОП",  # legal entity types (LLC equivalents)
        "ДО", "ВІД", "ТА",  # prepositions that sometimes get capitalised
        # File / media extensions
        "PDF", "JPG", "JPEG", "PNG", "GIF", "MP4", "MP3", "ZIP", "CSV", "XLS", "XLSX",
        # Generic tech
        "URL", "API", "HTML", "HTTP", "HTTPS", "JSON", "XML", "SQL", "VPN", "DNS",
        # Punctuation-y noise
        "OK", "AM", "PM",
    })


@dataclass
class LLMConfig:
    """LLM client behaviour."""

    model: str = os.getenv("GLOSSARY_MODEL", "claude-sonnet-4-5")
    max_tokens: int = 1500
    temperature: float = 0.0
    max_retries: int = 3
    concurrency: int = int(os.getenv("GLOSSARY_CONCURRENCY", "4"))
    api_key: str | None = field(default_factory=lambda: os.getenv("CRYPTO_BUYERS_API_KEY"))


@dataclass
class PhraseConfig:
    """Stage 1b — LLM-assisted multi-language phrase extraction.

    Stage 1 (rules) only catches Latin acronyms and mixed-script tokens. This
    stage handles Cyrillic-only domain phrases (Russian and Ukrainian) and
    multi-word English phrases that the regex can't see.
    """

    enabled: bool = True

    messages_per_batch: int = 20
    """Number of messages per LLM extraction call."""

    max_batches: int = 30
    """Cap on the number of batches sampled. ~30 covers a 3k-message slice well."""

    min_frequency: int = 3
    """A phrase must appear at least this many times in the corpus to keep."""

    max_examples_per_phrase: int = 15
    """Examples retained for downstream LLM stages."""

    min_phrase_length: int = 3
    max_phrase_length: int = 60

    seed_block: str | None = None
    """Optional override for the domain-seed examples shown to the LLM.

    If ``None``, defaults to the crypto OTC/P2P seed block in
    ``glossary_builder.domain_data.crypto_otc_buyers.phrase_seed_block``.
    To retarget a new domain, provide your own multi-line string listing
    illustrative phrases the extractor should hunt for.
    """


@dataclass
class GazetteerConfig:
    """Stage 1c — payment-rail / brand gazetteer matching."""
    enabled: bool = True
    min_frequency: int = 1
    max_examples_per_term: int = 15


@dataclass
class BrandMiningConfig:
    """Stage 1d — PSP / company brand mining via contextual regex + LLM."""
    enabled: bool = True
    confirmation_batch_size: int = 15


@dataclass
class InferenceConfig:
    """Stage 3 — in-corpus LLM inference (Tier 1)."""

    examples_per_call: int = 15
    """How many example messages to show the LLM per term."""

    confidence_accept: float = 0.80
    """Above this, we skip downstream research (Stage 4)."""


@dataclass
class ResearchConfig:
    """Stage 4 — external research (Tier 3). Optional."""

    enabled: bool = False
    """Off by default — needs a web search backend wired in."""

    confidence_floor: float = 0.5
    """Terms with Stage 3 confidence at or below this trigger research."""


@dataclass
class ValidatorConfig:
    """Stage 5 — adversarial critique."""

    enabled: bool = True
    confidence_penalty: float = 0.20
    """If the adversary finds compelling issues, drop confidence by this much."""

    # Deterministic backstop — terms in this set are ALWAYS marked
    # relevant=False regardless of LLM judgment. The audit showed LLMs
    # tend to accept generic English business vocab when those words
    # happen to co-occur with payment topics ("manager", "payments",
    # "business", etc.). These words are not domain terminology; they're
    # standard business English that ANY industry uses.
    generic_vocab_blocklist: set[str] = field(default_factory=lambda: {
        # ---- Ukrainian common business vocab ----
        "компанія", "компанії", "рішення", "послуги", "послуга",
        "команда", "проект", "проекту", "проектів",
        "досвід", "досвід роботи", "робота", "роботи",
        "інформація", "інформацію",
        "питання", "запитання",
        "платежі", "платіж", "оплата",
        "зворотний зв'язок",
        # Ukrainian job titles
        "керівник", "директор", "менеджер", "власник", "засновник",
        "старший", "молодший",
        # ---- Job-title constituents (single word) ----
        "manager", "head", "chief", "senior", "junior", "lead", "director",
        "officer", "executive", "founder", "owner", "partner", "partnership",
        # ---- Multi-word job titles ----
        "account manager", "payment manager", "business manager",
        "business development manager", "business development",
        "head of payments", "head of business development",
        "head of sales", "head of product", "head of account management",
        "key account manager",
        "chief executive officer", "chief business development officer",
        # ---- Domain-adjacent but generic English nouns ----
        "payment", "payments", "business", "development", "operations",
        "sales", "marketing", "product", "account", "accounts", "support",
        "team", "member", "project", "company", "agency", "service",
        "services", "solution", "solutions", "platform", "platforms",
        "system", "systems", "technology", "tech", "finance", "financial",
        "risk",  # alone — distinct from "high-risk"
        # ---- Russian common business vocab ----
        # These are cross-domain Russian generics — NOT specific to this
        # PSP/iGaming chat. They would be equally generic in any Russian-
        # language business context. If you retarget to a non-Russian
        # domain, override this set in PipelineConfig.
        "обратную связь", "обратный связь", "компания", "компании",
        "решение", "решения", "решений", "услуги", "услуга",
        "команда", "проект", "проекта", "проектов",
        "опыт работы", "опыт", "работа", "работы",
        "информация", "информацию", "информацией",
        "вопрос", "вопросы", "вопросов",
        # Bare 'платежи' / 'платежей' (= "payments" generically) — distinct
        # from compound terms like 'платежные системы' which still get
        # admitted via the LLM stages if they prove specific. The current
        # bias was that these become "domain-specific via context", which
        # is the trap.
        "платежи", "платежей", "платеж",
        # ---- ISO currency codes (alone — domain-relevant only with modifier) ----
        "usd", "eur", "gbp", "jpy", "cny", "chf", "cad", "aud", "nzd",
        "rub", "uah", "kzt", "try", "inr", "brl", "mxn", "ars", "cop",
        "pen", "clp", "ngn", "kes", "zar", "egp", "aed", "sar", "pln",
        "czk", "huf", "sek", "nok", "dkk", "ils", "thb", "vnd", "idr",
        "myr", "php", "sgd", "hkd", "twd", "krw", "azn", "byn", "amd",
        # ---- ISO geo codes / region shorthand (alone) ----
        "usa", "uk", "eu", "ru", "ua", "by", "kz", "tr", "il", "ae",
        "in", "cn", "jp", "kr", "sg", "au", "nz", "ca", "mx", "br",
        "ar", "cl", "co", "pe", "ng", "ke", "za", "eg", "sa", "de",
        "fr", "it", "es", "nl", "be", "at", "ch", "se", "no", "dk",
        "fi", "pl", "cz", "ie", "pt", "gb",
        "mena", "cis", "latam", "apac", "emea", "sea",
        # NOTE: deliberately NOT blocking `латам` — it's community-specific
        # Russian shorthand for LatAm (worth keeping in a payments glossary).
        # ---- Generic social / messaging shorthand ----
        "dm", "pm", "лс",
        # ---- Crypto-generic vocab (NOT specific to this OTC/P2P chat —
        # equally generic in any crypto-adjacent conversation) ----
        "token", "coin", "network", "протокол", "blockchain", "блокчейн",
        "крипта", "криптовалюта", "cryptocurrency", "wallet", "кошелёк",
        # ---- P2P private-trade noise (distinct from the business/volume
        # signal we're targeting — a bare mention of "rate" or "exchange"
        # is not evidence of a business buyer) ----
        "курс", "обмен", "обмінник", "сделка", "угода",
    })


@dataclass
class PipelineConfig:
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    phrases: PhraseConfig = field(default_factory=PhraseConfig)
    gazetteer: GazetteerConfig = field(default_factory=GazetteerConfig)
    brand_mining: BrandMiningConfig = field(default_factory=BrandMiningConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)

    output_dir: Path = DEFAULT_OUTPUT_DIR

    domain_brief: str = (
        "A multilingual Telegram group centered on OTC/P2P crypto trading — "
        "buyers and sellers of USDT/BTC/ETH and similar assets post in the same "
        "stream. Participants range from private individuals doing one-off "
        "trades to company/business representatives sourcing crypto on a "
        "recurring basis or in large volume (e.g. gaming companies needing "
        "daily USDT supply, businesses seeking a long-term OTC partner). "
        "Messages are in Russian, Ukrainian, and English, often mixed within "
        "a single message. Common jargon: OTC, P2P, TRC20/ERC20/BEP20, "
        "мейкер/тейкер, эскроу, гарант, KYC, and their Russian/Ukrainian "
        "equivalents. The pipeline's goal for this corpus is to find BUSINESS "
        "buyers with a recurring/large-volume need — not private one-off "
        "buyers, not sellers, not two-sided exchangers."
    )
    """High-level context handed to the LLM with every call.

    Change this to retarget the pipeline at a different domain / language.
    """