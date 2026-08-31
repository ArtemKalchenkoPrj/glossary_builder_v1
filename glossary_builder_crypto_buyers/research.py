"""Stage 4 — external research (Tier 3).

Only invoked for terms where Stages 2 and 3 returned low confidence. The agent
constructs a domain-aware query (e.g. ``"FTD" igaming acronym meaning``),
fetches a few results, and asks the LLM to reconcile what it learned with the
example messages it has already seen — the round-trip validation step.

This module ships with the *interface* and a clean abstraction over a
``WebSearchTool`` so you can plug in whichever backend you prefer:

    * Tavily Search API (default implementation, recommended)
    * Anthropic's web search tool
    * Brave Search / Serper APIs
    * Bring-your-own scraper

The default ``NoOpWebSearch`` simply returns no results and short-circuits
this stage, so the pipeline runs end-to-end without web access. Replace it
with ``TavilyWebSearch`` to enable real research.

Setup:
    pip install httpx
    # Add to .env:
    TAVILY_API_KEY=tvly-...
    RESEARCH_ENABLED=true
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .config import ResearchConfig
from .inference import DraftEntry, Sense
from .llm import LLMClient


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool(Protocol):
    """Plug-in interface for any web-search backend."""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        ...


class NoOpWebSearch:
    """Default — returns nothing. Stage 4 effectively skipped until replaced."""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return []


class TavilyWebSearch:
    """Tavily Search API backend.

    Free tier: 1000 requests/month — enough for glossary builds.

    Usage:
        search = TavilyWebSearch()                        # reads TAVILY_API_KEY from env
        search = TavilyWebSearch(api_key="tvly-...")      # explicit key
        search = TavilyWebSearch(search_depth="advanced") # slower but better results
    """

    def __init__(
        self,
        api_key: str | None = None,
        search_depth: str = "basic",  # "basic" or "advanced"
        timeout: int = 10,
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY in .env "
                "or pass api_key= explicitly."
            )
        self.search_depth = search_depth
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TavilyWebSearch. "
                "Run: pip install httpx"
            ) from exc

        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self.search_depth,
                    # Ask Tavily to return clean text snippets, not raw HTML.
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            # Network / auth errors should not kill the whole pipeline —
            # research is optional. Log and return empty.
            import logging
            logging.getLogger(__name__).warning("Tavily search failed: %s", exc)
            return []

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in resp.json().get("results", [])
        ]


def build_search_backend(cfg: ResearchConfig) -> WebSearchTool:
    """Convenience factory — reads config/env and returns the right backend.

    Called once at pipeline startup. Returns NoOpWebSearch when research is
    disabled so callers don't need to check cfg.enabled themselves.
    """
    if not cfg.enabled:
        return NoOpWebSearch()
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        import logging
        logging.getLogger(__name__).warning(
            "ResearchConfig.enabled=True but TAVILY_API_KEY is not set. "
            "Stage 4 will be skipped. Add TAVILY_API_KEY to .env to enable."
        )
        return NoOpWebSearch()
    return TavilyWebSearch(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_SYSTEM = """\
You craft web-search queries to disambiguate niche jargon in the payments /
iGaming industry. Output a single short query, no quotes around the whole
thing, no commentary. Keep it under 12 words.
"""

_QUERY_USER_TEMPLATE = """\
Term to research: {term}

Existing low-confidence draft definition: {draft}

Example messages where the term appears (short snippets):
{snippets}

Suggest ONE web search query likely to return an authoritative explanation
of this term in its industry context. Include the term plus 2-4 disambiguating
keywords drawn from the snippets above. Do not put quotes around the whole
query.
"""

_RECONCILE_SYSTEM = """\
You reconcile new external information with prior chat-derived evidence.
Output strict JSON only.
"""

_RECONCILE_USER_TEMPLATE = """\
TERM: {term}

PRIOR LOW-CONFIDENCE DRAFT:
definition: {prior_definition}
confidence: {prior_confidence}

WEB SEARCH RESULTS:
{web_snippets}

ORIGINAL EXAMPLE MESSAGES FROM CORPUS:
{examples}

Decide:
  1. Does the web information fit the original example messages?
  2. If yes, refine the definition (English, <=160 chars).
  3. If the web info contradicts or doesn't match the examples, REJECT and
     keep the prior draft.

Return JSON:
{{
  "accept_web_info": <bool>,
  "definition": "<refined or prior definition>",
  "confidence": <float 0..1>,
  "reasoning": "<one or two sentences>",
  "sources": ["<url>", ...]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def needs_research(draft: DraftEntry, cfg: ResearchConfig) -> bool:
    """Trigger condition: low-confidence AND deemed relevant."""
    if not cfg.enabled:
        return False
    if not draft.relevant:
        return False
    return draft.confidence <= cfg.confidence_floor


def _format_snippets(examples: list[dict], limit: int = 8) -> str:
    lines = []
    for ex in examples[:limit]:
        t = (ex.get("text") or "").replace("\n", " ").strip()
        if len(t) > 160:
            t = t[:157] + "..."
        lines.append(f"  - {t}")
    return "\n".join(lines) if lines else "  (none)"


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "  (no web results)"
    lines = []
    for r in results:
        snip = r.snippet.replace("\n", " ").strip()
        if len(snip) > 220:
            snip = snip[:217] + "..."
        lines.append(f"  - {r.title}\n    {r.url}\n    {snip}")
    return "\n".join(lines)


def research_term(
    draft: DraftEntry,
    examples: list[dict],
    llm: LLMClient,
    search: WebSearchTool,
) -> DraftEntry:
    """Run the Tier-3 research loop for a single low-confidence draft."""
    prior_def = draft.senses[0].definition if draft.senses else "(no draft)"

    # 1. Build a domain-aware query.
    query_resp = llm.complete(
        system=_QUERY_SYSTEM,
        user=_QUERY_USER_TEMPLATE.format(
            term=draft.term,
            draft=prior_def,
            snippets=_format_snippets(examples, limit=5),
        ),
        max_tokens=80,
    )
    query = query_resp.text.strip().splitlines()[0].strip()
    if not query:
        return draft

    # 2. Hit the web.
    results = search.search(query, max_results=5)
    if not results:
        draft.reasoning += " [research skipped: no web results]"
        return draft

    # 3. Reconcile with the corpus evidence.
    reconcile_user = _RECONCILE_USER_TEMPLATE.format(
        term=draft.term,
        prior_definition=prior_def,
        prior_confidence=draft.confidence,
        web_snippets=_format_results(results),
        examples=_format_snippets(examples, limit=10),
    )
    data, _ = llm.complete_json(_RECONCILE_SYSTEM, reconcile_user)

    if not data.get("accept_web_info"):
        draft.reasoning += " [research rejected: web info did not fit examples]"
        return draft

    new_def = str(data.get("definition", prior_def)).strip()
    try:
        new_conf = float(data.get("confidence", draft.confidence))
    except (TypeError, ValueError):
        new_conf = draft.confidence

    if draft.senses:
        draft.senses[0] = Sense(
            definition=new_def,
            context_clues=draft.senses[0].context_clues,
            example_message_ids=draft.senses[0].example_message_ids,
        )
    else:
        draft.senses = [Sense(definition=new_def)]
    draft.confidence = max(0.0, min(1.0, new_conf))
    draft.reasoning = str(data.get("reasoning", draft.reasoning)).strip()
    draft.source_stage = "tier3_research"
    return draft