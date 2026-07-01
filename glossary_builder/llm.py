"""Thin LLM client wrapper — supports Anthropic and OpenAI.

Provider selection:
  * If ``GLOSSARY_PROVIDER`` env var is set, that wins.
  * Else if ``ANTHROPIC_API_KEY`` is set, Anthropic is used.
  * Else if ``OPENAI_API_KEY`` is set, OpenAI is used.
  * Else: clear error.

The public surface is provider-agnostic: ``complete`` and ``complete_json``.
Downstream stages don't know or care which backend they're talking to.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from .config import LLMConfig


load_dotenv()


# Default model names per provider when GLOSSARY_MODEL isn't set.
_DEFAULTS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "ollama": "qwen3.6:35b",
}


# Per-million-token prices (USD) used purely for budget estimates in the run
# summary. These move; keep this table current or override via env.
# Sources: provider pricing pages, May 2026.
_PRICING_PER_MILLION_USD: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-5":  {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":   {"input": 1.00, "output":  5.00},
    "claude-opus-4-5":    {"input": 15.0, "output": 75.00},
    # OpenAI
    "gpt-4o":             {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15, "output":  0.60},
    "gpt-4.1":            {"input": 2.00, "output":  8.00},
    "gpt-4.1-mini":       {"input": 0.40, "output":  1.60},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort cost estimate. Returns 0.0 for unknown models."""
    rates = _PRICING_PER_MILLION_USD.get(model)
    if not rates:
        return 0.0
    return (
        input_tokens / 1_000_000 * rates["input"]
        + output_tokens / 1_000_000 * rates["output"]
    )


class LLMError(RuntimeError):
    pass


class LLMConfigError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    raw_usage: dict
    model: str


@dataclass
class UsageTotals:
    """Cumulative token / cost counters for an LLM client."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_stage: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "by_stage": {
                k: {**v, "estimated_cost_usd": round(v["estimated_cost_usd"], 4)}
                for k, v in self.by_stage.items()
            },
        }


class LLMClient:
    """Provider-agnostic LLM client.

    Instantiate once and pass around. Thread-safe for the call methods.
    Tracks cumulative usage across all calls — see ``usage`` attribute.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.provider = self._resolve_provider()
        self.model = self._resolve_model()
        self._local = threading.local()   # per-thread client instances
        self.usage = UsageTotals()
        self._usage_lock = threading.Lock()
        self._current_stage = "unknown"

    @property
    def _client(self):
        """Return a per-thread client, creating it on first access."""
        if not hasattr(self._local, "client"):
            self._local.client = self._build_client()
        return self._local.client

    def set_stage(self, stage: str) -> None:
        """Tag subsequent calls with a stage name for per-stage usage stats."""
        self._current_stage = stage

    def _record(self, resp: LLMResponse) -> None:
        cost = estimate_cost_usd(
            resp.model,
            resp.raw_usage.get("input_tokens", 0),
            resp.raw_usage.get("output_tokens", 0),
        )
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += resp.raw_usage.get("input_tokens", 0)
            self.usage.output_tokens += resp.raw_usage.get("output_tokens", 0)
            self.usage.estimated_cost_usd += cost
            stage = self.usage.by_stage.setdefault(self._current_stage, {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            })
            stage["calls"] += 1
            stage["input_tokens"] += resp.raw_usage.get("input_tokens", 0)
            stage["output_tokens"] += resp.raw_usage.get("output_tokens", 0)
            stage["estimated_cost_usd"] += cost

    # ----- provider resolution ---------------------------------------------

    @staticmethod
    def _resolve_provider() -> str:
        explicit = (os.environ.get("GLOSSARY_PROVIDER") or "").strip().lower()
        if os.environ.get("OLLAMA_BASE_URL"):
            return "ollama"
        if explicit in ("anthropic", "openai"):
            return explicit
        if explicit and explicit != "auto":
            raise LLMConfigError(
                f"Unknown GLOSSARY_PROVIDER={explicit!r}. "
                "Use 'anthropic', 'openai', or leave it unset."
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        raise LLMConfigError(
            "No LLM credentials found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "in your .env file."
        )

    def _resolve_model(self) -> str:
        configured = os.environ.get("GLOSSARY_MODEL") or self.config.model
        # The default in LLMConfig is the Anthropic default; only swap if the
        # user didn't override AND we're on OpenAI.
        if configured == "claude-sonnet-4-5" and self.provider == "openai":
            return _DEFAULTS["openai"]
        return configured

    def _build_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise LLMConfigError(
                    "anthropic package not installed. Run: "
                    "pip install -r requirements.txt"
                ) from exc
            return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMConfigError(
                    "openai package not installed. Run: "
                    "pip install -r requirements.txt"
                ) from exc
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.provider == "ollama":
            from openai import OpenAI
            return OpenAI(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://100.97.127.60:11434/v1"),
                api_key="ollama",
            )
        else:
            raise LLMConfigError(f"Unsupported provider: {self.provider}")

    # ----- public API ------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return self._call(
            system=system,
            user=user,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            json_mode=False,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[dict, LLMResponse]:
        """Call the model and parse the response as JSON.

        On OpenAI we use native JSON-mode (``response_format``) for reliability.
        On Anthropic we rely on the tolerant extractor below.
        """
        resp = self._call(
            system=system,
            user=user,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            json_mode=True,
        )
        data = extract_json(resp.text)
        if data is None:
            raise LLMError(f"Could not parse JSON from response:\n{resp.text[:500]}")
        return data, resp

    # ----- internals -------------------------------------------------------

    def _call(self, *, system, user, max_tokens, temperature, json_mode) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                if self.provider == "anthropic":
                    resp = self._call_anthropic(system, user, max_tokens, temperature)
                else:
                    resp = self._call_openai(system, user, max_tokens, temperature, json_mode)
                self._record(resp)
                return resp
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last_exc = exc
                sleep_for = min(30.0, 2 ** attempt)
                time.sleep(sleep_for)
        raise LLMError(
            f"LLM call failed after {self.config.max_retries} attempts: {last_exc}"
        )

    def _call_anthropic(self, system, user, max_tokens, temperature) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "text", None))
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }
        return LLMResponse(text=text, raw_usage=usage, model=self.model)

    def _call_openai(self, system, user, max_tokens, temperature, json_mode) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if self.provider == "ollama":
            kwargs["extra_body"] = {"reasoning_effort": "none"}


        if json_mode and "gemini" not in self.model:
            # OpenAI's JSON mode requires the word "json" to appear in the
            # prompt — every one of our JSON prompts says "Return JSON" so we
            # satisfy that. Belt and braces: append a hint just in case.
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()  # вже імпортований в llm.py
        resp = self._client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        choice = resp.choices[0]
        text = choice.message.content or ""

        # # --- логи ---
        # reasoning = getattr(choice.message, "reasoning", None)
        # stage = self._current_stage
        # print(f"[{stage}] ⏱ {elapsed:.1f}с | reasoning: {reasoning or '—'}")
        # # ------------
        # stage = self._current_stage
        # print(f"[{stage}] raw response: {text[:300]!r}")
        #
        # print(f"[DEBUG] finish_reason: {choice.finish_reason}")
        # print(f"[DEBUG] full choice: {choice}")
        # print(f"[DEBUG] message: {choice.message}")

        usage_obj = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
            "output_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
        }
        return LLMResponse(text=text, raw_usage=usage, model=self.model)


# -------- JSON extraction ---------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)

# Qwen3 thinking models sometimes emit `">` before the JSON object.
_THINKING_PREFIX_RE = re.compile(r'^[\s">]+')


from json_repair import repair_json

def extract_json(text: str) -> Any | None:
    text = text.strip()

    # Strip thinking-mode artifacts like `">` that Qwen3 prepends to output.
    text = _THINKING_PREFIX_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Walk forward to the first { or [ and find its matching closing bracket.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Last resort — json_repair
    try:
        repaired = repair_json(text, return_objects=True)
        if repaired:
            return repaired
    except Exception:
        pass

    return None