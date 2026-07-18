"""fusion.py — multi-model panel for lead extraction."""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import time
from openai import AsyncOpenAI

from .llm import extract_json
import logging

logger = logging.getLogger(__name__)

import threading

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            t = threading.Thread(target=_loop.run_forever, daemon=True)
            t.start()
    return _loop

@dataclass
class FusionConfig:
    panel_models: list[str]
    judge_model: str


    consensus_lead_goes_to_judge: bool = True
    # False = lead+lead->skip (mark as REAL_LEAD)
    # True  = lead+lead->judge

    concurrency: int = 8
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""

@dataclass
class PanelResult:
    model: str
    status: str          # "ok" | "error" | "parse_error"
    data: dict | None
    raw: str | None
    elapsed_ms: float
    error: str | None = None

    @property
    def is_lead(self) -> bool | None:
        if self.status != "ok" or not self.data:
            return None
        return bool(self.data.get("is_lead", False))


def decide(
    results: list[PanelResult],
    config: FusionConfig,
) -> tuple[str, list[PanelResult]]:
    votes = [r.is_lead for r in results if r.status == "ok"]

    # Логуємо результати панелі
    for r in results:
        if r.status == "ok":
            logger.info(f"[panel] {r.model}: is_lead={r.is_lead} | {r.data.get('rationale', '')}")
        else:
            logger.warning(f"[panel] {r.model}: status={r.status} | {r.error or r.raw}")

    if not votes:
        logger.info("[decide] outcome=disagreement (all models failed)")
        return "disagreement", results

    if len(votes) == 1:
        logger.info("[decide] outcome=disagreement (one model failed)")
        return "disagreement", results

    all_lead = all(v is True for v in votes)
    all_not_lead = all(v is False for v in votes)

    if all_not_lead:
        logger.info("[decide] outcome=consensus_not_lead")
        return "consensus_not_lead", results

    if all_lead:
        if config.consensus_lead_goes_to_judge:
            logger.info("[decide] outcome=disagreement (consensus_lead_goes_to_judge=True)")
            return "disagreement", results
        logger.info("[decide] outcome=consensus_lead → judge skipped")
        return "consensus_lead", results

    logger.info(f"[decide] outcome=disagreement (votes={votes})")
    return "disagreement", results

async def _call_one_model(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
) -> PanelResult:
    start = time.monotonic()
    logger.info(f"[call_start] {model}")
    try:
        extra = {}
        if "gpt-oss" in model:
            extra["extra_body"] = {"reasoning_effort": "none"}
            logger.info(f"[reasoning_disabled] {model} → reasoning_effort=none")
        if "qwen3" in model.lower():
            extra["extra_body"] = {"thinking": {"type": "disabled"}}
            logger.info(f"[thinking_disabled] {model}")
        if "deepseek" in model.lower():
            extra["extra_body"] = {"reasoning_effort": "none"}
            logger.info(f"[reasoning_disabled] {model}")

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=500,
            temperature=0.0,
            **extra,
        )
        elapsed = (time.monotonic() - start) * 1000
        logger.info(f"[call_done] {model} in {elapsed:.0f}ms")

        text = response.choices[0].message.content or ""
        logger.debug(f"[raw_response] {model}:\n{text[:300]}")

        # Логуємо usage якщо є
        usage = getattr(response, 'usage', None)
        if usage:
            logger.info(
                f"[usage] {model}: "
                f"input={getattr(usage, 'prompt_tokens', '?')} "
                f"output={getattr(usage, 'completion_tokens', '?')} "
                f"reasoning={getattr(usage, 'reasoning_tokens', 0)}"
            )

        data = extract_json(text)
        if data is None:
            logger.warning(f"[parse_error] {model} raw:\n{text[:500]}")
            return PanelResult(
                model=model,
                status="parse_error",
                data=None,
                raw=text[:300],
                elapsed_ms=elapsed,
            )

        # Логуємо якщо data є списком
        if isinstance(data, list):
            logger.warning(f"[list_response] {model} повернув список з {len(data)} елементів — беремо перший")
            data = data[0] if data else {}

        return PanelResult(
            model=model,
            status="ok",
            data=data,
            raw=text,
            elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        logger.error(f"[call_error] {model} after {elapsed:.0f}ms: {e}")
        return PanelResult(
            model=model,
            status="error",
            data=None,
            raw=None,
            elapsed_ms=elapsed,
            error=str(e),
        )


async def _run_panel_async(
    client: AsyncOpenAI,
    config: FusionConfig,
    system: str,
    user: str,
) -> list[PanelResult]:
    tasks = [
        _call_one_model(client, model, system, user)
        for model in config.panel_models
    ]
    return list(await asyncio.gather(*tasks))


def run_panel(config: FusionConfig, system: str, user: str) -> list[PanelResult]:
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(
            _run_panel_async(client, config, system, user), loop
    )
    return future.result(timeout=60)


class FusionExtractor:
    def __init__(self, config: FusionConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def run(
            self,
            system: str,
            user: str,
    ) -> tuple[dict, str]:
        """Паралельний виклик панелі + decide.

        Повертає (data, raw) — той самий формат що і llm.complete_json,
        щоб _build_decision не знав різниці.
        """
        results = run_panel(self.config, system, user)
        outcome, results = decide(results, self.config)

        if outcome == "consensus_not_lead":
            ok = next((r for r in results if r.status == "ok"), None)
            if ok:
                data = ok.data[0] if isinstance(ok.data, list) else ok.data
                return data, ok.raw or ""
            return {"is_lead": False, "rationale": "[fusion: both models failed]"}, ""

        if outcome == "consensus_lead":
            # Обидві сказали lead, суддя пропускається
            # Мерджимо: беремо geo/vertical/methods від обох моделей
            return _merge_lead_results(results), _pick_raw(results)

        # disagreement — йдемо на суддю
        return self._call_judge(system, user, results)

    def _call_judge(self, system, user, panel_results):
        judge_system = (
                "You are an ARBITRATOR, not a classifier. "
                "Two models have already analyzed the message and provided their verdicts. "
                "Your job is to review BOTH verdicts and their reasoning, then make the FINAL decision. "
                "Do NOT ignore the panel results — they are the primary input. "
                "If both models agree, you should have a very strong reason to disagree. "
                "Focus on which model's reasoning is more consistent with the lead definition.\n\n"
                + system  # додаємо оригінальний system з lead_definition як довідку
        )
        judge_user = _format_judge_prompt(user, panel_results)

        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {"role": "system", "content": judge_system},
                    {"role": "user", "content": judge_user},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

        loop = _get_loop()
        future = asyncio.run_coroutine_threadsafe(_call(), loop)
        raw = future.result(timeout=60)
        data = extract_json(raw)
        if isinstance(data, list):
            data = data[0] if data else None

        if data:
            logger.info(
                f"[judge] {self.config.judge_model}: is_lead={data.get('is_lead')} | {data.get('rationale', '')[:80]}")
        else:
            logger.warning(f"[judge] parse error | raw={raw[:200]}")

        if data is None:
            return {"is_lead": False, "rationale": "[fusion judge: parse error]"}, raw

        data = _merge_with_panel(data, panel_results)
        return data, raw

def _merge_with_panel(judge_data: dict, panel_results: list[PanelResult]) -> dict:
    """Доповнює результат судді geo/vertical/methods з панелі."""
    for r in panel_results:
        if r.status != "ok" or not r.data:
            continue
        for field in ("geo", "vertical", "payment_methods_mentioned"):
            base_val = judge_data.get(field) or []
            panel_val = r.data.get(field) or []
            judge_data[field] = list(dict.fromkeys(base_val + panel_val))
    return judge_data

def _merge_lead_results(results: list[PanelResult]) -> dict:
    ok_results = [r for r in results if r.status == "ok" and r.data]
    if not ok_results:
        return {"is_lead": False}

    def _unwrap(data):
        if isinstance(data, list):
            return data[0] if data else {}
        return data

    base = _unwrap(ok_results[0].data).copy()

    for r in ok_results[1:]:
        r_data = _unwrap(r.data)
        for field in ("geo", "vertical", "payment_methods_mentioned"):
            base_val = base.get(field) or []
            new_val = r_data.get(field) or []
            base[field] = list(dict.fromkeys(base_val + new_val))

    confidences = [_unwrap(r.data).get("confidence", 0.0) for r in ok_results]
    base["confidence"] = round(sum(confidences) / len(confidences), 3)

    rationale_parts = []
    for r in ok_results:
        r_data = _unwrap(r.data)
        conf = r_data.get("confidence", 0.0)
        rat = r_data.get("rationale", "").strip()
        rationale_parts.append(f"{conf} ({r.model}): {rat}")
    base["rationale"] = " | ".join(rationale_parts)

    return base


def _pick_raw(results: list[PanelResult]) -> str:
    """Бере raw від першої успішної моделі."""
    ok = next((r for r in results if r.status == "ok"), None)
    return ok.raw or "" if ok else ""


def _format_judge_prompt(original_user: str, results: list[PanelResult]) -> str:
    """Формує промпт для судді з результатами панелі."""
    lines = [
        "Two models analyzed this message independently. Their verdicts differ.",
        "Review their reasoning and make the final decision.",
        "",
        "ORIGINAL MESSAGE:",
        original_user,
        "",
        "PANEL RESULTS:",
    ]
    for r in results:
        lines.append(f"\nModel: {r.model} | status: {r.status}")
        if r.status == "ok" and r.data:
            lines.append(f"is_lead: {r.data.get('is_lead')}")
            lines.append(f"rationale: {r.data.get('rationale', '')}")
        elif r.error:
            lines.append(f"error: {r.error}")
    lines.append("\nReturn the same JSON format as the panel models.")
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import json
    from dotenv import load_dotenv
    from glossary_builder.lead_prompt_versions import BY_VERSION
    from glossary_builder.lead_extraction import (
        _SYSTEM_TEMPLATE,
        _USER_TEMPLATE,
        LeadExtractionConfig,
    )
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    load_dotenv()

    cfg = LeadExtractionConfig()
    cfg.lead_definition = BY_VERSION["v5-01-short"]
    system = _SYSTEM_TEMPLATE.replace("{lead_definition}", cfg.lead_definition)

    user = _USER_TEMPLATE.format(
        glossary_block="(none)",
        context_block="(none)",
        message_id=12345,
        username="testuser",
        timestamp="2026-05-20T10:00:00",
        text="Привет нужен USDT TRC20 в большом объеме, условия обсудим.",
    )

    config = FusionConfig(
        panel_models=["meta-llama/llama-3.2-1b-instruct", "openai/gpt-oss-20b"],
        judge_model="google/gemini-2.5-flash-lite",
        api_key=os.environ["OPENAI_API_KEY"],
        consensus_lead_goes_to_judge=False,
    )

    extractor = FusionExtractor(config)
    data, raw = extractor.run(system, user)

    print("=== РЕЗУЛЬТАТ ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))