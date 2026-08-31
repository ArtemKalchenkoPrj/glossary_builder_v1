"""retry_failed_leads.py

Перезапускає тільки ті повідомлення, які завершились з помилкою
'extraction failed' у попередньому запуску.

Використання:
    python retry_failed_leads.py \
        --xlsx data/your_corpus.xlsx \
        --errors data/output/extract_all_leads_with_errors.jsonl \
        --glossary data/output/glossary.json \
        --out data/output/extract_retried.jsonl

Після завершення об'єднай результати:
    cat data/output/extract_all_leads.jsonl \
        data/output/extract_retried.jsonl \
        > data/output/extract_final.jsonl

Примітка: extract_all_leads.jsonl (pre-filtered) залишаємо без змін,
бо там все ОК. extract_all_leads_with_errors.jsonl вже не потрібен.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_failed_ids(errors_jsonl: Path) -> set[int]:
    """Повертає message_id усіх рядків з 'extraction failed' у rationale."""
    failed: set[int] = set()
    with errors_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "extraction failed" in (d.get("rationale") or ""):
                mid = d.get("message_id")
                if isinstance(mid, int):
                    failed.add(mid)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed lead extractions")
    parser.add_argument("--xlsx", required=True, help="Шлях до .xlsx файлу з корпусом")
    parser.add_argument("--sheet", default=None, help="Назва аркуша (за замовчуванням — перший)")
    parser.add_argument(
        "--errors",
        required=True,
        help="JSONL-файл з попереднього запуску, що містить рядки з 'extraction failed'",
    )
    parser.add_argument("--glossary", required=True, help="Шлях до glossary.json")
    parser.add_argument(
        "--out",
        default="data/output/extract_retried.jsonl",
        help="Куди писати результати retry (JSONL, append-mode з resume)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--db",
        default="data/output/glossary.sqlite",
        help="SQLite DB для intent pre-filter (якщо є)",
    )
    args = parser.parse_args()

    errors_path = Path(args.errors)
    if not errors_path.exists():
        sys.exit(f"Файл не знайдено: {errors_path}")

    print(f"Читаємо failed IDs з {errors_path} …")
    failed_ids = extract_failed_ids(errors_path)
    if not failed_ids:
        print("Жодного 'extraction failed' не знайдено. Виходимо.")
        return
    print(f"Знайдено {len(failed_ids)} повідомлень для retry.")

    # Імпортуємо після аргументів, щоб help-повідомлення не вимагало середовища.
    try:
        from glossary_builder.lead_extraction import LeadExtractor, LeadExtractionConfig, load_glossary
        from glossary_builder.llm import LLMClient
        from glossary_builder.loader import load_messages
    except ImportError as exc:
        sys.exit(
            f"Не вдалося імпортувати glossary_builder: {exc}\n"
            "Переконайся, що запускаєш скрипт з кореневої директорії проекту "
            "де встановлено пакет (pip install -e .)."
        )

    print(f"Завантажуємо повідомлення з {args.xlsx} …")
    messages = load_messages(args.xlsx, sheet=args.sheet)
    print(f"Завантажено {len(messages)} повідомлень.")

    print(f"Завантажуємо глосарій з {args.glossary} …")
    glossary = load_glossary(args.glossary)
    print(f"Глосарій: {len(glossary)} записів.")

    llm = LLMClient()

    cfg = LeadExtractionConfig(
        pre_filter_db_path=args.db,
        # pre-filter і stage1 залишаємо включеними —
        # якщо повідомлення знову пройде через них, то й добре.
        pre_filter_enabled=True,
        stage1_enabled=True,
    )

    extractor = LeadExtractor(glossary=glossary, llm=llm, config=cfg)

    out_path = Path(args.out)
    print(f"\nЗапускаємо retry для {len(failed_ids)} повідомлень → {out_path}")
    print("Resume увімкнено: якщо файл вже існує, пропустимо вже оброблені ID.\n")

    decisions = extractor.extract_batch(
        messages=messages,
        sample_message_ids=failed_ids,   # <-- ключовий параметр
        concurrency=args.concurrency,
        stream_jsonl_path=out_path,
        resume=True,                      # продовжить після краша
        progress_desc="Retry",
    )

    leads = [d for d in decisions if d.is_lead]
    errors = [d for d in decisions if "extraction failed" in (d.rationale or "")]

    print(f"\n✓ Оброблено: {len(decisions)}")
    print(f"  Лідів знайдено: {len(leads)}")
    print(f"  Знову з помилкою: {len(errors)}")
    if errors:
        print("  Перші помилки:")
        for d in errors[:5]:
            print(f"    msg_id={d.message_id}: {d.rationale[:120]}")

    print(f"\nРезультати збережено у {out_path}")
    print("\nЩоб об'єднати фінальний файл:")
    print(f"  cat data/output/extract_all_leads.jsonl {out_path} > data/output/extract_final.jsonl")


if __name__ == "__main__":
    main()
