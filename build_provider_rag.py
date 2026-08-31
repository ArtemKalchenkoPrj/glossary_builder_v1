"""Build Chroma RAG index for PSP Provider classifier.

Reads labeled examples from Excel/CSV and builds a vector index
for few-shot retrieval during classification.

Usage:
    python build_provider_rag.py --input rag_examples.xlsx
    python build_provider_rag.py --input rag_examples.csv --db PSP_providers_glossary_builder/data/rag/chroma
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glossary_builder.rag import RagIndex, RagConfig


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Excel or CSV with columns: text, label (true/false)")
@click.option("--db", "db_path",
              default="PSP_providers_glossary_builder/data/rag/chroma",
              help="Chroma DB path for output.")
@click.option("--collection", default="provider_examples",
              help="Chroma collection name.")
@click.option("--api-key-env", default="PSP_PROVIDERS_OPENROUTER_API_KEY",
              help="Env var name for embedding API key.")
def main(input_path, db_path, collection, api_key_env):
    """Build RAG index from labeled provider examples."""

    # ── Load data ─────────────────────────────────────────────────────────
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False,
              "provider": True, "not_provider": False, "not provider": False})
    )

    # Видаляємо рядки без тексту або з невизначеним лейблом
    df = df[df["text"].str.len() > 10].dropna(subset=["label"]).reset_index(drop=True)

    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    print(f"Loaded: {len(df)} examples ({n_pos} provider, {n_neg} not provider)")

    if len(df) == 0:
        print("No valid examples — check input file.")
        return

    # ── Build index ───────────────────────────────────────────────────────
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        # Fallback до OPENAI_API_KEY
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(f"ERROR: neither {api_key_env} nor OPENAI_API_KEY set in environment")
        return

    Path(db_path).mkdir(parents=True, exist_ok=True)

    cfg = RagConfig(
        enabled=True,
        db_path=db_path,
        collection_name=collection,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    rag = RagIndex(cfg)

    examples = []
    for i, (_, row) in enumerate(df.iterrows()):
        examples.append({
            "text": row["text"],
            "label": row["label"],
            "message_id": f"rag_{i}",
        })

    rag.build(examples)

    print(f"\nDone. Chroma DB: {db_path}")
    print(f"Collection: {collection}")
    print(f"Total indexed: {rag.count()}")

    # ── Quick test ────────────────────────────────────────────────────────
    test_text = "Работаю по EU картам, апрув 72%, пишите в лс"
    result = rag.query(test_text)
    print(f"\nTest query: '{test_text}'")
    print(f"  Similar PROVIDER examples: {len(result['leads'])}")
    for t in result["leads"]:
        print(f"    → {t[:100]}")
    print(f"  Similar NOT_PROVIDER examples: {len(result['not_leads'])}")
    for t in result["not_leads"]:
        print(f"    → {t[:100]}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
