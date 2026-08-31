"""RAG module — dynamic few-shot retrieval from labeled example index.

Builds a Chroma vector index from labeled messages and retrieves
the k most similar LEAD and NOT_LEAD examples for each message
before classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import chromadb
from openai import OpenAI


@dataclass
class RagConfig:
    enabled: bool = False
    db_path: str = "data/rag/chroma"
    embedding_model: str = "perplexity/pplx-embed-v1-0.6b"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    k: int = 2  # k прикладів кожного класу (2 LEAD + 2 NOT LEAD)
    collection_name: str = "lead_examples"


class RagIndex:
    def __init__(self, config: RagConfig):
        self.config = config
        self._client = chromadb.PersistentClient(path=config.db_path)
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenRouter."""
        response = self._embed_client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def build(self, examples: list[dict]) -> None:
        """Populate index from a list of labeled examples.

        Each example must have:
            text: str
            label: 0 | 1  (or 'true'/'false' / True/False)
            message_id: str | int (optional)
        """
        if not examples:
            return

        # Normalize labels
        def _to_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))  # 1.0 → 1 → True
            return str(v).lower() in ("1", "true", "yes")

        texts = [e["text"] for e in examples]
        ids = [str(e.get("message_id", i)) for i, e in enumerate(examples)]
        labels = [_to_bool(e["label"]) for e in examples]
        metadatas = [{"is_lead": int(lbl)} for lbl in labels]

        # Embed in batches of 100
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._embed(batch))

        # Upsert into Chroma
        self._collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[rag] indexed {len(examples)} examples "
              f"({sum(labels)} leads, {len(labels) - sum(labels)} not leads)")

    def query(self, text: str) -> dict[str, list[str]]:
        """Return k LEAD and k NOT_LEAD examples most similar to text.

        Returns:
            {"leads": [...texts], "not_leads": [...texts]}
        """
        k = self.config.k
        embedding = self._embed([text])[0]

        leads, not_leads = [], []

        for is_lead, bucket in [(1, leads), (0, not_leads)]:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where={"is_lead": is_lead},
                include=["documents"],
            )
            bucket.extend(results["documents"][0])

        return {"leads": leads, "not_leads": not_leads}

    def count(self) -> int:
        return self._collection.count()


def format_few_shot(examples: dict[str, list[str]]) -> str:
    """Format retrieved examples for injection into the prompt."""
    if not examples["leads"] and not examples["not_leads"]:
        return ""

    lines = ["SIMILAR EXAMPLES FROM LABELED DATA:"]

    if examples["not_leads"]:
        lines.append("\nNOT LEAD:")
        for i, text in enumerate(examples["not_leads"], 1):
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            lines.append(f"  {i}. {snippet}")

    if examples["leads"]:
        lines.append("\nLEAD:")
        for i, text in enumerate(examples["leads"], 1):
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            lines.append(f"  {i}. {snippet}")

    lines.append("")
    return "\n".join(lines)