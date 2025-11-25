"""Hybrid retrieval and RAG chain utilities."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.llm.clients import call_xai_chat


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\W+", text.lower()) if token]


class LexicalCorpus:
    """Tiny BM25-like scorer built on a JSONL corpus."""

    def __init__(self, corpus_path: Path):
        self.records: list[dict] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_len = 0.0
        self._load(corpus_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        lengths: list[int] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                tokens = tokenize(record["text"])
                token_counts = Counter(tokens)
                record["tokens"] = token_counts
                record["length"] = len(tokens)
                self.records.append(record)
                lengths.append(record["length"])
                for token in token_counts:
                    self.doc_freq[token] += 1
        if lengths:
            self.avg_len = sum(lengths) / len(lengths)
        else:
            self.avg_len = 0.0

    @property
    def size(self) -> int:
        return len(self.records)

    def search(self, query: str, limit: int = 20) -> list[dict]:
        if not self.records:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores: list[tuple[float, dict]] = []
        for record in self.records:
            score = 0.0
            for token in tokens:
                score += self._bm25(token, record)
            if score > 0:
                scores.append((score, record))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"id": rec["id"], "text": rec["text"], "score": score}
            for score, rec in scores[:limit]
        ]

    def _bm25(self, token: str, record: dict, k1: float = 1.5, b: float = 0.75) -> float:
        df = self.doc_freq.get(token)
        if not df:
            return 0.0
        tf = record["tokens"].get(token, 0)
        if tf == 0:
            return 0.0
        N = self.size
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        denom = tf + k1 * (1 - b + b * (record["length"] / (self.avg_len or 1)))
        return idf * ((tf * (k1 + 1)) / denom)


class HybridRetriever:
    """Combines vector search from Qdrant with lexical BM25-style scoring."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: SentenceTransformer,
        collection: str,
        lexical_corpus: LexicalCorpus | None = None,
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.lexical = lexical_corpus

    def _vector_results(self, query: str, limit: int) -> list[dict]:
        vector = self.embedder.encode(query).tolist()
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": hit.id, "text": hit.payload["text"], "score": float(hit.score)}
            for hit in hits
        ]

    def retrieve(self, query: str, limit: int = 6) -> list[dict]:
        vector_hits = self._vector_results(query, limit)
        lexical_hits = self.lexical.search(query, limit * 2) if self.lexical else []
        combined = self._rrf(vector_hits, lexical_hits, limit)
        return combined

    def _rrf(self, vector_hits: list[dict], lexical_hits: list[dict], limit: int) -> list[dict]:
        k = 60
        scores: dict[str, float] = {}
        payload: dict[str, dict] = {}

        for rank, item in enumerate(vector_hits):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
            payload[item["id"]] = item

        for rank, item in enumerate(lexical_hits):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
            payload.setdefault(item["id"], item)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [payload[item_id] for item_id, _ in ranked[:limit]]


class RAGChain:
    """End-to-end retrieval + synthesis pipeline."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def invoke(self, question: str, k: int = 6) -> dict:
        docs = self.retriever.retrieve(question, limit=k)
        context = self._format_context(docs)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a marketing intelligence analyst. Answer only using the provided "
                    "context. Cite sources as [1], [2], etc. Prefer actionable recommendations."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        answer = call_xai_chat(messages, max_tokens=700)
        return {"answer": answer, "sources": docs}

    @staticmethod
    def _format_context(docs: Iterable[dict]) -> str:
        formatted = []
        for idx, doc in enumerate(docs, start=1):
            formatted.append(f"[{idx}] {doc['text']}")
        return "\n\n".join(formatted)


def build_rag_chain() -> RAGChain:
    settings = get_settings()
    client = QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    lex_path = Path(settings.data_processed_dir) / "lexical_corpus.jsonl"
    lexical = LexicalCorpus(lex_path) if lex_path.exists() else None
    retriever = HybridRetriever(client, embedder, settings.qdrant_collection, lexical)
    return RAGChain(retriever)

