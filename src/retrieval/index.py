"""Data indexing pipeline (embeddings + lexical corpus)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings


def iter_chunks(texts: Iterable[str], chunk_size: int = 512, chunk_overlap: int = 50) -> Iterator[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        for chunk in splitter.split_text(cleaned):
            yield chunk


def ensure_collection(client: QdrantClient, dim: int, collection: str) -> None:
    existing = client.get_collections()
    if any(col.name == collection for col in existing.collections):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
    )


def index_reviews(
    parquet_path: Path | None = None,
    batch_size: int = 64,
) -> None:
    settings = get_settings()
    parquet_path = parquet_path or Path(settings.data_processed_dir) / "reviews.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found. Run src/data/etl.py first.")

    df = pd.read_parquet(parquet_path)
    texts = list(iter_chunks(df["review_comment_message"].tolist()))
    if not texts:
        raise RuntimeError("No review chunks produced. Check the ETL outputs.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    ensure_collection(client, vector_dim, settings.qdrant_collection)

    payload_corpus: List[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        points = []
        for chunk_text, vector in zip(batch, embeddings, strict=True):
            point_id = str(uuid.uuid4())
            payload = {"text": chunk_text}
            points.append(rest.PointStruct(id=point_id, vector=vector.tolist(), payload=payload))
            payload_corpus.append({"id": point_id, "text": chunk_text})
        client.upsert(collection_name=settings.qdrant_collection, wait=False, points=points)

    lex_path = Path(settings.data_processed_dir) / "lexical_corpus.jsonl"
    with lex_path.open("w", encoding="utf-8") as fh:
        for item in payload_corpus:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Indexed {len(payload_corpus)} review chunks into Qdrant ({settings.qdrant_collection}).")
    print(f"Wrote lexical corpus to {lex_path}")


if __name__ == "__main__":
    index_reviews()

