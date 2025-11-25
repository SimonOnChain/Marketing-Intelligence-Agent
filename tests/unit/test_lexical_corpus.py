from pathlib import Path

import pytest

from src.retrieval.rag_chain import LexicalCorpus


@pytest.fixture()
def corpus_file(tmp_path: Path) -> Path:
    data = [
        {"id": "1", "text": "Shipping was late and the box arrived damaged."},
        {"id": "2", "text": "Battery life is excellent and customers love the comfort."},
        {"id": "3", "text": "Setup was confusing but support eventually helped."},
    ]
    corpus_path = tmp_path / "lexical.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for row in data:
            fh.write(f'{{"id":"{row["id"]}","text":"{row["text"]}"}}\n')
    return corpus_path


def test_lexical_corpus_prefers_matching_docs(corpus_file: Path):
    corpus = LexicalCorpus(corpus_file)
    hits = corpus.search("shipping problems", limit=2)
    assert hits
    assert hits[0]["id"] == "1"

