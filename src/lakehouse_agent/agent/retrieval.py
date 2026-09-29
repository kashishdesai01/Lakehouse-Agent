"""Document retrieval. Local: TF-IDF over chunks. Databricks: Mosaic AI Vector Search."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def chunk_document(doc_id: str, text: str, max_chars: int = 400) -> list[Chunk]:
    """Split on paragraph boundaries, then pack paragraphs up to max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        chunks.append(buf)
    return [Chunk(doc_id, f"{doc_id}#{i}", c) for i, c in enumerate(chunks)]


def load_chunks(docs_dir: str) -> list[Chunk]:
    out: list[Chunk] = []
    for fn in sorted(os.listdir(docs_dir)):
        if fn.endswith(".md"):
            with open(os.path.join(docs_dir, fn)) as f:
                out.extend(chunk_document(fn[:-3], f.read()))
    return out


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[dict]: ...


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {"doc_id": self.chunks[i].doc_id, "chunk_id": self.chunks[i].chunk_id,
             "text": self.chunks[i].text, "score": round(float(scores[i]), 4)}
            for i in order if scores[i] > 0
        ]


class DatabricksVectorSearchRetriever:
    """Adapter over a Databricks Vector Search index."""

    def __init__(self, endpoint: str, index_name: str):
        from databricks.vector_search.client import VectorSearchClient

        self.index = VectorSearchClient().get_index(endpoint_name=endpoint, index_name=index_name)

    def search(self, query: str, k: int = 3) -> list[dict]:
        res = self.index.similarity_search(
            query_text=query, columns=["doc_id", "chunk_id", "text"], num_results=k)
        return [{"doc_id": r[0], "chunk_id": r[1], "text": r[2], "score": float(r[-1])}
                for r in res["result"]["data_array"]]
