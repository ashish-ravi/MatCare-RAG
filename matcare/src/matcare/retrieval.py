"""Retrievers.

Two implementations behind one interface:

    DenseRetriever — MongoDB Atlas $vectorSearch over voyage-4-nano embeddings
    BM25Retriever  — rank_bm25 over the same chunks, in-process

Both return list[Hit], so the evaluation harness scores them identically and
retrieval strategy is the only variable. BM25 is deliberately NOT stored in
MongoDB: it is a baseline for measuring the delta, not a production path.

Why keep a lexical baseline at all? On Walert, BM25 beat dense retrieval on
inferred questions. We expect MatCare to reverse that, because patients write
"bleeding a lot" where the fact sheet says "postpartum haemorrhage" — a pure
vocabulary gap that BM25 cannot cross. If that reversal shows up in the numbers
it is a real, reportable finding rather than an assumption.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import SETTINGS
from .embeddings import embed
from .schema import Chunk, Hit


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


# ── dense ────────────────────────────────────────────────────────────────────
class DenseRetriever:
    name = "dense"

    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, k: int = 5) -> list[Hit]:
        qv = embed([query], input_type="query")[0]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": SETTINGS.vector_index,
                    "queryVector": qv,
                    "path": "embedding",
                    "numCandidates": max(k * 10, 100),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "embedding": 0,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return [
            Hit(
                chunk_id    = d["chunk_id"],
                doc_id      = d["doc_id"],
                score       = float(d["score"]),
                body        = d["body"],
                title       = d["title"],
                url         = d["url"],
                source_name = d["source_name"],
                section     = d.get("section"),
            )
            for d in self.collection.aggregate(pipeline)
        ]


# ── lexical baseline ─────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation.

    NOTE: no stemming. On the Walert reproduction, adding Porter stemming
    tripled inferred-question nDCG, so this is a live tuning decision and must
    be recorded in the report, not left as an inherited default.
    """
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], tokenizer=tokenize):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.tokenizer = tokenizer
        self.bm25 = BM25Okapi([tokenizer(c.body) for c in chunks])

    def search(self, query: str, k: int = 5, drop_zero: bool = True) -> list[Hit]:
        scores = self.bm25.get_scores(self.tokenizer(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        hits = []
        for i in order:
            if drop_zero and scores[i] <= 0:
                continue
            c = self.chunks[i]
            hits.append(
                Hit(
                    chunk_id    = c.chunk_id,
                    doc_id      = c.doc_id,
                    score       = float(scores[i]),
                    body        = c.body,
                    title       = c.title,
                    url         = c.url,
                    source_name = c.source_name,
                    section     = c.section,
                )
            )
        return hits
