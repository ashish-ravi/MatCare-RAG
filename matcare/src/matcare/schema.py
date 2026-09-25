"""The data contract.

Three types flow through the system:

    Document  — one source fact sheet, as produced by the knowledge-base team
    Chunk     — a retrievable passage cut from a Document; the unit of judgement
    Hit       — one retrieval result, from any retriever

Keeping Hit retriever-agnostic is what lets the dense and BM25 paths share the
evaluation harness without either knowing the other exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    body: str
    source_name: str
    updated: str | None = None
    format: str = "html"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_json(d: dict) -> "Document":
        return Document(
            doc_id      = d["doc_id"],
            title       = d["title"],
            url         = d["url"],
            body        = d["body"],
            source_name = d["sourceName"],
            updated     = d.get("updated"),
            format      = d.get("format", "html"),
            metadata    = d.get("metadata", {}) or {},
        )


@dataclass
class Chunk:
    """A retrievable passage.

    `chunk_id` is the passage identifier used in qrels, so it must be STABLE:
    re-chunking with different parameters invalidates existing judgements.
    See docs/TEST_COLLECTION_SPEC.md for how we handle that.
    """
    chunk_id: str
    doc_id: str
    title: str
    url: str
    source_name: str
    section: str | None
    body: str
    updated: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_mongo(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Hit:
    """One retrieval result. Retriever-agnostic by design."""
    chunk_id: str
    doc_id: str
    score: float
    body: str
    title: str
    url: str
    source_name: str
    section: str | None = None
