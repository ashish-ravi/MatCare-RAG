"""Knowledge base → chunks → embeddings → MongoDB.

This is a SCRIPT, not a notebook step, on purpose: every team member runs their
own Local Atlas container with its own data volume, so the only way five people
end up with the same knowledge base is by running the same deterministic build.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_corpus
from .config import KB_FILE, SETTINGS
from .db import collections, create_vector_index
from .embeddings import embed, embedding_dimensions
from .schema import Chunk, Document


def load_documents(path: Path = KB_FILE) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"knowledge base not found: {path}\n"
            f"The KB team produces this file — see docs/KNOWLEDGE_BASE_SPEC.md"
        )
    with open(path) as f:
        raw = json.load(f)
    return [Document.from_json(d) for d in raw]


def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks = chunk_corpus(docs)
    print(f"  {len(docs)} documents → {len(chunks)} chunks "
          f"({len(chunks)/max(len(docs),1):.1f} per doc)")
    return chunks


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    vectors = embed([c.body for c in chunks], input_type="document", show_progress_bar=True)
    for c, v in zip(chunks, vectors):
        c.embedding = v
    return chunks


def ingest(rebuild_index: bool = True) -> int:
    docs = load_documents()
    chunks = embed_chunks(build_chunks(docs))

    coll, _ = collections()
    coll.delete_many({})
    coll.insert_many([c.to_mongo() for c in chunks])
    print(f"  inserted {coll.count_documents({})} chunks into "
          f"{SETTINGS.db_name}.{SETTINGS.collection}")

    if rebuild_index:
        create_vector_index(coll, dimensions=embedding_dimensions())
    return len(chunks)


def load_chunks_from_db() -> list[Chunk]:
    """Read chunks back out — used by the BM25 baseline, which indexes the same
    text the dense retriever sees, so the comparison is like for like."""
    coll, _ = collections()
    out = []
    for d in coll.find({}, {"_id": 0, "embedding": 0}):
        out.append(Chunk(**{k: d.get(k) for k in
                            ("chunk_id", "doc_id", "title", "url", "source_name",
                             "section", "body", "updated", "metadata")}))
    return out
