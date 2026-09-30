"""Chunking ablation: section-aware (default) vs naive fixed-800 character split.

Evaluation card, item 6. Runs entirely in-process against BM25 - no MongoDB,
no Ollama needed - so it can be run the moment the knowledge base file exists,
even before the live pipeline is wired up.

Usage (from the `matcare` project folder):
    uv run python -m matcare.evaluation.chunking_ablation
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..chunking import chunk_corpus
from ..config import SETTINGS
from ..ingest import load_documents
from ..retrieval import BM25Retriever
from ..schema import Chunk, Document
from .evaluate import evaluate_retrieval, retrieve_all
from .testset import load_test_collection


def chunk_document_fixed(
    doc: Document, chunk_size: int = 800, chunk_overlap: int = 80
) -> list[Chunk]:
    """The baseline section-aware chunking is compared against: ignore
    headings entirely and split the whole body by character count only.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pieces = splitter.split_text(doc.body)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{n}",
            doc_id=doc.doc_id,
            title=doc.title,
            url=doc.url,
            source_name=doc.source_name,
            section=None,
            body=piece.strip(),
            updated=doc.updated,
            metadata=doc.metadata,
        )
        for n, piece in enumerate(pieces)
    ]


def chunk_corpus_fixed(docs: list[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document_fixed(d, **kw))
    return out


def run_ablation(k: int = 10, cutoffs: tuple[int, ...] = (1, 3, 5)) -> None:
    docs = load_documents()
    tc = load_test_collection()

    variants = {
        "section-aware": chunk_corpus(docs),
        "fixed-800": chunk_corpus_fixed(
            docs, chunk_size=SETTINGS.chunk_size, chunk_overlap=SETTINGS.chunk_overlap
        ),
    }

    for name, chunks in variants.items():
        print(
            f"\n{name}: {len(docs)} documents -> {len(chunks)} chunks "
            f"({len(chunks) / max(len(docs), 1):.1f} per doc)"
        )
        retriever = BM25Retriever(chunks)
        results = retrieve_all(retriever, tc, k=k)
        res = evaluate_retrieval(
            results, tc, tag=f"bm25-{name}", granularity="document", cutoffs=cutoffs
        )
        print(res.report())


if __name__ == "__main__":
    run_ablation()