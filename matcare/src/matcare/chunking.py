"""Section-aware chunking.

Why not just RecursiveCharacterTextSplitter at 800 characters?

Our sources are headed clinical fact sheets. A blind character split can put
"heavy bleeding, fever, or severe pain" in one chunk and "call your midwife
immediately" in the next. Retrieving the symptom without the escalation advice
is a safety failure, not a ranking inconvenience — and it is exactly the kind of
failure our out-of-KB / refusal metrics are meant to catch.

So we split on headings first, and fall back to character splitting only for
sections that are still too long. Each chunk carries its section heading in the
indexed text, which also helps retrieval: the heading is often the clearest
statement of what the passage is about.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SETTINGS
from .schema import Chunk, Document

# Markdown headings, or a short ALL-CAPS / Title-Case line on its own.
_HEADING = re.compile(
    r"^(?:#{1,6}\s+(?P<md>.+)|(?P<bare>[A-Z][^\n.!?]{2,70}))\s*$",
    re.MULTILINE,
)


def split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split a document body into (heading, text) pairs.

    Text before the first heading is returned with heading=None.
    """
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [(None, body.strip())]

    sections: list[tuple[str | None, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, m in enumerate(matches):
        heading = (m.group("md") or m.group("bare") or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
    return sections


def chunk_document(
    doc: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Cut one Document into Chunks, preserving all provenance metadata.

    chunk_id is f"{doc_id}::{n}" — stable as long as the document text and the
    chunking parameters do not change.
    """
    size    = chunk_size    or SETTINGS.chunk_size
    overlap = chunk_overlap or SETTINGS.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=size,
        chunk_overlap=overlap,
    )

    chunks: list[Chunk] = []
    n = 0
    for heading, text in split_sections(doc.body):
        # Prefix the heading so the indexed text carries its own context.
        pieces = [text] if len(text) <= size else splitter.split_text(text)
        for piece in pieces:
            indexed = f"{heading}\n\n{piece}" if heading else piece
            chunks.append(
                Chunk(
                    chunk_id    = f"{doc.doc_id}::{n}",
                    doc_id      = doc.doc_id,
                    title       = doc.title,
                    url         = doc.url,
                    source_name = doc.source_name,
                    section     = heading,
                    body        = indexed.strip(),
                    updated     = doc.updated,
                    metadata    = doc.metadata,
                )
            )
            n += 1
    return chunks


def chunk_corpus(docs: list[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out
