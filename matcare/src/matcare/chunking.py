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

# Markdown headings — the reliable signal.
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

# Fallback for documents with no markdown structure: a short line on its own
# that looks like a title.
_BARE_HEADING = re.compile(r"^(?P<bare>[A-Z][^\n.!?]{2,70})\s*$", re.MULTILINE)

# A "heading" ending in one of these is really a wrapped sentence fragment.
# PDF extraction produces plenty of these — e.g. "Women's Emergency Department (see"
# or "Emergency Department if you notice any of" — and promoting them to section
# titles both creates junk chunks and orphans the text that followed.
_DANGLING = re.compile(
    r"(?:[,;:(\[]|\b(?:a|an|the|and|or|but|if|of|to|for|with|from|in|on|at|by|as|"
    r"is|are|was|were|be|been|any|all|your|our|their|this|that|these|those|"
    r"than|then|when|while|which|who|whom|whose|into|onto|upon)\s*)$",
    re.IGNORECASE,
)

# Sections shorter than this are folded into the preceding section rather than
# becoming chunks of their own. A three-word chunk cannot answer anything, and
# it usually means a heading got separated from its own body text.
MIN_SECTION_WORDS = 12


def _is_plausible_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t.split()) > 14:
        return False
    return not _DANGLING.search(t)


def _find_headings(body: str) -> list[tuple[int, int, str]]:
    """Return (start, end, heading_text) for each heading, in document order.

    Markdown headings win outright. The bare-line heuristic is used only when a
    document has essentially no markdown structure, because on a PDF-extracted
    body it produces far more false positives than real headings.
    """
    md = [(m.start(), m.end(), m.group(1).strip()) for m in _MD_HEADING.finditer(body)]
    if len(md) >= 2:
        return md

    bare = [
        (m.start(), m.end(), m.group("bare").strip())
        for m in _BARE_HEADING.finditer(body)
        if _is_plausible_heading(m.group("bare"))
    ]
    return sorted(md + bare)


def split_sections(body: str) -> list[tuple[str | None, str]]:
    """Split a document body into (heading, text) pairs.

    Text before the first heading is returned with heading=None. Sections whose
    body is shorter than MIN_SECTION_WORDS are merged into the previous section,
    keeping their heading inline so nothing is lost.
    """
    headings = _find_headings(body)
    if not headings:
        return [(None, body.strip())]

    raw: list[tuple[str | None, str]] = []
    preamble = body[: headings[0][0]].strip()
    if preamble:
        raw.append((None, preamble))

    for i, (_s, end, heading) in enumerate(headings):
        nxt = headings[i + 1][0] if i + 1 < len(headings) else len(body)
        text = body[end:nxt].strip()
        raw.append((heading, text))

    return _merge_short_sections(raw)


def _merge_short_sections(
    sections: list[tuple[str | None, str]],
    min_words: int = MIN_SECTION_WORDS,
) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    for heading, text in sections:
        too_short = len(text.split()) < min_words
        if too_short and out:
            prev_h, prev_t = out[-1]
            joined = f"{prev_t}\n\n{heading}\n{text}".strip() if heading else f"{prev_t}\n\n{text}".strip()
            out[-1] = (prev_h, joined)
        else:
            out.append((heading, text))

    # A leading short section has no predecessor to merge into; push it forward.
    if len(out) > 1 and len(out[0][1].split()) < min_words:
        h0, t0 = out[0]
        h1, t1 = out[1]
        merged = f"{h0}\n{t0}\n\n{t1}".strip() if h0 else f"{t0}\n\n{t1}".strip()
        out = [(h1, merged)] + out[2:]
    return out


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
