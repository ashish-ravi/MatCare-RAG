"""End-to-end: question in, cited answer out.

Deliberately thin. The pipeline composes a Retriever and the generator; it owns
no retrieval or prompting logic of its own, so swapping dense for BM25 (or
adding a re-ranker later) touches one line.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import SETTINGS
from .db import collections
from .generation import Answer, generate
from .retrieval import DenseRetriever, Retriever


class MatCarePipeline:
    def __init__(self, retriever: Retriever | None = None, session_id: str | None = None):
        self.coll, self.history_coll = collections()
        self.retriever = retriever or DenseRetriever(self.coll)
        self.session_id = session_id

    # ── chat memory (optional) ───────────────────────────────────────────────
    def _history(self) -> list[dict]:
        if not self.session_id:
            return []
        cur = self.history_coll.find(
            {"session_id": self.session_id}, {"_id": 0, "role": 1, "content": 1}
        ).sort("timestamp", 1)
        return [{"role": m["role"], "content": m["content"]} for m in cur]

    def _store(self, role: str, content: str) -> None:
        if not self.session_id:
            return
        self.history_coll.insert_one({
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
        })

    # ── main entry point ─────────────────────────────────────────────────────
    def ask(self, query: str, k: int | None = None) -> Answer:
        hits = self.retriever.search(query, k=k or SETTINGS.top_k)
        answer = generate(query, hits, history=self._history())
        self._store("user", query)
        self._store("assistant", answer.text)
        return answer
