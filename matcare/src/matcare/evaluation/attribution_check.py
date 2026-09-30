"""Attribution precision: of the documents an answer actually cites, what
fraction are truly relevant per the qrels?

Evaluation card, item 4. Needs the LIVE pipeline (MongoDB + Ollama running,
KB already ingested) since it calls the actual generator.

Usage (from the `matcare` project folder):
    uv run python -m matcare.evaluation.attribution_check
"""

from __future__ import annotations

from tqdm import tqdm

from ..db import collections
from ..generation import CITATION_SCORE_FLOOR, generate
from ..retrieval import DenseRetriever
from .metrics import attribution_precision
from .testset import load_test_collection


def cited_doc_ids(answer) -> list[str]:
    """Mirror Answer.citations' filtering logic, but keep doc_id (which the
    citations property itself doesn't expose)."""
    if answer.refused:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for h in answer.hits:
        if h.score < CITATION_SCORE_FLOOR or h.doc_id in seen:
            continue
        seen.add(h.doc_id)
        out.append(h.doc_id)
    return out


def run_check(k: int = 5) -> None:
    tc = load_test_collection()
    coll, _ = collections()
    retriever = DenseRetriever(coll)

    cited_by_qid: dict[str, list[str]] = {}
    for qid in tqdm(tc.scorable, desc="generating"):
        q = tc.questions[qid]
        hits = retriever.search(q.question, k=k)
        answer = generate(q.question, hits)
        cited_by_qid[qid] = cited_doc_ids(answer)

    precision = attribution_precision(cited_by_qid, tc.qrels)
    print(f"\nattribution precision (dense, k={k}, n={len(cited_by_qid)}): {precision:.2%}")


if __name__ == "__main__":
    run_check()