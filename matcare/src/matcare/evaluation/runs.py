"""TREC run files — the interchange format between retrieval and scoring.

    qid Q0 passage_id rank score tag

Space-separated (qrels are tab-separated — different files, different rules).
Rank starts at 1. The scorer re-sorts by SCORE and ignores the rank column, so
passing rank as the score silently inverts the ranking without erroring.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..config import RUNS_DIR
from ..schema import Hit


def write_run(results: dict[str, list[Hit]], path: Path | str, tag: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, hits in results.items():
            for rank, h in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {h.chunk_id} {rank} {h.score:.6f} {tag}\n")
    return path


def load_run(path: Path | str) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, pid, _rank, score, _tag = line.split()
            run[qid][pid] = float(score)
    return dict(run)


def rollup_to_documents(run: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Collapse chunk-level results to document level, keeping each document's
    best-scoring chunk.

    Why this exists: chunk ids change whenever chunking parameters change, which
    would invalidate hand-written judgements. Judging at DOCUMENT level ("which
    fact sheet answers this?") is both easier for the KB team and stable across
    re-chunking. This lets one run be scored either way.
    """
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for qid, hits in run.items():
        for chunk_id, score in hits.items():
            doc_id = chunk_id.split("::")[0]
            if score > out[qid].get(doc_id, float("-inf")):
                out[qid][doc_id] = score
    return dict(out)


def run_path(name: str) -> Path:
    return RUNS_DIR / f"{name}.txt"
