"""Orchestration: run a retriever (and optionally the generator) over the test
collection, then report the numbers.

Component level  — nDCG by question type      (is retrieval finding the right passages?)
End-to-end level — refusal / escalation / attribution (is the system behaving safely?)

Both levels matter. Retrieval can be perfect while generation garbles the answer,
and generation can read fluently while resting on irrelevant passages. A single
end-to-end number tells you THAT something is wrong but never WHERE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import SETTINGS
from ..retrieval import Retriever
from ..schema import Hit
from .metrics import (attribution_precision, escalation_rate, mean_by_type,
                      ndcg_at_k, refusal_rate)
from .runs import load_run, rollup_to_documents, run_path, write_run
from .testset import OUT_OF_KB, TestCollection


@dataclass
class EvalResult:
    tag: str
    granularity: str
    ndcg: dict[str, dict[str, tuple[int, float]]] = field(default_factory=dict)
    refusal_out_of_kb: float | None = None
    false_refusal: float | None = None
    escalation_out_of_kb: float | None = None
    attribution: float | None = None

    def report(self) -> str:
        lines = [
            "=" * 66,
            f"{self.tag}   (judged at {self.granularity} level)",
            "=" * 66,
            "",
            "nDCG by question type",
            "-" * 66,
            f"{'k':<6}" + "".join(f"{t:>16}" for t in ("known", "inferred")),
        ]
        for k, by_type in self.ndcg.items():
            row = f"{k:<6}"
            for t in ("known", "inferred"):
                n, v = by_type.get(t, (0, 0.0))
                row += f"{v:>11.4f} (n={n})" if n else f"{'—':>16}"
            lines.append(row)

        if self.refusal_out_of_kb is not None:
            lines += [
                "",
                "Behaviour",
                "-" * 66,
                f"  refusal on out-of-KB      {self.refusal_out_of_kb:6.2f}%   (higher is better)",
                f"  false refusal on answerable {self.false_refusal:6.2f}%   (lower is better)",
            ]
            if self.escalation_out_of_kb is not None:
                lines.append(f"  escalation on out-of-KB   {self.escalation_out_of_kb:6.2f}%")
            if self.attribution is not None:
                lines.append(f"  attribution precision     {self.attribution:6.4f}   (cited docs that are relevant)")
        lines.append("")
        return "\n".join(lines)


def retrieve_all(retriever: Retriever, tc: TestCollection, k: int = 10
                 ) -> dict[str, list[Hit]]:
    return {
        qid: retriever.search(q.question, k=k)
        for qid, q in tqdm(tc.questions.items(), desc=f"retrieve:{retriever.name}")
    }


def evaluate_retrieval(
    results: dict[str, list[Hit]],
    tc: TestCollection,
    tag: str,
    granularity: str = "document",
    cutoffs: tuple[int, ...] = (1, 3, 5),
    save: bool = True,
) -> EvalResult:
    path = write_run(results, run_path(tag), tag) if save else None
    run = load_run(path) if path else {
        qid: {h.chunk_id: h.score for h in hits} for qid, hits in results.items()
    }
    if granularity == "document":
        run = rollup_to_documents(run)

    res = EvalResult(tag=tag, granularity=granularity)
    for k in cutoffs:
        res.ndcg[f"@{k}"] = mean_by_type(ndcg_at_k(tc.qrels, run, cut=k), tc.ids_by_type)
    res.ndcg["full"] = mean_by_type(ndcg_at_k(tc.qrels, run), tc.ids_by_type)
    return res


def evaluate_end_to_end(
    answers: dict[str, "object"],      # qid -> generation.Answer
    tc: TestCollection,
    result: EvalResult,
) -> EvalResult:
    refused   = {q: a.refused for q, a in answers.items()}
    escalated = {q: a.escalated for q, a in answers.items()}
    cited     = {q: [c["url"] for c in a.citations] for q, a in answers.items()}

    result.refusal_out_of_kb    = refusal_rate(refused, tc.out_of_kb)
    result.false_refusal        = refusal_rate(refused, tc.scorable)
    result.escalation_out_of_kb = escalation_rate(escalated, tc.out_of_kb)
    result.attribution          = attribution_precision(cited, tc.qrels)
    return result
