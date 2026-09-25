"""Metrics.

nDCG is ported verbatim from the Walert reproduction, where it was validated
against trec_eval (query W01Q01 on the BM25 run → 0.3521, matching exactly).
Do not rewrite it: reusing a validated scorer unchanged is what lets us treat
retrieval as the only variable when comparing systems.
"""

from __future__ import annotations

import math
from collections import defaultdict


# ── nDCG ─────────────────────────────────────────────────────────────────────
def ndcg(rel_by_id: dict[str, int], ranked_ids: list[str], cut: int | None = None) -> float:
    """trec_eval-compatible nDCG.

    gain     = relevance label (2 = fully relevant, 1 = partial, 0 = unjudged)
    discount = log2(rank + 1), rank starting at 1
    IDCG     = ideal ranking of all judged-relevant passages, truncated to `cut`

    Truncating the ideal as well is why nDCG@k DECREASES as k grows: at k=1 one
    relevant passage at rank 1 scores 1.0, while at k=3 you need three.
    """
    docs = ranked_ids[:cut] if cut else ranked_ids
    dcg = sum(rel_by_id.get(d, 0) / math.log2(i + 2) for i, d in enumerate(docs))

    ideal = sorted(rel_by_id.values(), reverse=True)
    if cut:
        ideal = ideal[:cut]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
              cut: int | None = None) -> dict[str, float]:
    """Per-query nDCG.

    Queries absent from qrels (out-of-KB) are skipped: their IDCG is 0, so nDCG
    is undefined rather than zero. They are scored by refusal rate instead.
    """
    out = {}
    for qid, rel in qrels.items():
        if qid not in run:
            continue
        ranked = sorted(run[qid], key=lambda d: -run[qid][d])
        out[qid] = ndcg(rel, ranked, cut=cut)
    return out


def mean_by_type(per_query: dict[str, float], ids_by_type: dict[str, list[str]]
                 ) -> dict[str, tuple[int, float]]:
    """Average within each question type.

    The denominator is EVERY question of that type, not just those the system
    answered. A system that returns nothing for a known question scores 0 — it
    does not get to drop the question from its own average. Skipping misses
    inflated Walert's intent-based known nDCG from 0.643 to 0.771.
    """
    out = {}
    for t, qids in ids_by_type.items():
        if t == "out_of_kb":          # nDCG undefined; see refusal_rate
            continue
        vals = [per_query.get(q, 0.0) for q in qids]
        if vals:
            out[t] = (len(vals), sum(vals) / len(vals))
    return out


# ── refusal ──────────────────────────────────────────────────────────────────
def refusal_rate(refused_by_qid: dict[str, bool], qids: list[str]) -> float:
    """% of the given questions the system declined to answer.

    On out-of-KB questions, HIGH is good — refusing is correct behaviour.
    On answerable questions this is the FALSE refusal rate, where LOW is good.
    Reporting only the first number hides a system that simply refuses everything.
    """
    if not qids:
        return 0.0
    flags = [bool(refused_by_qid.get(q, False)) for q in qids]
    return 100.0 * sum(flags) / len(flags)


def escalation_rate(escalated_by_qid: dict[str, bool], qids: list[str]) -> float:
    """% of questions routed to a clinician. MatCare's addition beyond Walert."""
    if not qids:
        return 0.0
    flags = [bool(escalated_by_qid.get(q, False)) for q in qids]
    return 100.0 * sum(flags) / len(flags)


# ── attribution ──────────────────────────────────────────────────────────────
def attribution_precision(cited_by_qid: dict[str, list[str]],
                          qrels: dict[str, dict[str, int]]) -> float:
    """Of the documents cited in answers, what fraction are actually relevant?

    A confident answer citing an authoritative-looking but irrelevant source is
    worse than no citation, because it manufactures trust. This measures that.
    """
    num = den = 0
    for qid, cited in cited_by_qid.items():
        rel = {d for d, lbl in qrels.get(qid, {}).items() if lbl > 0}
        if not rel:
            continue                      # out-of-KB: nothing is citable
        for c in cited:
            den += 1
            num += 1 if c in rel else 0
    return num / den if den else 0.0
