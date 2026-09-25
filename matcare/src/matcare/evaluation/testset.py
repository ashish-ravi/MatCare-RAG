"""The test collection: questions, judgements, and the three question types.

Walert shipped no question-type column, so the split had to be derived from the
qrels. We carry an explicit `type` column AND re-derive it as a cross-check,
because a question mislabelled by hand is invisible otherwise.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import QRELS_FILE, QUESTIONS_FILE

KNOWN, INFERRED, OUT_OF_KB = "known", "inferred", "out_of_kb"
VALID_TYPES = {KNOWN, INFERRED, OUT_OF_KB}


@dataclass
class Question:
    question_id: str
    question: str
    type: str
    topic_id: str = ""
    variation_of: str = ""
    gold_answer: str = ""
    notes: str = ""


@dataclass
class TestCollection:
    questions: dict[str, Question]
    qrels: dict[str, dict[str, int]]
    ids_by_type: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        by_type = defaultdict(list)
        for q in self.questions.values():
            by_type[q.type].append(q.question_id)
        self.ids_by_type = dict(by_type)

    @property
    def scorable(self) -> list[str]:
        """Questions that have judgements — known + inferred."""
        return self.ids_by_type.get(KNOWN, []) + self.ids_by_type.get(INFERRED, [])

    @property
    def out_of_kb(self) -> list[str]:
        return self.ids_by_type.get(OUT_OF_KB, [])

    def summary(self) -> str:
        counts = {t: len(v) for t, v in sorted(self.ids_by_type.items())}
        return (f"{len(self.questions)} questions "
                f"({' / '.join(f'{v} {k}' for k, v in counts.items())}), "
                f"{sum(len(v) for v in self.qrels.values())} judgements")


def load_qrels(path: Path = QRELS_FILE) -> dict[str, dict[str, int]]:
    """TREC qrels: qid <tab> 0 <tab> passage_id <tab> label.

    Note the separator: qrels are TAB-separated, run files are SPACE-separated.
    """
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            qid, _, pid, label = line.split()
            qrels[qid][pid] = int(label)
    return dict(qrels)


def load_questions(path: Path = QUESTIONS_FILE) -> dict[str, Question]:
    if not path.exists():
        raise FileNotFoundError(
            f"test collection not found: {path}\n"
            f"See docs/TEST_COLLECTION_SPEC.md"
        )
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qtype = row["type"].strip()
            if qtype not in VALID_TYPES:
                raise ValueError(
                    f"{row['question_id']}: type '{qtype}' not one of {sorted(VALID_TYPES)}"
                )
            out[row["question_id"]] = Question(
                question_id  = row["question_id"].strip(),
                question     = row["question"].strip(),
                type         = qtype,
                topic_id     = row.get("topic_id", "").strip(),
                variation_of = row.get("variation_of", "").strip(),
                gold_answer  = row.get("gold_answer", "").strip(),
                notes        = row.get("notes", "").strip(),
            )
    return out


def derive_type(qid: str, qrels: dict[str, dict[str, int]]) -> str:
    """Walert's rule: label 2 → known, label 1 → inferred, no judgements → out-of-KB."""
    labels = set(qrels.get(qid, {}).values())
    if not labels:
        return OUT_OF_KB
    if 1 in labels:
        return INFERRED
    return KNOWN


def load_test_collection(
    questions_path: Path = QUESTIONS_FILE,
    qrels_path: Path = QRELS_FILE,
    strict: bool = True,
) -> TestCollection:
    """Load and cross-check.

    With strict=True, a declared type that disagrees with the judgements is an
    error. That catches the two most common authoring mistakes: an out-of-KB
    question that accidentally has judgements, and a known question nobody
    judged.
    """
    questions = load_questions(questions_path)
    qrels = load_qrels(qrels_path)

    problems = []
    for qid, q in questions.items():
        derived = derive_type(qid, qrels)
        if derived != q.type:
            problems.append(
                f"  {qid}: declared '{q.type}' but judgements imply '{derived}'"
            )
    orphans = [q for q in qrels if q not in questions]
    if orphans:
        problems.append(f"  qrels reference unknown question ids: {orphans[:5]}")

    if problems:
        msg = "test collection inconsistencies:\n" + "\n".join(problems)
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg)

    return TestCollection(questions=questions, qrels=qrels)
