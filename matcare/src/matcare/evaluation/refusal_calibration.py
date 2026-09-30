"""Refusal calibration: try alternative refusal-instruction wordings and
report the refusal-rate / false-refusal-rate trade-off for each.

Evaluation card, item 7. Needs the LIVE pipeline (MongoDB + Ollama running,
KB already ingested) because it calls the actual generator.

Usage (from the `matcare` project folder):
    uv run python -m matcare.evaluation.refusal_calibration
"""

from __future__ import annotations

import random

from openai import OpenAI
from tqdm import tqdm

from ..config import SETTINGS
from ..db import collections
from ..generation import Answer, is_escalation, is_refusal
from ..retrieval import DenseRetriever
from .metrics import refusal_rate
from .testset import load_test_collection

SAMPLE_SIZE = 30  # cap so this finishes in a reasonable time on CPU

PROMPT_VARIANTS: dict[str, str] = {
    "baseline": (
        "If the sources do not contain the answer, reply with exactly {refuse} "
        "followed by one sentence saying you do not have that information and "
        "suggesting they ask their midwife or GP. Do not guess, and do not "
        "answer from general knowledge."
    ),
    "strict": (
        "If the sources do not contain a direct and complete answer to the "
        "exact question asked, reply with exactly {refuse} followed by one "
        "sentence directing them to their midwife or GP. When in doubt, refuse "
        "rather than infer or combine partial information."
    ),
    "lenient": (
        "If the sources give you enough related information to make a "
        "reasonable, cautious attempt at an answer, do so and note any "
        "uncertainty. Only reply with exactly {refuse} if the sources are "
        "completely unrelated to the question."
    ),
}

BASE_SYSTEM_PROMPT = """You are MatCare, an assistant for parents who have recently been \
discharged from hospital after giving birth. You answer only from the provided \
source extracts.

Rules, in priority order:

1. If the question describes a possible medical emergency or warning sign (heavy \
bleeding, fever, severe pain, difficulty breathing, signs of infection, thoughts of \
self-harm, or a baby who is unresponsive, not feeding, or has a fever), begin your \
reply with {{escalate}} and tell the user to contact their midwife, GP, or emergency \
services immediately. Then, in two or three sentences, explain what the sources \
say about this symptom and what to expect - do not omit this explanation.

2. {refusal_rule}

3. Otherwise answer using only the sources. Be brief and plain. Do not give \
personalised medical advice or dosages.

Sources:
{{context}}"""


def _client() -> OpenAI:
    return OpenAI(base_url=f"{SETTINGS.ollama_base_url}/v1", api_key="ollama")


def generate_with_prompt(query: str, hits, refusal_rule: str) -> Answer:
    context = (
        "\n\n".join(
            f"[{i}] ({h.source_name} - {h.title})\n{h.body}"
            for i, h in enumerate(hits, start=1)
        )
        if hits
        else "(no sources retrieved)"
    )
    system = BASE_SYSTEM_PROMPT.format(refusal_rule=refusal_rule).format(
        escalate="SEEK_CARE", refuse="NO_ANSWER", context=context
    )
    resp = _client().chat.completions.create(
        model=SETTINGS.ollama_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=SETTINGS.temperature,
        seed=SETTINGS.seed,
    )
    text = (resp.choices[0].message.content or "").strip()
    return Answer(text=text, hits=hits, refused=is_refusal(text), escalated=is_escalation(text))


def run_calibration(k: int = 5, sample_size: int = SAMPLE_SIZE) -> None:
    tc = load_test_collection()
    coll, _ = collections()
    retriever = DenseRetriever(coll)

    all_qids = list(tc.questions.keys())
    random.seed(42)
    sample_qids = random.sample(all_qids, min(sample_size, len(all_qids)))
    print(f"sampling {len(sample_qids)} of {len(all_qids)} questions\n")

    print(f"{'variant':<12}{'refusal on out-of-KB':>24}{'false refusal':>18}")
    print("-" * 54)
    for name, rule in PROMPT_VARIANTS.items():
        refused = {}
        for qid in tqdm(sample_qids, desc=name):
            q = tc.questions[qid]
            hits = retriever.search(q.question, k=k)
            answer = generate_with_prompt(q.question, hits, refusal_rule=rule)
            refused[qid] = answer.refused
        sample_out_of_kb = [q for q in tc.out_of_kb if q in refused]
        sample_scorable = [q for q in tc.scorable if q in refused]
        r_out = refusal_rate(refused, sample_out_of_kb)
        r_false = refusal_rate(refused, sample_scorable)
        print(f"{name:<12}{r_out:>23.2f}%{r_false:>17.2f}%")


if __name__ == "__main__":
    run_calibration()