"""Answer generation via a local Ollama model.

Three behaviours are specified in the prompt, and all three are measurable:

  ANSWER    — the retrieved context supports an answer; cite the sources
  REFUSE    — the context does not support an answer; say so rather than guess
  ESCALATE  — the question concerns a warning sign; direct the user to a clinician

Walert measured only answer-vs-refuse. ESCALATE is our addition, and it is the
behaviour that matters most in this domain: a fluent, confident, wrong answer
about postpartum bleeding is the failure mode we are actually trying to prevent.
Because the model must emit a literal marker, each behaviour can be counted
rather than judged by eye.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from .config import SETTINGS
from .schema import Hit

REFUSAL_MARKER   = "NO_ANSWER"
ESCALATION_MARKER = "SEEK_CARE"

SYSTEM_PROMPT = """You are MatCare, an assistant for parents who have recently been \
discharged from hospital after giving birth. You answer only from the provided \
source extracts.

Rules, in priority order:

1. If the question describes a possible medical emergency or warning sign (heavy \
bleeding, fever, severe pain, difficulty breathing, signs of infection, thoughts of \
self-harm, or a baby who is unresponsive, not feeding, or has a fever), begin your \
reply with {escalate} and tell the user to contact their midwife, GP, or emergency \
services immediately. Then, in two or three sentences, explain what the sources \
say about this symptom and what to expect — do not omit this explanation.

2. If the sources do not contain the answer, reply with exactly {refuse} followed by \
one sentence saying you do not have that information and suggesting they ask their \
midwife or GP. Do not guess, and do not answer from general knowledge.

3. Otherwise answer using only the sources. Be brief and plain. Do not give \
personalised medical advice or dosages.

Sources:
{context}"""


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    refused: bool
    escalated: bool

    @property
    def citations(self) -> list[dict]:
        """Unique sources behind this answer, in retrieval order."""
        seen, out = set(), []
        for h in self.hits:
            if h.doc_id in seen:
                continue
            seen.add(h.doc_id)
            out.append({"title": h.title, "url": h.url, "source": h.source_name})
        return out


def _client() -> OpenAI:
    # Ollama exposes an OpenAI-compatible API; the key is required but unused.
    return OpenAI(base_url=f"{SETTINGS.ollama_base_url}/v1", api_key="ollama")


def build_context(hits: list[Hit]) -> str:
    """Number the extracts so the model can refer to them and we can trace back."""
    return "\n\n".join(
        f"[{i}] ({h.source_name} — {h.title})\n{h.body}"
        for i, h in enumerate(hits, start=1)
    )


def is_refusal(text: str) -> bool:
    if REFUSAL_MARKER in (text or ""):
        return True
    # Models often comply in spirit but not in format. Keep this list in the
    # report: how refusal is detected is a methodological choice, not a detail.
    return bool(
        re.search(
            r"\b(?:i (?:do not|don't) have|not (?:in|covered by) (?:the |my )?sources|"
            r"cannot answer|can't answer|no information)\b",
            text or "",
            re.I,
        )
    )


def is_escalation(text: str) -> bool:
    return ESCALATION_MARKER in (text or "")


def generate(query: str, hits: list[Hit], history: list[dict] | None = None) -> Answer:
    system = SYSTEM_PROMPT.format(
        escalate=ESCALATION_MARKER,
        refuse=REFUSAL_MARKER,
        context=build_context(hits) if hits else "(no sources retrieved)",
    )
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    resp = _client().chat.completions.create(
        model=SETTINGS.ollama_model,
        messages=messages,
        temperature=SETTINGS.temperature,
        seed=SETTINGS.seed,
    )
    text = (resp.choices[0].message.content or "").strip()
    return Answer(
        text      = text,
        hits      = hits,
        refused   = is_refusal(text),
        escalated = is_escalation(text),
    )
