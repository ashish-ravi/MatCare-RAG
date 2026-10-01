"""MatCare chat interface.

    uv run streamlit run app.py

A thin layer over MatCarePipeline.ask(): everything on screen comes from the
Answer it returns. No retrieval or prompting logic lives here, so the interface
cannot drift from what the evaluation measures.

Each question is answered on its own (no conversation memory). Follow-up
questions currently retrieve with the wrong words — see the backlog item on
query rewriting — so the interface does not pretend to support them yet.
"""

from __future__ import annotations

import re
import time

import streamlit as st

from matcare.config import SETTINGS
from matcare.generation import ESCALATION_MARKER, REFUSAL_MARKER

st.set_page_config(page_title="MatCare", page_icon="🤱", layout="centered")

# Chosen for the demo, and each checked against the live system: a plain question,
# everyday wording (the fact sheet says "umbilical", not "belly button"), a warning
# sign that should escalate, and one the fact sheets do not cover, which should be
# refused. "my baby looks a bit yellow" was tried and dropped: it escalated while
# its own explanation called jaundice "often a normal event" — a red banner over
# "this is usually normal" reads as a contradiction.
EXAMPLES = [
    "What are the signs of mastitis?",
    "I had gestational diabetes. What follow-up do I need after birth?",
    "I'm soaking a pad every hour and feel dizzy",
    "Am I eligible for paid parental leave?",
]


# ── backend, loaded once per server process ──────────────────────────────────
@st.cache_resource(show_spinner="Starting MatCare — loading the models…")
def get_pipeline():
    """Build the pipeline once and keep it for every visitor and every click.

    The CLI reloads the embedding model on each `matcare ask` (~10 s) because each
    call is a new process. Streamlit re-runs this whole script on every click, so
    without caching it would be worse still. Cached, the model loads exactly once.
    """
    from matcare.embeddings import embedding_dimensions
    from matcare.pipeline import MatCarePipeline

    pipeline = MatCarePipeline()   # no session_id → no conversation memory
    embedding_dimensions()         # warm the model now, not on the first question
    return pipeline


# ── turning an Answer into something to show ─────────────────────────────────
_MARKER = re.compile(rf"\b(?:{ESCALATION_MARKER}|{REFUSAL_MARKER})\b")


def display_text(text: str) -> str:
    """Remove the marker words — they exist to be counted, not read.

    The model writes them inline ("SEEK_CARE and contact your midwife ..."), so
    the leftover joining word is dropped too and the sentence recapitalised.
    """
    t = _MARKER.sub("", text)
    t = re.sub(r"^[\s.,:;\-–—]*(?:and\s+)?", "", t).strip()
    return t[:1].upper() + t[1:]


# The opening "contact your midwife…" sentence of an escalation. Removed on screen
# because the red banner already says it — in fixed wording. Left to the model,
# this sentence varies, and has come out as a literal copy of the instruction:
# "Tell the user to contact their midwife, GP, or emergency services immediately."
_CONTACT_LEAD = re.compile(
    r"^(?:tell the user to\s+|please\s+)?(?:contact|call|see)\b[^.]*?"
    r"\b(?:midwife|gp|doctor|emergency|000)\b[^.]*\.\s*",
    re.IGNORECASE,
)


def escalation_body(text: str) -> str:
    body = _CONTACT_LEAD.sub("", text, count=1).strip()
    if not body:                 # nothing but the contact sentence: keep it
        return text
    return body[:1].upper() + body[1:]


def kind_of(answer) -> str:
    # Escalation wins: a warning sign matters more than whether it was covered.
    if answer.escalated:
        return "escalate"
    if answer.refused:
        return "refuse"
    return "answer"


def friendly_error(e: Exception) -> str:
    name = type(e).__name__
    if name in ("APIConnectionError", "ConnectError", "APITimeoutError"):
        return "The language model isn't responding. Start it with `ollama serve`, then try again."
    if name == "NotFoundError":
        return (f"The language model `{SETTINGS.ollama_model}` isn't installed. "
                f"Run `ollama pull {SETTINGS.ollama_model}`.")
    if "ServerSelection" in name or "AutoReconnect" in name:
        return ("The database isn't responding. Check Docker Desktop is running, then "
                "`atlas local start matcare`.")
    return f"Something went wrong ({name}: {e}). `uv run matcare check` will diagnose it."


def answer_message(pipeline, question: str) -> dict:
    start = time.perf_counter()
    try:
        a = pipeline.ask(question)
    except Exception as e:  # show it in the chat rather than as a traceback
        return {"role": "assistant", "kind": "error", "text": friendly_error(e),
                "citations": [], "hits": [], "seconds": 0.0}
    kind = kind_of(a)
    text = display_text(a.text)
    if kind == "escalate":
        text = escalation_body(text)
    return {
        "role": "assistant",
        "kind": kind,
        "text": text,
        "citations": a.citations,
        "hits": a.hits,
        "seconds": time.perf_counter() - start,
    }


def render_assistant(msg: dict, show_internals: bool) -> None:
    kind = msg["kind"]
    if kind == "error":
        st.warning(msg["text"], icon="🔌")
        return
    if kind == "escalate":
        st.error("**This may need medical attention.** Contact your midwife or GP now, "
                 "or call **000** in an emergency.", icon="⚠️")
        st.markdown(msg["text"])
    elif kind == "refuse":
        # A refusal shows no sources: "I don't know" followed by links would
        # suggest the answer was in them.
        st.info(msg["text"], icon="ℹ️")
    else:
        st.markdown(msg["text"])

    if msg["citations"]:
        st.markdown("**Sources**")
        for c in msg["citations"]:
            st.markdown(f"- {c['source']} — [{c['title']}]({c['url']})")

    if show_internals and msg["hits"]:
        cited_urls = {c["url"] for c in msg["citations"]}
        label = f"How it found this — {len(msg['hits'])} passages, {msg['seconds']:.1f} s"
        with st.expander(label, icon="🔎"):
            st.caption("The passages the search returned, closest match first. Only these "
                       "were given to the language model; it was told to use nothing else.")
            for i, h in enumerate(msg["hits"], start=1):
                tag = "cited" if h.url in cited_urls else "not cited"
                st.markdown(f"**[{i}] similarity {h.score:.3f}** · {h.source_name} — "
                            f"{h.title} · *{h.section or '—'}* · `{tag}`")
                body = h.body if len(h.body) <= 400 else h.body[:400] + "…"
                st.caption(body)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Try asking")
    clicked = None
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            clicked = q

    st.divider()
    if st.button("New chat", width="stretch", type="primary"):
        st.session_state.messages = []
    show_internals = st.toggle(
        "Show how it found this",
        help="Reveal the fact-sheet passages each answer was built from.",
    )

    st.divider()
    st.caption(
        "Answers come only from 27 official Australian fact sheets — The Royal "
        "Women's Hospital, Pregnancy Birth & Baby, Raising Children Network and others. "
        f"Everything runs on this computer (model: `{SETTINGS.ollama_model}`); "
        "no question leaves the machine."
    )


# ── header ───────────────────────────────────────────────────────────────────
st.title("🤱 MatCare")
st.markdown("Questions about the first weeks after birth, answered from official "
            "Australian fact sheets.")
st.warning("MatCare gives general information, not medical advice. "
           "If you are worried, contact your midwife or GP. In an emergency call **000**.",
           icon="🩺")


# ── start-up checks ──────────────────────────────────────────────────────────
try:
    pipeline = get_pipeline()
except Exception as e:
    st.error("**MatCare couldn't start.**")
    st.markdown(
        "Check that:\n"
        "1. Docker Desktop is running, and `atlas local start matcare` has been run\n"
        "2. `.env` has the current port — `atlas local connect matcare --connectWith connectionString`\n"
        "3. `uv run matcare check` passes"
    )
    st.caption(f"{type(e).__name__}: {e}")
    st.stop()

if pipeline.coll.estimated_document_count() == 0:
    st.warning("The knowledge base is empty. Run `uv run matcare ingest` first.")
    st.stop()


# ── conversation ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages and not clicked:
    st.markdown("Ask about recovery, feeding, sleep or warning signs — or pick an "
                "example on the left.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_assistant(msg, show_internals)

typed = st.chat_input("Ask a question…")
question = clicked or typed

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the fact sheets and writing an answer…"):
            msg = answer_message(pipeline, question)
        render_assistant(msg, show_internals)
    st.session_state.messages.append(msg)
