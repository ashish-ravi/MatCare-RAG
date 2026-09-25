# Architecture

How MatCare works, and why it is built this way.

---

## 1. The two flows

```
INGESTION (offline, run once per KB change)
───────────────────────────────────────────────────────────────────────

  matcare_docs.json          chunking.py              embeddings.py
  (KB team delivers)   ──►   section-aware      ──►   voyage-4-nano      ──┐
  27 fact sheets             split, provenance        (local, 384-d)       │
                             preserved                                     │
                                                                           ▼
                                              MongoDB Local Atlas (Docker container)
                                              ┌──────────────────────────────────┐
                                              │  mongod   knowledge_base         │
                                              │           chat_history           │
                                              │  mongot   vector_index           │
                                              └──────────────────────────────────┘

QUERY (online)
───────────────────────────────────────────────────────────────────────

  user question
        │
        ├──► embeddings.encode_query ──► $vectorSearch ──► top-k chunks
        │                                                      │
        │                                                      ▼
        │                                          generation.py: prompt assembly
        │                                          (system rules + sources + history)
        │                                                      │
        │                                                      ▼
        │                                          Ollama gemma4 (OpenAI-compatible API)
        │                                                      │
        └──────────────────────────────────────────────────────┤
                                                               ▼
                                              Answer(text, citations, refused, escalated)
```

The container bundles **two** processes. `mongod` is the database; `mongot` is the Atlas
Search node that makes `$vectorSearch` possible. Community MongoDB cannot do vector
search at all — that is why the Atlas Local image is required rather than a plain
`mongo` container.

---

## 2. The data contract

Three types, defined in `schema.py`:

| Type | What it is | Produced by |
|---|---|---|
| `Document` | one source fact sheet | KB team, as JSON |
| `Chunk` | a retrievable passage cut from a Document | `chunking.py` |
| `Hit` | one retrieval result | any retriever |

**`Hit` is retriever-agnostic on purpose.** Both the dense and BM25 paths return
`list[Hit]`, so the evaluation harness scores them identically and retrieval strategy
becomes the only variable in the comparison. Adding a re-ranker later means adding a
class that returns `list[Hit]` — nothing downstream changes.

---

## 3. Chunking, and why it isn't just a character split

The reference lab splits at 800 characters with `RecursiveCharacterTextSplitter`. For
generic documentation that is fine. For clinical fact sheets it is a hazard.

Consider a mastitis fact sheet:

```
## Signs and symptoms
A red, painful area on the breast. Flu-like symptoms. A fever above 38.5 degrees.

## When to seek help
Contact your midwife or GP immediately if you have a fever...
```

A blind character split can land the boundary between those two sections. The retriever
then returns the symptom list **without** the escalation advice, and the model answers
"yes, redness and fever are common with mastitis" — true, and dangerous.

So `chunking.py` splits on **headings first**, and character-splits only sections that
are still oversized. Each chunk carries its heading in the indexed text, which also
improves retrieval: the heading is often the clearest statement of what the passage is
about.

This is a decision with a safety consequence, so it is recorded here rather than left
as an inherited default.

---

## 4. Chunk identity and the qrels problem

`chunk_id` is `{doc_id}::{n}`. It is the passage identifier used in relevance
judgements, which creates a dependency:

> **Re-chunking invalidates hand-written judgements.**

Change `chunk_size` and `rwh-mastitis::2` may no longer be the passage someone judged.
Two ways out, and we support both:

**Document-level judgement (default).** The KB team answers "which fact sheet answers
this question?" rather than "which chunk?". `runs.rollup_to_documents()` collapses a
chunk-level run by keeping each document's best-scoring chunk, so the same retrieval
output can be scored either way. Judgements survive any re-chunking.

**Chunk-level judgement.** More precise, and what Walert did — but only safe once
chunking parameters are frozen.

Start at document level. Move to chunk level only if the numbers turn out to be too
coarse to distinguish systems.

---

## 5. Two retrievers, one interface

```python
class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int = 5) -> list[Hit]: ...
```

**`DenseRetriever`** — MongoDB `$vectorSearch` over voyage-4-nano embeddings. Cosine
similarity, ANN with `numCandidates = 10k`. This is the production path.

**`BM25Retriever`** — `rank_bm25` over the same chunks, in process. Deliberately *not*
stored in MongoDB: it is a measuring stick, not a deployment target.

Why keep a lexical baseline? Two reasons.

The brief asks for value demonstrated as a **delta**, not in absolute terms. "Our nDCG
is 0.64" means nothing without a comparison.

More interestingly, we have a hypothesis to test. On Walert, BM25 *beat* dense retrieval
on inferred questions. MatCare should reverse that, because patients write

> *"I'm bleeding a lot, is that normal?"*

while the fact sheet says

> *"Postpartum haemorrhage is defined as..."*

That is a pure vocabulary gap — zero shared terms, so BM25 scores zero, and it is
precisely what dense retrieval exists to cross. If the reversal shows up in the numbers
it is a real, reportable domain finding rather than an assumption.

---

## 6. Three behaviours, all measurable

`generation.py` specifies three behaviours in the system prompt, each emitting a literal
marker so it can be **counted** rather than judged by eye:

| Behaviour | Marker | When |
|---|---|---|
| ANSWER | — | sources support an answer; cite them |
| REFUSE | `NO_ANSWER` | sources do not cover it; say so, don't guess |
| ESCALATE | `SEEK_CARE` | question describes a warning sign; direct to a clinician |

Walert measured only answer-vs-refuse. **ESCALATE is our addition**, and it is the
behaviour that matters most here. A fluent, confident, wrong answer about postpartum
bleeding is the failure mode this whole project exists to prevent.

`is_refusal()` also matches prose refusals, because a model will often comply in spirit
("I don't have that information") without emitting the literal marker. How refusal is
detected is a methodological choice that belongs in the report — a loose matcher inflates
the refusal rate, a strict one deflates it. With a small out-of-KB set, hand-check them.

---

## 7. Evaluation

The part that earns the marks. Component-level and end-to-end, because a single number
tells you *that* something is wrong but never *where*.

```
                    ┌─────────────────────────────────────┐
  test collection   │ questions.csv    known / inferred / │
  (evaluation team) │ qrels.txt        out-of-KB + labels │
                    └─────────────────────────────────────┘
                                    │
   retriever ──► list[Hit] ──► TREC run file ──► metrics.py
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        COMPONENT LEVEL                     END-TO-END LEVEL
        nDCG@k by question type             % refusal on out-of-KB
        (known / inferred)                  % false refusal on answerable
                                            % escalation
                                            attribution precision
```

### The three question types

| Type | Definition | Judgement | Correct behaviour |
|---|---|---|---|
| **known** | answered by one passage | label `2` | retrieve it |
| **inferred** | needs several passages combined | label `1` | retrieve all |
| **out-of-KB** | in-domain, plausible, **not** answerable | none | refuse |

Out-of-KB questions are the load-bearing idea. A system that always produces a fluent
answer is dangerous, not impressive.

### Two things inherited from the Walert reproduction

**nDCG is ported unchanged**, validated against `trec_eval` (query W01Q01 → 0.3521,
exact match). Reusing a validated scorer is what lets us treat retrieval as the only
variable.

**Missing queries score 0, not skipped.** Standard libraries silently omit queries a
system returned nothing for. A system that answers nothing for a known question must
score 0 — it does not get to drop the question from its own average. During the Walert
reproduction, skipping misses inflated intent-based known nDCG from 0.643 to 0.771.

### Why nDCG@k decreases as k grows

Counter-intuitive but correct: the **ideal** DCG is truncated to k as well. At k=1 one
relevant passage at rank 1 scores 1.0; at k=3 you need three in the top three. The bar
rises faster than the score.

### Refusal has two sides

Reporting only "refused 80% of out-of-KB questions" hides a system that refuses
everything. So we report both:

- `refusal_out_of_kb` — **higher is better**
- `false_refusal` on answerable questions — **lower is better**

Moving deliberately along that trade-off is the refusal-calibration story.

---

## 8. What is deliberately absent

**No re-ranker.** A cross-encoder over the top-50 would likely improve ranking, but the
project is graded on demonstrating value, not on pipeline sophistication. Add it only if
the evaluation shows retrieval is the bottleneck.

**No fine-tuning.** Same reason, plus it would make the privacy story worse.

**BM25 is not in MongoDB.** It is a baseline. Putting it in the database would imply it
is a deployment path.

**No cloud anything.** Self-hosting is a **privacy** decision, not merely a cost one.
Postnatal health questions are among the most sensitive data a person generates; sending
them to a third-party API would undermine the product's premise. State this in the report
as a design argument rather than a budget constraint.
