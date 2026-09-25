# Test Collection Spec

**For the evaluation team.** Produce two files:

- `data/eval/questions.csv`
- `data/eval/qrels.txt`

This is the part of the project the marks are actually for. The pipeline is necessary but
not sufficient; the test collection is what turns "we built a chatbot" into "we measured
whether it helps."

You do **not** need the pipeline to be finished. Start now.

---

## The three question types

| Type | Definition | Judgement | Correct system behaviour |
|---|---|---|---|
| **known** | answered directly by one document | label `2` | retrieve that document |
| **inferred** | answerable only by combining documents | label `1` on each | retrieve all of them |
| **out-of-KB** | in-domain and plausible, but **not** answerable from the KB | *none* | refuse to answer |

### Out-of-KB is the important one

A chatbot that always produces a confident, fluent answer is **dangerous**, not
impressive. These questions test whether the system knows its own limits.

They must be **in-domain and plausible** — a real parent could ask them. Off-topic
trivia ("who won the 2024 grand final?") tests nothing, because any system refuses that.
The trap is a question that *sounds* covered but isn't:

- *"Can I take ibuprofen while breastfeeding?"* — dosage advice, deliberately not in our KB
- *"How much does a lactation consultant cost in Melbourne?"* — plausible, not covered
- *"Is it safe to drive two weeks after a caesarean?"* — sounds like it should be there

Aim for **15–20** out-of-KB questions, not Walert's 10. With 10, each question is worth
10 percentage points and the estimate is fragile. A larger set is a defensible
contribution in its own right.

### Target sizes

| Type | Target | Why |
|---|---|---|
| known | 50–70 | the bulk |
| inferred | 10–15 | hard to write; don't force them |
| out-of-KB | 15–20 | larger than Walert, for stability |
| **total** | **80–100** | comparable to Walert's 106 |

### Paraphrase variations

Write 2–3 phrasings of the same underlying question, sharing a `topic_id`:

```
Q01A  T01  (blank)  known  What are the signs of mastitis?
Q01B  T01  Q01A     known  How do I know if I have mastitis?
Q01C  T01  Q01A     known  my boob is red and sore and I feel awful, what is it
```

Without these you are testing string matching, not retrieval. Deliberately include
**lay phrasing** — the vocabulary gap between how parents write and how fact sheets are
written is the thing dense retrieval is supposed to cross, and you can only measure that
if you write questions in real language.

---

## `questions.csv`

```csv
question_id,topic_id,variation_of,type,question,gold_answer,notes
Q01A,T01,,known,What are the signs of mastitis?,"A red painful area on the breast, flu-like symptoms, and a fever above 38.5 degrees.",from RWH mastitis fact sheet
Q04A,T04,,out_of_kb,Can I take ibuprofen while breastfeeding?,I don't have that information — ask your midwife or pharmacist.,dosage advice; deliberately not in KB
```

| Column | Required | Notes |
|---|---|---|
| `question_id` | yes | unique, stable |
| `topic_id` | no | groups paraphrases |
| `variation_of` | no | the `question_id` this paraphrases |
| `type` | yes | `known` \| `inferred` \| `out_of_kb` |
| `question` | yes | as a real user would type it |
| `gold_answer` | yes | for known: quote/paraphrase the source. For out-of-KB: the refusal |
| `notes` | no | why you classified it this way — invaluable later |

`data/eval/questions.example.csv` is a working example.

---

## `qrels.txt`

TREC format, **tab**-separated:

```
Q01A	0	rwh-mastitis	2
Q03A	0	rwh-mastitis	1
Q03A	0	rwh-caesarean-home	1
```

```
question_id <TAB> 0 <TAB> doc_id <TAB> label
                  │                     └── 2 = fully answers, 1 = partly answers
                  └── unused legacy column, always 0
```

**Out-of-KB questions get no rows at all.** That absence is how they are identified.

### Judge at document level

Use `doc_id` (`rwh-mastitis`), not chunk ids (`rwh-mastitis::2`).

Chunk ids change whenever chunking parameters change, which would invalidate everything
you wrote. Document-level judgements survive re-chunking, and the harness collapses
chunk-level retrieval to document level automatically.

### Graded relevance is not decoration

- `2` — this document answers the question
- `1` — this document gets you part of the way

Walert's data never mixed labels within a question, so its grading did no work. **Yours
should.** If a question is mostly answered by one fact sheet with useful context in
another, that's `2` and `1` — and that is what makes nDCG the right metric rather than a
binary one.

---

## Validation

The loader cross-checks your declared `type` against the judgements and **fails** on
disagreement:

```
Q04A: declared 'known' but judgements imply 'out_of_kb'
```

It catches the two common authoring errors: an out-of-KB question that accidentally has
judgements, and a known question nobody judged.

```bash
uv run python -c "from matcare.evaluation.testset import load_test_collection as l; print(l().summary())"
```

---

## Process

1. **Read the corpus first.** You cannot write good questions about documents you haven't read.
2. **Known questions as you read.** One or two per document, in the user's words, not the fact sheet's.
3. **Inferred questions second.** Look for pairs of documents that together answer something neither does alone. Don't force these — 10 genuine ones beat 20 contrived.
4. **Out-of-KB questions last**, once you know exactly what the corpus covers. This is the only way to be sure they really aren't answerable.
5. **Judge.** For each known/inferred question, list the `doc_id`s that answer it and label them.
6. **Have someone else check a sample.** Agreement between two judges is worth reporting.

---

## Checklist

- [ ] 80–100 questions, with 15–20 out-of-KB
- [ ] Paraphrase variations for a good share of topics
- [ ] Lay phrasing represented, not just clinical terms
- [ ] Every known/inferred question has qrels; every out-of-KB has none
- [ ] Some questions use mixed `2`/`1` labels
- [ ] `gold_answer` filled for every question
- [ ] `notes` explaining each out-of-KB classification
- [ ] Loader runs without error
- [ ] A second judge has checked a sample
