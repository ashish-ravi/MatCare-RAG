# MatCare

A test-driven RAG assistant for parents discharged from hospital after giving birth.

Answers come **only** from curated Australian maternity fact sheets, with sources cited.
When the sources don't cover a question, the system says so rather than guessing — and
when a question describes a warning sign, it directs the user to a clinician.

Everything runs locally. No API keys, no cost.

> **The point of this project is not the pipeline.** A working RAG pipeline is necessary
> but not sufficient. The claim we have to substantiate is that MatCare adds measurable
> value, which is what [`src/matcare/evaluation/`](src/matcare/evaluation/) is for.

---

## Setup

Five steps. Steps 1–3 are one-time; budget ~20 minutes including downloads.

### 1. Docker Desktop

MongoDB Local Atlas runs as a Docker container, so the Docker daemon must be running.

```bash
docker --version && docker ps
```

If `docker ps` errors with "cannot connect to the Docker daemon", **launch Docker Desktop**
and try again. Install from https://docs.docker.com/get-docker/ if missing.

### 2. Atlas CLI and the local database

The Atlas CLI is a *controller* — it tells Docker to run MongoDB's prebuilt
`mongodb-atlas-local` image. You never write a Dockerfile.

```bash
brew install mongodb-atlas-cli
atlas local setup matcare
```

Answer `With default settings`, then `Skip` for the connection method.

**Then get your actual connection string** — the port is random if 27017 is taken:

```bash
atlas local connect matcare --connectWith connectionString
```

On later sessions you only need to start it:

```bash
atlas local start matcare
```

### 3. Ollama and the language model

```bash
brew install ollama
ollama serve          # leave running in its own terminal
ollama pull gemma4:e4b
```

Use `gemma4:e2b` on a machine with 8 GB RAM.

### 4. Python environment

```bash
uv sync
```

Pre-download the embedding model so the first run isn't a surprise (~5 min):

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('voyageai/voyage-4-nano', trust_remote_code=True); print('cached')"
```

### 5. Configure

```bash
cp .env.example .env
```

Edit `.env` and paste the URI from step 2. **Keep `?directConnection=true`** — without it
the driver tries to resolve the container's internal replica-set hostname from your host
and times out.

---

## Verify

```bash
uv run matcare check
```

Checks MongoDB connectivity, the vector index, the Ollama model, and the embedding model.
Every failure prints the command that fixes it.

---

## Running it

```bash
# build the knowledge base into MongoDB (chunk → embed → index)
uv run matcare ingest

# ask one question
uv run matcare ask "What are the signs of mastitis?"

# with conversation memory
uv run matcare ask "Is that serious?" --session demo-1

# evaluate: dense vs BM25, nDCG by question type
uv run matcare eval --system both
```

---

## Who builds what

| Component | Owner | Contract |
|---|---|---|
| Knowledge base (`data/kb/matcare_docs.json`) | KB team | [`docs/KNOWLEDGE_BASE_SPEC.md`](docs/KNOWLEDGE_BASE_SPEC.md) |
| Test collection (`data/eval/`) | Evaluation team | [`docs/TEST_COLLECTION_SPEC.md`](docs/TEST_COLLECTION_SPEC.md) |
| Pipeline + harness (`src/matcare/`) | Pipeline team | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

The two specs are deliberately written as **contracts**: if `scripts/validate_kb.py`
passes, ingestion will work, and the three teams can work in parallel without blocking
each other.

```bash
python scripts/validate_kb.py data/kb/matcare_docs.json
```

---

## Important: everyone has their own database

Each team member runs their own Local Atlas container with its own data volume. **Nothing
is shared.** The only way five people get the same knowledge base is by all running
`matcare ingest` against the same committed `matcare_docs.json`. Never load data by hand.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Socket not found: /var/run/docker.sock` | Docker Desktop not running | Launch Docker Desktop |
| Connection timeout to MongoDB | missing `?directConnection=true`, or wrong port | Re-run `atlas local connect matcare --connectWith connectionString` |
| `could not connect to ollama server` | Ollama not serving | `ollama serve` |
| `Model 'gemma4:e4b' not found` | model not pulled | `ollama pull gemma4:e4b` |
| Vector index stuck `PENDING` | mongot still building | Wait — it can take 60s on first build |
| `knowledge base not found` | KB team hasn't delivered yet | See `data/kb/README.md` |

---

## Layout

```
matcare/
├── src/matcare/
│   ├── config.py        settings from .env
│   ├── schema.py        Document / Chunk / Hit — the data contract
│   ├── chunking.py      section-aware splitting
│   ├── embeddings.py    voyage-4-nano, local
│   ├── db.py            MongoDB + vector index
│   ├── ingest.py        KB → chunks → embeddings → MongoDB
│   ├── retrieval.py     DenseRetriever | BM25Retriever
│   ├── generation.py    Ollama, prompt, refusal + escalation
│   ├── pipeline.py      end to end
│   ├── cli.py           the `matcare` command
│   └── evaluation/      ← the graded core
│       ├── testset.py   questions + qrels + type derivation
│       ├── runs.py      TREC run format
│       ├── metrics.py   nDCG, refusal, attribution
│       └── evaluate.py  orchestration + reporting
├── data/kb/             knowledge base (KB team)
├── data/eval/           test collection (evaluation team)
├── scripts/             validators
└── docs/                architecture + the two specs
```
