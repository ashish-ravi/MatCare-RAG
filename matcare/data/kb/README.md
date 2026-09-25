# Knowledge base drop zone

The KB team delivers **`matcare_docs.json`** here.

```bash
python scripts/validate_kb.py data/kb/matcare_docs.json
uv run matcare ingest
```

Format and requirements: [`docs/KNOWLEDGE_BASE_SPEC.md`](../../docs/KNOWLEDGE_BASE_SPEC.md)

`matcare_docs.json` is gitignored by default while it is being worked on. Once it is
final, force-add it so the whole team ingests identical data:

```bash
git add -f data/kb/matcare_docs.json
```
