# Knowledge Base Spec

**For the KB team.** Produce one file: `data/kb/matcare_docs.json`.

If `python scripts/validate_kb.py data/kb/matcare_docs.json` passes, ingestion will work.
That is the whole contract — you don't need to read any pipeline code.

---

## Format

A JSON **array** of document objects:

```json
[
  {
    "doc_id": "rwh-mastitis",
    "title": "Mastitis",
    "url": "https://www.thewomens.org.au/images/uploads/fact-sheets/Mastitis-231122.pdf",
    "sourceName": "The Royal Women's Hospital",
    "updated": "2023-11-22T00:00:00Z",
    "format": "pdf",
    "body": "# Mastitis\n\nMastitis is inflammation of the breast tissue...\n\n## Signs and symptoms\n\n...",
    "metadata": {
      "category": "Feeding and breastfeeding",
      "retrieved": "2026-09-26",
      "publisher_type": "hospital"
    }
  }
]
```

### Required

| Field | Notes |
|---|---|
| `doc_id` | short slug, unique, stable. `rwh-mastitis`, `pbb-postpartum-haemorrhage`. **Never change it after judgements are written** |
| `title` | as published |
| `url` | absolute, and the real one — it is shown to users as a citation |
| `sourceName` | publisher, e.g. `The Royal Women's Hospital` |
| `body` | cleaned full text (see below) |

### Optional but wanted

| Field | Notes |
|---|---|
| `updated` | ISO 8601. Publication/revision date from the document. Shown to users — "updated 2023" matters for clinical guidance |
| `format` | `pdf` or `html` |
| `metadata.category` | one of the five below |
| `metadata.retrieved` | date you fetched it (provenance) |
| `metadata.publisher_type` | `hospital`, `government`, `ngo`, `professional-body` |

### Categories

Use exactly these five, matching the report appendix:

- `Postnatal recovery and going home`
- `Wound and perineal care`
- `Feeding and breastfeeding`
- `Newborn care and safe sleep`
- `Warning signs and complications`

---

## Getting `body` right

This is the part that decides whether the system works, and the only part that is real work.

### Keep the headings

The chunker splits on headings, and a chunk that keeps `## When to seek help` attached to
its advice is the difference between safe and unsafe retrieval. **Markdown headings
(`##`) are ideal.** If the source has headings, preserve them.

### Strip page furniture

Navigation, cookie banners, "Skip to main content", social share buttons, related-links
sidebars, footers. The validator warns if a body *starts* with obvious chrome, but it
cannot catch everything.

For HTML, `trafilatura` handles this far better than BeautifulSoup:

```python
import trafilatura
html = trafilatura.fetch_url(url)
body = trafilatura.extract(html, include_comments=False, include_tables=True,
                           favor_precision=True, output_format="markdown")
```

For PDFs, `pdfplumber`:

```python
import pdfplumber
with pdfplumber.open(path) as pdf:
    body = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
```

### Check every single one by hand

Twenty-seven documents is small enough to read. **Do it.** A two-column PDF that extracts
as interleaved nonsense will silently poison retrieval, and "silently wrong" is exactly
the failure mode this project is meant to measure. Budget an afternoon.

Specific things to look for:

- Two-column PDFs interleaving into word salad
- Tables flattened into unreadable runs of numbers
- Bullet characters becoming `�` or dropping out
- Headings losing their `#` and merging into the paragraph below
- The same content appearing twice (print + screen versions)

### Cache the raw downloads

Put the original HTML/PDF in `data/raw/` (gitignored). You will re-run extraction several
times and there is no reason to re-hit those servers.

---

## Scope

The brief says keep the knowledge base manageable — a handful of documents, not
thousands. **The 27 sources in the report appendix are the scope.** Do not expand it.

A bigger corpus is not a better project here; it just makes out-of-KB questions harder to
write and the failure modes harder to see.

---

## URLs

Take them from `99_appendix.tex`, **not** from the compiled PDF. Several URLs are
line-wrapped in the PDF and at least two (RACGP, MAGICapp) are truncated mid-token.

---

## Checklist

- [ ] All 27 sources fetched, raw copies in `data/raw/`
- [ ] Every `body` read by a human
- [ ] Headings preserved
- [ ] `doc_id` unique, slug-like, and final
- [ ] `url` absolute and correct
- [ ] `metadata.category` on every document
- [ ] `metadata.retrieved` on every document
- [ ] `python scripts/validate_kb.py data/kb/matcare_docs.json` passes
- [ ] Committed to the repo
