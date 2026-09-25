#!/usr/bin/env python3
"""Validate a knowledge-base JSON file against the MatCare schema.

Run this BEFORE committing data/kb/matcare_docs.json. It is the contract
between the knowledge-base team and the pipeline: if this passes, ingestion
will work.

    python scripts/validate_kb.py data/kb/matcare_docs.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = ["doc_id", "title", "url", "body", "sourceName"]
OPTIONAL = ["updated", "format", "metadata"]

MIN_BODY_WORDS = 40


def validate(path: Path) -> int:
    try:
        docs = json.loads(path.read_text())
    except Exception as e:
        print(f"FAIL: cannot parse {path}: {e}")
        return 1

    if not isinstance(docs, list):
        print("FAIL: top level must be a JSON array of documents")
        return 1

    errors, warnings = [], []
    seen_ids, seen_urls = set(), set()

    for i, d in enumerate(docs):
        where = f"[{i}] {d.get('doc_id') or d.get('title') or '<no id>'}"

        for key in REQUIRED:
            if not d.get(key):
                errors.append(f"{where}: missing required field '{key}'")

        did = d.get("doc_id")
        if did:
            if did in seen_ids:
                errors.append(f"{where}: duplicate doc_id '{did}'")
            seen_ids.add(did)
            if not did.replace("-", "").replace("_", "").isalnum():
                warnings.append(f"{where}: doc_id should be slug-like (a-z, 0-9, -, _)")

        url = d.get("url", "")
        if url:
            if not url.startswith("http"):
                errors.append(f"{where}: url must be absolute")
            if url in seen_urls:
                warnings.append(f"{where}: duplicate url")
            seen_urls.add(url)

        body = d.get("body") or ""
        words = len(body.split())
        if words < MIN_BODY_WORDS:
            errors.append(f"{where}: body only {words} words — extraction probably failed")
        if "cookie" in body[:400].lower() or "skip to main content" in body[:400].lower():
            warnings.append(f"{where}: body starts with page chrome — check extraction")
        if body.count("�"):
            warnings.append(f"{where}: contains replacement characters (encoding issue)")

        meta = d.get("metadata") or {}
        if not meta.get("category"):
            warnings.append(f"{where}: metadata.category missing (used for reporting)")
        if not meta.get("retrieved"):
            warnings.append(f"{where}: metadata.retrieved missing (provenance date)")

    print(f"{path}: {len(docs)} documents")
    if docs:
        wl = [len((d.get('body') or '').split()) for d in docs]
        print(f"  body words: min {min(wl)} / median {sorted(wl)[len(wl)//2]} / max {max(wl)}")
        cats = Counter((d.get("metadata") or {}).get("category", "<none>") for d in docs)
        print("  categories: " + ", ".join(f"{k} ({v})" for k, v in cats.most_common()))
        srcs = Counter(d.get("sourceName", "<none>") for d in docs)
        print("  publishers: " + ", ".join(f"{k} ({v})" for k, v in srcs.most_common()))

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")

    print()
    if errors:
        print(f"FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"PASSED — {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/kb/matcare_docs.json")
    sys.exit(validate(target))
