"""Command-line entry points.

    matcare check          verify the local stack is up
    matcare ingest         build the knowledge base into MongoDB
    matcare ask "..."      one question, with citations
    matcare eval           run the evaluation harness
"""

from __future__ import annotations

import argparse
import sys


def cmd_check(args) -> int:
    from .config import SETTINGS
    ok = True

    print("MongoDB Local Atlas")
    try:
        from .db import collections
        coll, _ = collections()
        print(f"  connected    {SETTINGS.mongodb_uri}")
        print(f"  chunks       {coll.count_documents({})}")
        idx = list(coll.list_search_indexes(name=SETTINGS.vector_index))
        print(f"  vector index {idx[0].get('status') if idx else 'MISSING'}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  → is Docker Desktop running?   atlas local start matcare")
        ok = False

    print("\nOllama")
    try:
        from openai import OpenAI
        c = OpenAI(base_url=f"{SETTINGS.ollama_base_url}/v1", api_key="ollama")
        available = [m.id for m in c.models.list().data]
        print(f"  models       {available}")
        if SETTINGS.ollama_model not in available:
            print(f"  MISSING '{SETTINGS.ollama_model}'  →  ollama pull {SETTINGS.ollama_model}")
            ok = False
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  → ollama serve")
        ok = False

    print("\nEmbedding model")
    try:
        from .embeddings import embedding_dimensions
        print(f"  {SETTINGS.embedding_model}  dim={embedding_dimensions()}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        ok = False

    print("\n" + ("all checks passed" if ok else "some checks FAILED — see above"))
    return 0 if ok else 1


def cmd_ingest(args) -> int:
    from .ingest import ingest
    n = ingest(rebuild_index=not args.no_index)
    print(f"\ningested {n} chunks")
    return 0


def cmd_ask(args) -> int:
    from .pipeline import MatCarePipeline
    p = MatCarePipeline(session_id=args.session)
    a = p.ask(args.question, k=args.k)
    print(f"\n{a.text}\n")
    if a.citations:
        print("Sources:")
        for c in a.citations:
            print(f"  - {c['source']} — {c['title']}\n    {c['url']}")
    flags = [f for f, v in (("REFUSED", a.refused), ("ESCALATED", a.escalated)) if v]
    if flags:
        print(f"\n[{', '.join(flags)}]")
    return 0


def cmd_eval(args) -> int:
    from .db import collections
    from .evaluation.evaluate import evaluate_retrieval, retrieve_all
    from .evaluation.testset import load_test_collection
    from .retrieval import BM25Retriever, DenseRetriever

    tc = load_test_collection(strict=not args.lenient)
    print(tc.summary() + "\n")

    coll, _ = collections()
    retrievers = []
    if args.system in ("dense", "both"):
        retrievers.append(DenseRetriever(coll))
    if args.system in ("bm25", "both"):
        from .ingest import load_chunks_from_db
        retrievers.append(BM25Retriever(load_chunks_from_db()))

    for r in retrievers:
        results = retrieve_all(r, tc, k=args.depth)
        res = evaluate_retrieval(results, tc, tag=r.name, granularity=args.granularity)
        print(res.report())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="matcare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify the local stack").set_defaults(fn=cmd_check)

    p = sub.add_parser("ingest", help="build the knowledge base into MongoDB")
    p.add_argument("--no-index", action="store_true", help="skip vector index rebuild")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None, help="passages to retrieve")
    p.add_argument("--session", default=None, help="session id to enable chat memory")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation harness")
    p.add_argument("--system", choices=["dense", "bm25", "both"], default="both")
    p.add_argument("--granularity", choices=["document", "chunk"], default="document")
    p.add_argument("--depth", type=int, default=10, help="retrieval depth")
    p.add_argument("--lenient", action="store_true",
                   help="warn instead of failing on test-collection inconsistencies")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
