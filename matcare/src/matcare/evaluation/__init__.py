"""Evaluation harness.

The pipeline is necessary but not sufficient — this package is where the
project's claims are actually substantiated.
"""

from .metrics import ndcg, ndcg_at_k
from .testset import TestCollection, load_test_collection
from .runs import write_run, load_run, rollup_to_documents

__all__ = [
    "ndcg", "ndcg_at_k",
    "TestCollection", "load_test_collection",
    "write_run", "load_run", "rollup_to_documents",
]
