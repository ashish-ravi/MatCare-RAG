"""Local embeddings via voyage-4-nano.

The model exposes separate encode_query / encode_document methods. They are not
interchangeable: documents and queries are encoded into the same space but with
different prefixes, and mixing them up quietly degrades retrieval without ever
raising an error. That is why `input_type` is required rather than defaulted.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from .config import SETTINGS

InputType = Literal["document", "query"]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    # trust_remote_code is needed for the custom encode_query/encode_document API.
    return SentenceTransformer(SETTINGS.embedding_model, trust_remote_code=True)


def embedding_dimensions() -> int:
    return _model().encode_query(["dimension probe"]).shape[1]


def embed(
    texts: list[str],
    input_type: InputType,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    m = _model()
    fn = m.encode_query if input_type == "query" else m.encode_document
    return fn(texts, show_progress_bar=show_progress_bar).tolist()
