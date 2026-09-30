"""MongoDB Local Atlas access.

Local Atlas runs as a Docker container bundling two processes: mongod (the
database) and mongot (the Atlas Search node). $vectorSearch is a mongot feature
— community MongoDB cannot do it, which is why a plain `mongo` container will
not work here.
"""

from __future__ import annotations

import time

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from .config import SETTINGS


def client() -> MongoClient:
    c = MongoClient(SETTINGS.mongodb_uri, serverSelectionTimeoutMS=8000)
    c.admin.command("ping")  # fail fast with a clear error
    return c


def collections(c: MongoClient | None = None) -> tuple[Collection, Collection]:
    c = c or client()
    db = c[SETTINGS.db_name]
    return db[SETTINGS.collection], db[SETTINGS.chat_collection]


def create_vector_index(coll: Collection, dimensions: int, wait: bool = True) -> None:
    """Create (or recreate) the vector search index.

    `filter` fields are declared so we can later restrict search by category —
    useful for the "warning signs" subset without re-indexing.
    """
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "doc_id"},
                {"type": "filter", "path": "source_name"},
            ]
        },
        name=SETTINGS.vector_index,
        type="vectorSearch",
    )

    if list(coll.list_search_indexes(name=SETTINGS.vector_index)):
        coll.drop_search_index(SETTINGS.vector_index)
        time.sleep(5)

    coll.create_search_index(model=model)
    if wait:
        wait_for_index(coll, SETTINGS.vector_index)


def wait_for_index(coll: Collection, name: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        idx = list(coll.list_search_indexes(name=name))
        status = idx[0].get("status", "PENDING") if idx else "PENDING"
        print(f"  index status: {status:<12}", end="\r", flush=True)
        if status == "READY":
            print(f"  index '{name}' ready        ")
            return
        time.sleep(5)
    raise TimeoutError(f"index '{name}' not READY within {timeout}s")
