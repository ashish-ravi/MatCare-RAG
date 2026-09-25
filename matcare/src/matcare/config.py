"""Central configuration. Everything environment-dependent lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT     = Path(__file__).resolve().parents[2]
DATA     = ROOT / "data"
KB_DIR   = DATA / "kb"
EVAL_DIR = DATA / "eval"
RAW_DIR  = DATA / "raw"
RUNS_DIR = ROOT / "runs"

KB_FILE        = KB_DIR / "matcare_docs.json"
QUESTIONS_FILE = EVAL_DIR / "questions.csv"
QRELS_FILE     = EVAL_DIR / "qrels.txt"


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    db_name: str
    collection: str
    chat_collection: str
    vector_index: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str

    # Chunking. Section-aware first, character split only for oversized sections.
    chunk_size: int
    chunk_overlap: int

    # Retrieval / generation defaults.
    top_k: int
    temperature: float
    seed: int


def load_settings() -> Settings:
    return Settings(
        mongodb_uri     = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017/?directConnection=true"),
        db_name         = os.environ.get("MONGODB_DB", "matcare"),
        collection      = "knowledge_base",
        chat_collection = "chat_history",
        vector_index    = "vector_index",
        embedding_model = os.environ.get("EMBEDDING_MODEL_ID", "voyageai/voyage-4-nano"),
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model    = os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
        chunk_size      = int(os.environ.get("CHUNK_SIZE", 800)),
        chunk_overlap   = int(os.environ.get("CHUNK_OVERLAP", 80)),
        top_k           = int(os.environ.get("TOP_K", 5)),
        temperature     = float(os.environ.get("TEMPERATURE", 0.0)),
        seed            = int(os.environ.get("SEED", 42)),
    )


SETTINGS = load_settings()
