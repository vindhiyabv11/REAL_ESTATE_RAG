"""Chroma create / load / reset helpers (persisted on disk)."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from rag.embeddings import get_embeddings

BATCH = 100


def _open() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def reset_collection() -> None:
    """Drop old articles so they cannot pollute new answers."""
    try:
        _open().delete_collection()
    except Exception:
        pass


def create_store(chunks: list[Document]) -> Chroma:
    reset_collection()
    store = _open()
    for i in range(0, len(chunks), BATCH):
        store.add_documents(chunks[i:i + BATCH])
    return store


def load_store() -> Chroma | None:
    """Return the persisted store if it already holds data, else None."""
    try:
        store = _open()
        if store.get(limit=1)["ids"]:
            return store
    except Exception:
        pass
    return None