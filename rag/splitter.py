"""Chunking with RecursiveCharacterTextSplitter + a contextual header on each chunk."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _header(meta: dict) -> str:
    parts = [meta.get("title", ""), meta.get("source", ""), meta.get("published", "")]
    return " | ".join(p for p in parts if p)


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        header = _header(doc.metadata)
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            content = f"[{header}]\n{piece}" if header else piece
            chunks.append(Document(page_content=content, metadata={**doc.metadata, "chunk": i}))
    return chunks