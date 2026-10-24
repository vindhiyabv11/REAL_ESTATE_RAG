"""Orchestration: ingest(urls) builds the vector store, answer(question) queries it."""
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from langchain_core.documents import Document

import config
from rag import cleaner, loader, retriever, splitter, vectorstore
from rag.prompts import build_messages

Item = str| dict


@dataclass
class IngestReport:
    store: object = None
    articles: list[dict] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    n_chunks: int = 0


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    passages: list[Document]


def _normalize(items: list[Item]) -> list[dict]:
    out, seen = [], set()
    for it in items:
        d = {"url": it} if isinstance(it, str) else dict(it)
        d["url"] = d["url"].strip()
        if d["url"].startswith("http") and d["url"] not in seen:
            seen.add(d["url"])
            out.append(d)
    return out


def _load_and_clean(item: dict) -> Document:
    doc = loader.load_url(item["url"], fallback=item)
    doc.page_content = cleaner.clean_text(doc.page_content)
    if len(doc.page_content) < config.MIN_ARTICLE_CHARS:
        raise loader.LoaderError("too little text after cleaning (paywall, video page or blocked)")
    return doc


def ingest(items: list[Item], progress: Callable[[str], None] | None = None) -> IngestReport:
    """Fetch -> extract -> clean -> chunk -> embed -> store. Replaces the previous collection."""
    say = progress or (lambda _msg: None)
    items = _normalize(items)
    report = IngestReport()
    docs: list[Document] = []

    say(f"Fetching {len(items)} article(s)...")
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(_load_and_clean, it): it for it in items}
        for fut in as_completed(futures):
            url = futures[fut]["url"]
            try:
                doc = fut.result()
            except Exception as exc:
                report.failures.append((url, str(exc)))
                say(f"Skipped {url}: {exc}")
                continue
            docs.append(doc)
            say(f"Loaded: {doc.metadata['title'][:80]}")

    if not docs:
        return report

    docs.sort(key=lambda d: d.metadata.get("published", ""), reverse=True)
    chunks = splitter.split_documents(docs)
    say(f"Embedding {len(chunks)} chunks...")
    report.store = vectorstore.create_store(chunks)
    report.articles = [d.metadata for d in docs]
    report.n_chunks = len(chunks)
    return report


def load_existing_store():
    """Reuse the Chroma collection saved by a previous run, if any."""
    return vectorstore.load_store()


def _sources(passages: list[Document], answer: str) -> list[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    chosen = [p for i, p in enumerate(passages, start=1) if i in cited] or passages
    out, seen = [], set()
    for p in chosen:
        url = p.metadata.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append({k: p.metadata.get(k, "") for k in ("url", "title", "published", "source")})
    return out


def answer(store, question: str) -> AnswerResult:
    candidates = retriever.retrieve(store, question)
    if not candidates:
        return AnswerResult("No relevant passages were found. Process some URLs first.", [], [])
    passages = retriever.rerank(question, candidates) if config.USE_RERANK else candidates[:config.RERANK_TOP_N]

    from rag.llm import get_llm

    reply = get_llm().invoke(build_messages(question, passages))
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    return AnswerResult(answer=text.strip(), sources=_sources(passages, text), passages=passages)