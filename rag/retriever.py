"""Retrieval: MMR search in Chroma, then an LLM reranker that keeps the best few chunks."""
import json
import re

from langchain_core.documents import Document

import config
from rag.prompts import RERANK_PROMPT


def retrieve(store, question: str) -> list[Document]:
    return store.max_marginal_relevance_search(
        question,
        k=config.RETRIEVE_K,
        fetch_k=max(config.RETRIEVE_FETCH_K, config.RETRIEVE_K),
        lambda_mult=0.7,
    )


def parse_scores(raw: str, n: int) -> list[float]:
    """Parse '{"0": 8, "1": 2}' (possibly wrapped in ```json fences) into a list of n scores."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reranker reply")
    data = json.loads(match.group(0))
    scores = [0.0] * n
    for key, val in data.items():
        idx = int(re.sub(r"\D", "", str(key)) or -1)
        if 0 <= idx < n:
            scores[idx] = float(val)
    return scores


def rerank(question: str, docs: list[Document], top_n: int | None = None, llm=None) -> list[Document]:
    """Keep the top_n most useful chunks. Ties go to the more recently published article.
    If the reranker fails for any reason, fall back to the original MMR order."""
    top_n = top_n or config.RERANK_TOP_N
    if len(docs) <= top_n:
        return docs
    if llm is None:
        from rag.llm import get_llm

        llm = get_llm(config.RERANK_MODEL)

    passages = "\n\n".join(f"[{i}] {d.page_content[:700]}" for i, d in enumerate(docs))
    try:
        reply = llm.invoke(RERANK_PROMPT.format(question=question, passages=passages))
        scores = parse_scores(reply.content, len(docs))
    except Exception:
        return docs[:top_n]

    order = list(range(len(docs)))
    order.sort(key=lambda i: docs[i].metadata.get("published", ""), reverse=True)
    order.sort(key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:top_n]]