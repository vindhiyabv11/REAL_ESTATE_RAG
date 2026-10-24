from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag.retriever import parse_scores, rerank


def test_parse_scores_handles_code_fences():
    raw = '```json\n{"0": 3, "2": 9.5}\n```'
    assert parse_scores(raw, 3) == [3.0, 0.0, 9.5]


def test_parse_scores_rejects_garbage():
    with pytest.raises(ValueError):
        parse_scores("no idea", 3)


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.content)


def _docs(n):
    return [Document(page_content=f"chunk {i}", metadata={"published": f"2026-10-{10 + i:02d}"}) for i in range(n)]


def test_rerank_orders_by_score_then_recency():
    docs = _docs(4)
    llm = FakeLLM('{"0": 9, "1": 5, "2": 9, "3": 1}')
    top = rerank("q", docs, top_n=3, llm=llm)
    assert [d.page_content for d in top] == ["chunk 2", "chunk 0", "chunk 1"]


def test_rerank_falls_back_to_original_order_on_bad_reply():
    docs = _docs(4)
    top = rerank("q", docs, top_n=2, llm=FakeLLM("sorry, cannot"))
    assert [d.page_content for d in top] == ["chunk 0", "chunk 1"]


def test_rerank_skips_llm_when_few_docs():
    docs = _docs(2)
    assert rerank("q", docs, top_n=5, llm=None) == docs