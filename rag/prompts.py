"""Prompt templates."""
from datetime import date

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a real estate research assistant. Answer the user's question \
using ONLY the numbered context passages below. Each passage starts with a header: \
[article title | site | publication date].

Rules:
1. Use only facts found in the passages. If the answer is not there, say \
"The provided articles don't contain this information." and briefly mention what related \
information they do contain.
2. Be specific: give exact numbers, rates, percentages, dates and names as written.
3. Passages come from articles published on different days. If figures conflict, prefer \
the most recent article and say which article/date the figure comes from.
4. Cite sources inline like [1], [2] right after the claims they support.
5. Keep it concise: 2-6 sentences or a short bullet list. Do not start with "FINAL ANSWER".
6. Do not add background knowledge, explanations or context that is not written in the passages.
7. Answer only what was asked. If the passages state a specific figure, name or date, lead with it in a single direct sentence.
Today's date is {today}."""

RERANK_PROMPT = """You rank passages by how useful they are for answering a question.

Question: {question}

Passages:
{passages}

Score EVERY passage from 0 (irrelevant) to 10 (directly answers the question, with the \
specific facts or numbers). Reply with ONLY a JSON object mapping passage number to score, \
for example {{"0": 8, "1": 2, "2": 10}}."""


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))


def build_messages(question: str, docs: list[Document]) -> list[tuple[str, str]]:
    system = SYSTEM_PROMPT.format(today=date.today().strftime("%B %d, %Y"))
    user = f"Context passages:\n\n{format_context(docs)}\n\nQuestion: {question}"
    return [("system", system), ("human", user)]