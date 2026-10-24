from langchain_core.documents import Document

from rag.splitter import split_documents


def _doc(n_paragraphs=12):
    text = "\n\n".join(f"Paragraph {i}: mortgage rates and home prices moved again this week. " * 3 for i in range(n_paragraphs))
    return Document(
        page_content=text,
        metadata={"url": "https://example.com/a", "title": "Rates rise", "source": "example.com", "published": "2026-10-18"},
    )


def test_chunks_are_bounded_and_numbered():
    chunks = split_documents([_doc()], chunk_size=400, chunk_overlap=50)
    assert len(chunks) > 1
    header_len = len("[Rates rise | example.com | 2026-10-18]\n")
    assert all(len(c.page_content) <= 400 + header_len for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))


def test_every_chunk_has_header_and_metadata():
    chunks = split_documents([_doc()], chunk_size=400, chunk_overlap=50)
    for c in chunks:
        assert c.page_content.startswith("[Rates rise | example.com | 2026-10-18]")
        assert c.metadata["url"] == "https://example.com/a"
        assert c.metadata["published"] == "2026-10-18"


def test_short_document_is_one_chunk():
    d = Document(page_content="Short article text.", metadata={"title": "T"})
    chunks = split_documents([d])
    assert len(chunks) == 1