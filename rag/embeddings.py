"""OpenAI embeddings. Never mix embedding models inside one Chroma collection."""
from langchain_openai import OpenAIEmbeddings

import config


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)