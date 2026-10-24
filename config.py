"""Central Configuration. Everything tunable lives here or in the .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def _int(name: str, default: int) -> int:
    try:
        return(int(os.getenv(name, default)))
    except ValueError:
        return default

def api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()

#--------------Models-----------------
LLM_MODEL = os.getenv("LLM_MODEL","gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL",LLM_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
# Leave LLM_TEMPERATURE empty in .env for models that only accept the default.
_temp = os.getenv("LLM_TEMPERATURE","0").strip()
LLM_TEMPERATURE = float(_temp) if _temp else None

#-------------Chunking----------------------------
CHUNK_SIZE = _int("CHUNK_SIZE",900)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP",120)

#-------------Retrieval---------------------------
RETRIEVE_K = _int("RETRIEVE_K", 15)              # chunks pulled from Chroma (MMR)
RETRIEVE_FETCH_K = _int("RETRIEVE_FETCH_K", 40)  # candidate pool for MMR
RERANK_TOP_N = _int("RERANK_TOP_N", 5)           # chunks sent to the LLM
USE_RERANK = os.getenv("USE_RERANK","true").lower() in {"1","true","yes"}

#-------------Ingestion----------------------------
MIN_ARTICLE_CHARS = _int("MIN_ARTICLE_CHARS",300)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT",15)
MAX_WORKERS = _int("MAX_WORKERS",5)

#--------------Storage-----------------------------
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "real_estate_news"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)





