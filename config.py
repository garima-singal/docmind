# Central configuration — loads from .env locally and Streamlit secrets in production

import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str) -> str:
    """
    Gets config value from Streamlit secrets (production) or .env (local).
    """
    try:
        import streamlit as st
        return st.secrets.get(key) or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")

# LLM
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3

# ChromaDB
CHROMA_PERSIST_DIR = get_secret("CHROMA_PERSIST_DIR") or "data/chroma_db"
COLLECTION_NAME = "documents"