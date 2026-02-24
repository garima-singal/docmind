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
```

---

## **Important note about ChromaDB on Streamlit Cloud**

Streamlit Cloud has an **ephemeral filesystem** — meaning `data/chroma_db/` gets wiped every time the app restarts. This means users will need to re-upload PDFs after each restart. This is fine for a portfolio project — just mention it in your README:

Add this line under the Demo section in `README.md`:
```
> Note: Uploaded documents persist during the session. Re-upload after app restarts.