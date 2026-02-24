import os
from dotenv import load_dotenv

load_dotenv()

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
COLLECTION_NAME = "documents"
