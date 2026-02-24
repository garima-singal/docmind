# Converts text chunks into numerical vector embeddings using HuggingFace sentence-transformers

from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import EMBEDDING_MODEL


# Load model once at module level so it isn't reloaded on every function call
model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    
    """
    Takes chunks from chunker.py and adds an embedding vector to each one.
    Returns the same list of chunks with an 'embedding' key added.
    """
    if not chunks:
        print("⚠️ No chunks to embed — PDF may be image-based or empty.")
        return []
    
    texts = [chunk["text"] for chunk in chunks]

    print(f"⏳ Embedding {len(texts)} chunks... this may take a moment on first run.")
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # convert numpy array to list for storage

    print(f"✅ Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    return chunks


def embed_query(query: str) -> List[float]:
    """
    Embeds a single user query string for similarity search at retrieval time.
    Must use the same model used during indexing.
    """
    embedding = model.encode(query)
    return embedding.tolist()


# if __name__ == "__main__":
#     from app.ingestion.pdf_loader import load_pdf
#     from app.ingestion.chunker import chunk_pages

#     pages = load_pdf("data/uploads/test.pdf")
#     chunks = chunk_pages(pages)
#     embedded_chunks = embed_chunks(chunks)

#     # inspect first chunk
#     print("\n--- First Chunk ---")
#     print("Text:", embedded_chunks[0]["text"][:100], "...")
#     print("Embedding dimension:", len(embedded_chunks[0]["embedding"]))
#     print("First 5 values:", embedded_chunks[0]["embedding"][:5])

#     # test query embedding
#     query_vec = embed_query("What is RAG?")
#     print("\n--- Query Embedding ---")
#     print("Dimension:", len(query_vec))
#     print("First 5 values:", query_vec[:5])