# Takes top-k retrieved chunks and reranks them using a CrossEncoder for more accurate relevance scoring

from sentence_transformers import CrossEncoder
from typing import List, Dict
from config import TOP_K_RERANK

# Load CrossEncoder model once at module level
# This model jointly processes query + document together for more accurate scoring
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = TOP_K_RERANK) -> List[Dict]:
    """
    Reranks retrieved chunks by scoring each one against the query.
    Returns top_k most relevant chunks in order of relevance.
    """
    if not chunks:
        return []

    # create (query, chunk_text) pairs for the CrossEncoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # score each pair — higher score = more relevant
    scores = rerank_model.predict(pairs)

    # attach scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # sort by score descending and return top_k
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = reranked[:top_k]

    print(f"✅ Reranked {len(chunks)} chunks → kept top {len(top_chunks)}")
    return top_chunks


def rerank_with_scores(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Same as rerank_chunks but returns ALL chunks with scores attached,
    not just top_k. Useful for analysis and notebook comparisons.
    """
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = rerank_model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

# if __name__ == "__main__":
#     from app.ingestion.pdf_loader import load_pdf
#     from app.ingestion.chunker import chunk_pages
#     from app.ingestion.embedder import embed_chunks, embed_query
#     from app.retrieval.vector_store import store_chunks, retrieve_chunks

#     query = "How does reranking improve RAG systems?"

#     # retrieve top 10 first
#     query_vec = embed_query(query)
#     retrieved = retrieve_chunks(query_vec, top_k=10)

#     print(f"\n--- Before Reranking (by vector similarity) ---")
#     for i, r in enumerate(retrieved):
#         print(f"{i+1}. Page {r['metadata']['page']} | Distance: {r['distance']:.4f} | {r['text'][:80]}...")

#     # now rerank
#     reranked = rerank_chunks(query, retrieved, top_k=3)

#     print(f"\n--- After Reranking (top 3 by CrossEncoder score) ---")
#     for i, r in enumerate(reranked):
#         print(f"{i+1}. Page {r['metadata']['page']} | Rerank Score: {r['rerank_score']:.4f} | {r['text'][:80]}...")