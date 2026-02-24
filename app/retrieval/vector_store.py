# Handles storing, retrieving, and managing embedded chunks in ChromaDB vector database

import chromadb
from chromadb.config import Settings
from typing import List, Dict
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RETRIEVAL


# Initialize persistent ChromaDB client once at module level
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection():
    """
    Gets existing collection or creates a new one if it doesn't exist.
    A collection is like a table in a traditional database.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for semantic search
    )
    return collection


def store_chunks(chunks: List[Dict]) -> None:
    """
    Stores embedded chunks into ChromaDB.
    Each chunk needs a unique ID, its embedding, text, and metadata.
    """
    collection = get_or_create_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # unique ID per chunk using source filename + page + chunk index
        chunk_id = f"{chunk['metadata']['source']}_p{chunk['metadata']['page']}_c{chunk['metadata']['chunk_index']}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    # ChromaDB accepts batches — upsert means insert or update if ID exists
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Stored {len(ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'")


def retrieve_chunks(query_embedding: List[float], top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """
    Finds the most semantically similar chunks to a query embedding.
    Returns top_k chunks with their text, metadata, and similarity distance.
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # reformat ChromaDB's response into a clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]  # lower = more similar in cosine space
        })

    print(f"✅ Retrieved {len(chunks)} chunks for query")
    return chunks


def retrieve_chunks_by_source(query_embedding: List[float], source: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """
    Retrieves chunks filtered by a specific source document.
    Useful when user wants to search within a single uploaded PDF only.
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"source": source},   # metadata filter
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    print(f"✅ Retrieved {len(chunks)} chunks from source '{source}'")
    return chunks


def delete_source(source: str) -> None:
    """
    Deletes all chunks belonging to a specific PDF from the collection.
    Called when user removes a document from the app.
    """
    collection = get_or_create_collection()
    collection.delete(where={"source": source})
    print(f"✅ Deleted all chunks for source '{source}'")


def get_all_sources() -> List[str]:
    """
    Returns a list of all unique document names currently stored in ChromaDB.
    Used to display uploaded documents in the sidebar.
    """
    collection = get_or_create_collection()
    results = collection.get(include=["metadatas"])

    sources = list(set(
        meta["source"] for meta in results["metadatas"]
    ))
    return sources


def get_collection_count() -> int:
    """
    Returns total number of chunks stored in the collection.
    Useful for debugging and displaying stats in the UI.
    """
    collection = get_or_create_collection()
    return collection.count()



# if __name__ == "__main__":
#     from app.ingestion.pdf_loader import load_pdf
#     from app.ingestion.chunker import chunk_pages
#     from app.ingestion.embedder import embed_chunks, embed_query

#     # full pipeline test
#     pages = load_pdf("data/uploads/test.pdf")
#     chunks = chunk_pages(pages)
#     embedded_chunks = embed_chunks(chunks)
#     store_chunks(embedded_chunks)

#     print("\nTotal chunks in DB:", get_collection_count())
#     print("Sources in DB:", get_all_sources())

#     # test retrieval
#     query = "What is RAG and how does it work?"
#     query_vec = embed_query(query)
#     results = retrieve_chunks(query_vec, top_k=3)

#     print(f"\n--- Top 3 results for: '{query}' ---")
#     for r in results:
#         print(f"\nSource: {r['metadata']['source']} | Page: {r['metadata']['page']} | Distance: {r['distance']:.4f}")
#         print(r["text"][:200], "...")