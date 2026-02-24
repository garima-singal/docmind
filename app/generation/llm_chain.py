# Connects the LLM, prompt templates, and retrieval pipeline into a single callable RAG chain

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Tuple

from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE
from app.generation.prompt_templates import (
    QA_PROMPT,
    CONVERSATIONAL_PROMPT,
    format_context,
    format_chat_history
)
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import retrieve_chunks
from app.retrieval.reranker import rerank_chunks


# Initialize LLM once at module level
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMPERATURE
)


def answer_question(query: str) -> Dict:
    """
    Single-turn QA — takes a question, retrieves context, returns answer with sources.
    Returns a dict with answer text and the source chunks used.
    """
    # step 1 — embed the query
    query_embedding = embed_query(query)

    # step 2 — retrieve top 10 similar chunks from ChromaDB
    retrieved_chunks = retrieve_chunks(query_embedding, top_k=10)

    if not retrieved_chunks:
        return {
            "answer": "No relevant documents found. Please upload a PDF first.",
            "sources": []
        }

    # step 3 — rerank to top 3
    top_chunks = rerank_chunks(query, retrieved_chunks, top_k=3)

    # step 4 — format chunks into context string
    context = format_context(top_chunks)

    # step 5 — build chain and invoke
    chain = QA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": query
    })

    # step 6 — return answer + sources for UI display
    sources = [
        {
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "text": chunk["text"],
            "score": chunk.get("rerank_score", 0)
        }
        for chunk in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


def answer_with_history(query: str, chat_history: List[Dict]) -> Dict:
    """
    Multi-turn conversational QA — same as answer_question but
    passes full chat history so the LLM understands follow-up questions.
    chat_history format: [{"role": "user/assistant", "content": "..."}]
    """
    # step 1 — embed query
    query_embedding = embed_query(query)

    # step 2 — retrieve and rerank
    retrieved_chunks = retrieve_chunks(query_embedding, top_k=10)

    if not retrieved_chunks:
        return {
            "answer": "No relevant documents found. Please upload a PDF first.",
            "sources": []
        }

    top_chunks = rerank_chunks(query, retrieved_chunks, top_k=3)
    context = format_context(top_chunks)

    # step 3 — format history into LangChain message objects
    formatted_history = format_chat_history(chat_history)

    # step 4 — build conversational chain and invoke
    chain = CONVERSATIONAL_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": query,
        "chat_history": formatted_history
    })

    sources = [
        {
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "text": chunk["text"],
            "score": chunk.get("rerank_score", 0)
        }
        for chunk in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


def stream_answer(query: str, chat_history: List[Dict]):
    """
    Streaming version of answer_with_history — yields answer tokens one by one.
    Used in Streamlit to display the answer as it is being generated
    rather than waiting for the full response.
    """
    query_embedding = embed_query(query)
    retrieved_chunks = retrieve_chunks(query_embedding, top_k=10)

    if not retrieved_chunks:
        yield "No relevant documents found. Please upload a PDF first."
        return

    top_chunks = rerank_chunks(query, retrieved_chunks, top_k=3)
    context = format_context(top_chunks)
    formatted_history = format_chat_history(chat_history)

    chain = CONVERSATIONAL_PROMPT | llm | StrOutputParser()

    # stream tokens as they arrive from the LLM
    for token in chain.stream({
        "context": context,
        "question": query,
        "chat_history": formatted_history
    }):
        yield token



if __name__ == "__main__":
    # make sure you have already stored chunks in ChromaDB
    # if not, run vector_store.py test first

    print("--- Single Turn QA ---")
    result = answer_question("What is RAG and how does it work?")
    print("\nAnswer:", result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} | Page {s['page']} | Score: {s['score']:.4f}")

    print("\n--- Conversational QA ---")
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation..."}
    ]
    result2 = answer_with_history("Can you explain chunking in more detail?", history)
    print("\nAnswer:", result2["answer"])