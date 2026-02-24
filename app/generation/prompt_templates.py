# Defines all prompt templates used to instruct the LLM on how to answer using retrieved context

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# System prompt — defines the LLM's behavior and strict grounding rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

Follow these rules without exception:
1. Only use information from the provided context to answer the question.
2. If the answer is not found in the context, say "I don't have enough information in the provided documents to answer this question." Do not guess or make up an answer.
3. Always cite your source by mentioning the document name and page number at the end of your answer.
4. Keep your answers clear, concise, and well structured.
5. If the question is a follow-up, use the chat history to understand what the user is referring to.

Context:
{context}
"""


# ---------------------------------------------------------------------------
# Basic QA prompt — used for single-turn question answering
# ---------------------------------------------------------------------------

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# ---------------------------------------------------------------------------
# Conversational prompt — used for multi-turn chat with history
# ---------------------------------------------------------------------------

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),   # injects full conversation history
    ("human", "{question}")
])


# ---------------------------------------------------------------------------
# Utility — formats retrieved chunks into a clean context string for the prompt
# ---------------------------------------------------------------------------

def format_context(chunks: list) -> str:
    """
    Converts a list of reranked chunks into a formatted context string.
    Each chunk is labeled with its source and page number for citation.
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "?")

        context_parts.append(
            f"[Source {i+1}: {source} | Page {page}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Utility — formats chat history into LangChain message objects
# ---------------------------------------------------------------------------

def format_chat_history(history: list) -> list:
    """
    Converts plain chat history list into LangChain message objects.
    history format: [{"role": "user/assistant", "content": "..."}]
    """
    messages = []
    for message in history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(SystemMessage(content=message["content"]))
    return messages




# if __name__ == "__main__":
#     # simulate what formatted context looks like
#     sample_chunks = [
#         {
#             "text": "RAG stands for Retrieval Augmented Generation. It combines LLMs with external knowledge retrieval.",
#             "metadata": {"source": "test_document.pdf", "page": 3},
#             "rerank_score": 0.95
#         },
#         {
#             "text": "Vector databases store embeddings and support semantic similarity search using cosine distance.",
#             "metadata": {"source": "test_document.pdf", "page": 3},
#             "rerank_score": 0.87
#         }
#     ]

#     context = format_context(sample_chunks)
#     print("--- Formatted Context ---")
#     print(context)

#     print("\n--- QA Prompt Messages ---")
#     messages = QA_PROMPT.format_messages(
#         context=context,
#         question="What is RAG?"
#     )
#     for msg in messages:
#         print(f"\n[{msg.type.upper()}]")
#         print(msg.content)