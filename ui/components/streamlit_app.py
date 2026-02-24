# Main Streamlit entry point — initializes session state and renders the full chat UI with sidebar

import sys
import os

# go up two levels: ui/components/ → ui/ → project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import store_chunks, get_all_sources, delete_source, get_collection_count
from app.generation.llm_chain import stream_answer

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Session state initialization — persists data across reruns
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_sources" not in st.session_state:
    st.session_state.uploaded_sources = get_all_sources()

if "processing" not in st.session_state:
    st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar — document upload and management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📂 Documents")
    st.caption("Upload PDFs to chat with them")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=st.session_state.processing
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.uploaded_sources:
                st.info(f"'{uploaded_file.name}' already loaded.")
                continue

            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                # ensure uploads directory exists
                os.makedirs("data/uploads", exist_ok=True)

                # save uploaded file temporarily to disk
                temp_path = os.path.join(ROOT_DIR, "data", "uploads", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # run full ingestion pipeline
                pages = load_pdf(temp_path)
                chunks = chunk_pages(pages)
                embedded = embed_chunks(chunks)
                store_chunks(embedded)

                st.session_state.uploaded_sources.append(uploaded_file.name)

            st.success(f"✅ '{uploaded_file.name}' ready!")

    st.divider()

    # show loaded documents
    st.subheader("📚 Loaded Documents")
    if st.session_state.uploaded_sources:
        for source in st.session_state.uploaded_sources:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {source}")
            if col2.button("🗑️", key=f"del_{source}", help=f"Remove {source}"):
                delete_source(source)
                st.session_state.uploaded_sources.remove(source)
                st.rerun()
    else:
        st.caption("No documents loaded yet.")

    st.divider()

    # stats
    st.caption(f"Total chunks in DB: {get_collection_count()}")

    # clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("📄 RAG Document QA")
st.caption("Ask questions about your uploaded documents")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # show sources for assistant messages if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** — Page {src['page']} | Score: {src['score']:.4f}")
                    st.caption(src["text"][:300] + "...")
                    st.divider()

# chat input — disabled if no documents loaded or currently processing
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.processing or not st.session_state.uploaded_sources
):
    # add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate and stream assistant response
    with st.chat_message("assistant"):
        st.session_state.processing = True

        response_placeholder = st.empty()
        full_response = ""

        # stream tokens into placeholder with blinking cursor effect
        for token in stream_answer(prompt, st.session_state.chat_history[:-1]):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.processing = False

    # save assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response,
    })

    st.rerun()