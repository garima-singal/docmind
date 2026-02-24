# 📄 RAG Document QA System

A full end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with your PDF documents. Upload any PDF and ask questions — answers are grounded strictly in your documents, with source citations and page references.

Built with Python, LangChain, OpenAI, ChromaDB, and Streamlit.

---

## 🖥️ Demo

![RAG Document QA Demo](assets/demo.png)

> Upload a PDF → Ask a question → Get a cited, streamed answer

---

## ✨ Features

- 📥 **PDF Ingestion** — Extracts text from PDFs using PyMuPDF, with automatic OCR fallback for scanned documents
- ✂️ **Smart Chunking** — Splits documents into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- 🔢 **Semantic Embeddings** — Converts chunks into vector embeddings using HuggingFace `sentence-transformers`
- 🗄️ **Vector Store** — Persists embeddings in ChromaDB with cosine similarity search
- 🎯 **Two-Stage Retrieval** — Vector search (top-10) followed by CrossEncoder reranking (top-3) for higher relevance
- 🤖 **LLM Generation** — Uses OpenAI `gpt-4o-mini` via LangChain, strictly grounded in retrieved context
- 💬 **Multi-turn Chat** — Full conversation history support with follow-up question handling
- ⚡ **Streaming Responses** — Tokens stream in real-time with a typewriter effect in the UI
- 📎 **Source Citations** — Every answer includes the source document, page number, and rerank score
- 📊 **RAGAS Evaluation** — Built-in evaluation pipeline measuring Faithfulness, Context Precision, and Context Recall
- 🗂️ **Multi-document Support** — Upload and manage multiple PDFs simultaneously

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│              INGESTION                  │
│  pdf_loader → chunker → embedder        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │   ChromaDB    │  ← Persistent Vector Store
          └───────┬───────┘
                  │
    User Query    │
        │         ▼
        │   ┌─────────────────────────────────────┐
        │   │           RETRIEVAL                 │
        │   │  vector_store (top-10) →            │
        │   │  reranker / CrossEncoder (top-3)    │
        │   └─────────────────┬───────────────────┘
        │                     │
        ▼                     ▼
┌───────────────────────────────────────────┐
│              GENERATION                   │
│  prompt_templates + LangChain + OpenAI    │
│  → Streamed Answer + Source Citations     │
└───────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── app/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PDF text extraction + OCR fallback
│   │   ├── chunker.py          # Text splitting strategies
│   │   └── embedder.py         # HuggingFace sentence embeddings
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB operations (store, retrieve, delete)
│   │   └── reranker.py         # CrossEncoder reranking
│   ├── generation/
│   │   ├── prompt_templates.py # System prompts + context formatting
│   │   └── llm_chain.py        # Full RAG chain (QA, chat, streaming)
│   └── evaluation/
│       └── evaluate.py         # RAGAS evaluation pipeline
├── ui/
│   └── components/
│       └── streamlit_app.py    # Streamlit chat interface
├── data/
│   ├── uploads/                # Uploaded PDFs (gitignored)
│   └── chroma_db/              # Persisted vector index (gitignored)
├── config.py                   # Centralised configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key
- (Optional) Tesseract OCR — only needed for scanned/image-based PDFs

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```
Then open `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-openai-api-key-here
CHROMA_PERSIST_DIR=data/chroma_db
```

**5. Run the app**
```bash
streamlit run ui/components/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## ⚙️ Configuration

All settings are managed in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model used for generation |
| `TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Chunks retrieved from vector store |
| `TOP_K_RERANK` | `3` | Chunks kept after reranking |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |

---

## 📊 Evaluation

The project includes a RAGAS evaluation pipeline to measure RAG quality:

```bash
python app/evaluation/evaluate.py
```

This runs 10 test questions and reports:

| Metric | Description |
|---|---|
| **Faithfulness** | Are answers grounded in retrieved context? |
| **Context Precision** | Are retrieved chunks relevant to the query? |
| **Context Recall** | Were all relevant chunks retrieved? |

Results are saved to `app/evaluation/evaluation_results.csv`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4o-mini |
| **Orchestration** | LangChain |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector DB** | ChromaDB (persistent) |
| **PDF Parsing** | PyMuPDF + Tesseract OCR |
| **UI** | Streamlit |
| **Evaluation** | RAGAS |

---

## 🔒 Environment Variables

Create a `.env` file based on `.env.example`:

```env
OPENAI_API_KEY=your-openai-api-key-here
CHROMA_PERSIST_DIR=data/chroma_db
```

**Never commit your `.env` file.** It is listed in `.gitignore`.

---

## 📝 License

This project is licensed under the MIT License.