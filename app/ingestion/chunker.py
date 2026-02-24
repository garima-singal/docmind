# Splits extracted PDF pages into smaller overlapping text chunks using LangChain's text splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Takes page-wise text from pdf_loader and splits into smaller chunks.
    Each chunk retains the metadata of its source page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]  # tries these in order
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        splits = splitter.split_text(text)

        for i, chunk_text in enumerate(splits):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,                    # source, page, total_pages
                    "chunk_index": i,              # which chunk within the page
                    "chunk_total": len(splits)     # total chunks from this page
                }
            })

    print(f"✅ Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


def chunk_pages_semantic(pages: List[Dict]) -> List[Dict]:
    """
    Alternative: splits text at double newlines (paragraph boundaries).
    More natural splits but inconsistent chunk sizes.
    Use this in your notebook to compare with chunk_pages().
    """
    all_chunks = []

    for page in pages:
        paragraphs = page["text"].split("\n\n")

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:   # skip very short fragments
                continue

            all_chunks.append({
                "text": para,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": i,
                    "chunk_total": len(paragraphs)
                }
            })

    print(f"✅ [Semantic] Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks




# if __name__ == "__main__":
#     from app.ingestion.pdf_loader import load_pdf

#     pages = load_pdf("data/uploads/test.pdf")
#     chunks = chunk_pages(pages)

#     # inspect first 3 chunks
#     for chunk in chunks[:3]:
#         print("--- CHUNK ---")
#         print(chunk["text"])
#         print("Metadata:", chunk["metadata"])
#         print()