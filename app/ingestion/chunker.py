# Splits extracted PDF pages into smaller overlapping text chunks using LangChain's text splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
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
                    **metadata,
                    "chunk_index": i,
                    "chunk_total": len(splits)
                }
            })

    print(f"✅ Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


def chunk_pages_semantic(pages: List[Dict]) -> List[Dict]:
    all_chunks = []

    for page in pages:
        paragraphs = page["text"].split("\n\n")

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:
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