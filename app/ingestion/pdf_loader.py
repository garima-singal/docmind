# Handles PDF text extraction using PyMuPDF with OCR fallback for scanned/image-based PDFs

import fitz  # PyMuPDF
import os
from typing import List, Dict
import platform


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.
    Automatically falls back to OCR if page text is empty (scanned PDF).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    ocr_used = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        # if no text extracted, attempt OCR on this page
        if not text:
            text = _ocr_page(file_path, page_num)
            if text:
                ocr_used = True

        if not text:
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source": os.path.basename(file_path),
                "page": page_num + 1,
                "total_pages": len(doc)
            }
        })

    doc.close()

    if ocr_used:
        print(f"⚠️  OCR was used — scanned PDF detected")
    print(f"✅ Loaded {len(pages)} pages from '{os.path.basename(file_path)}'")
    return pages


def _ocr_page(file_path: str, page_num: int) -> str:
    """
    Runs OCR on a single page using pytesseract.
    Falls back gracefully if pytesseract or poppler is not installed.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path



        if platform.system() == "Windows":
             pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        images = convert_from_path(
            file_path,
            first_page=page_num + 1,
            last_page=page_num + 1
        )

        if images:
            text = pytesseract.image_to_string(images[0])
            return text.strip()

    except ImportError:
        print("⚠️  pytesseract or pdf2image not installed. Cannot OCR this page.")
    except Exception as e:
        print(f"⚠️  OCR failed on page {page_num + 1}: {e}")

    return ""


def load_multiple_pdfs(file_paths: List[str]) -> List[Dict]:
    """
    Load multiple PDFs and combine their pages into one list.
    """
    all_pages = []
    for path in file_paths:
        try:
            pages = load_pdf(path)
            all_pages.extend(pages)
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")

    print(f"✅ Total pages loaded: {len(all_pages)}")
    return all_pages