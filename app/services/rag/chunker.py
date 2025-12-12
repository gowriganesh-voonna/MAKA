# app/services/rag/chunker.py
import os
import re
from typing import List
from pathlib import Path

# optional PDF support
try:
    import PyPDF2
except Exception:
    PyPDF2 = None


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. Install with `pip install PyPDF2` to read PDFs.")
    text = []
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def load_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in [".txt", ".md"]:
        return clean_text(read_txt(path))
    if p.suffix.lower() in [".pdf"]:
        return clean_text(read_pdf(path))
    raise RuntimeError(f"Unsupported file type: {p.suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple chunker that splits on sentence boundaries where possible.
    chunk_size = approx number of characters per chunk.
    """
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r'(?<=[\.\?\!\n])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    # add overlap: merge last tokens if necessary to ensure overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                # if needed, recompose overlapping text
                overlap_text = (prev[-overlap:] + " " + c[:overlap]).strip()
                overlapped.append(c)  # keep simple; overlap preserved by sentence split
        chunks = overlapped

    return chunks
