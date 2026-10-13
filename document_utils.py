"""
document_utils.py
-----------------
Text extraction and chunking for LexiLaw.

Supports PDF (via PyMuPDF / fitz) and DOCX (via python-docx).

Every extracted passage and chunk keeps its source metadata:
    - source_id : a unique id for the uploaded document
    - filename  : the original file name
    - page      : 1-based page number for PDFs, or None for DOCX
    - chunk_id  : a unique id for the chunk (added during chunking)

The design is deliberately simple so it is easy to explain in a viva.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pymupdf as fitz  # PyMuPDF (modern import name)
except ImportError:  # older PyMuPDF releases only expose "fitz"
    import fitz
import docx  # python-docx


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    """A piece of text extracted from a document, with its source metadata."""
    text: str
    source_id: str
    filename: str
    page: Optional[int] = None  # 1-based for PDF, None for DOCX


@dataclass
class Chunk:
    """A retrieval unit built from one or more passages of the SAME document."""
    text: str
    source_id: str
    filename: str
    chunk_id: str
    page: Optional[int] = None
    # Embedding is attached later by retrieval.py (kept here for convenience).
    embedding: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

# Target chunk size in words. Short paragraphs are combined up to this size;
# very long paragraphs are split into pieces no larger than this. Kept modest
# so a short legal document yields several focused, clause-level passages
# instead of one giant chunk (see extract_pdf_text / chunk_document).
MAX_CHUNK_WORDS = 150
# Paragraphs below this many words are candidates to merge with neighbours.
MIN_CHUNK_WORDS = 40


# Narrow fix for a known PDF font-encoding issue: in some PDFs the rupee
# symbol (U+20B9, ₹) is extracted as a bare lowercase 'l'. We only convert an
# 'l' that is NOT part of a word/number and is immediately followed by a digit
# (e.g. "l18,000" -> "₹18,000"). Ordinary words such as "legal", "will",
# "clause", and amounts like "model 3" or "level 3" are never touched.
_RUPEE_MISREAD = re.compile(r"(?<![A-Za-z0-9])l(?=\d)")


def clean_extracted_text(text: str) -> str:
    """Apply narrow, lossless cleanups to freshly extracted text."""
    return _RUPEE_MISREAD.sub("₹", text)


def split_paragraphs(page_text: str) -> List[str]:
    """Split a page's text into paragraph units.

    Paragraphs are separated by blank lines. Within a paragraph, wrapped
    single newlines are collapsed to spaces so excerpts read naturally.
    """
    paragraphs: List[str] = []
    for block in re.split(r"\n\s*\n", page_text):
        para = re.sub(r"\s*\n\s*", " ", block).strip()
        para = re.sub(r"[ \t]{2,}", " ", para)
        if para:
            paragraphs.append(para)
    return paragraphs


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(file_bytes: bytes, filename: str, source_id: str) -> List[Passage]:
    """Extract text from a PDF as paragraph-level passages, preserving pages.

    Each page is split into paragraphs so that chunk_document can build small,
    focused passages instead of treating a whole page as one unit.

    Raises ValueError if the PDF cannot be opened or contains no extractable
    text (e.g. a scanned / image-only PDF). OCR is not supported in this
    first version.
    """
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # malformed / unreadable file
        raise ValueError(f"Could not open PDF '{filename}': {exc}") from exc

    passages: List[Passage] = []
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            raw = page.get_text("text")
            if not raw.strip():
                continue
            for para in split_paragraphs(raw):
                passages.append(
                    Passage(
                        text=clean_extracted_text(para),
                        source_id=source_id,
                        filename=filename,
                        page=page_index + 1,  # 1-based page numbers
                    )
                )
    finally:
        document.close()

    if not passages:
        raise ValueError(
            f"No extractable text found in '{filename}'. It may be a scanned "
            f"or image-only PDF. OCR is not supported in this version."
        )
    return passages


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def extract_docx_text(file_bytes: bytes, filename: str, source_id: str) -> List[Passage]:
    """Extract text from a DOCX file in document order.

    Paragraphs and (optionally) table cells are extracted. Empty paragraphs
    are ignored. DOCX has no reliable page numbers, so page is left as None.
    """
    import io

    try:
        document = docx.Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not open DOCX '{filename}': {exc}") from exc

    passages: List[Passage] = []

    # Paragraphs in document order.
    for paragraph in document.paragraphs:
        text = clean_extracted_text(paragraph.text.strip())
        if text:
            passages.append(
                Passage(text=text, source_id=source_id, filename=filename, page=None)
            )

    # Tables (kept simple: join each row's cells with " | ").
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                passages.append(
                    Passage(
                        text=clean_extracted_text(" | ".join(cells)),
                        source_id=source_id,
                        filename=filename,
                        page=None,
                    )
                )

    if not passages:
        raise ValueError(f"No extractable text found in '{filename}'.")
    return passages


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_document(file_bytes: bytes, filename: str, source_id: str) -> List[Passage]:
    """Extract text from a supported document based on its file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_text(file_bytes, filename, source_id)
    if lower.endswith(".docx"):
        return extract_docx_text(file_bytes, filename, source_id)
    raise ValueError(
        f"Unsupported file type: '{filename}'. Please upload a PDF or DOCX file."
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_long_paragraph(text: str, max_words: int) -> List[str]:
    """Split one long paragraph into sentence-grouped pieces <= max_words.

    Falls back to a hard word-based split if a single 'sentence' is huge.
    """
    # Simple sentence split on ., !, ? followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current_words + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current = []
            current_words = 0
        # A single sentence longer than max_words: hard-split by words.
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue
        current.extend(words)
        current_words += len(words)

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(passages: List[Passage]) -> List[Chunk]:
    """Turn extracted passages into retrieval chunks.

    Rules:
      - Split primarily on paragraph boundaries (each Passage is a paragraph).
      - Keep paragraphs separate; combine only short consecutive fragments
        (until the buffer reaches MIN_CHUNK_WORDS) so a short document yields
        several focused passages instead of one giant chunk.
      - Split exceptionally long passages (> MAX_CHUNK_WORDS) into pieces.
      - Never combine content from DIFFERENT documents or DIFFERENT pages.
      - Preserve filename, source_id, page, and assign a unique chunk_id.
    """
    chunks: List[Chunk] = []
    chunk_counter = 0

    # Buffer of passages waiting to be merged (same source + same page only).
    buffer_text: List[str] = []
    buffer_words = 0
    buffer_source: Optional[str] = None
    buffer_filename: Optional[str] = None
    buffer_page: Optional[int] = None

    def flush():
        nonlocal buffer_text, buffer_words, chunk_counter
        nonlocal buffer_source, buffer_filename, buffer_page
        if buffer_text:
            chunk_counter += 1
            chunks.append(
                Chunk(
                    text="\n".join(buffer_text).strip(),
                    source_id=buffer_source,
                    filename=buffer_filename,
                    page=buffer_page,
                    chunk_id=f"{buffer_source}::c{chunk_counter}",
                )
            )
        buffer_text = []
        buffer_words = 0

    for passage in passages:
        text = passage.text.strip()
        if not text:
            continue

        words = text.split()

        # Boundary: flush if the document or page changes (never mix sources/pages).
        if (
            buffer_text
            and (
                passage.source_id != buffer_source
                or passage.page != buffer_page
            )
        ):
            flush()

        buffer_source = passage.source_id
        buffer_filename = passage.filename
        buffer_page = passage.page

        # A very long single passage: flush buffer, then emit split pieces.
        if len(words) > MAX_CHUNK_WORDS:
            flush()
            buffer_source = passage.source_id
            buffer_filename = passage.filename
            buffer_page = passage.page
            for piece in _split_long_paragraph(text, MAX_CHUNK_WORDS):
                chunk_counter += 1
                chunks.append(
                    Chunk(
                        text=piece,
                        source_id=passage.source_id,
                        filename=passage.filename,
                        page=passage.page,
                        chunk_id=f"{passage.source_id}::c{chunk_counter}",
                    )
                )
            continue

        # Keep paragraphs separate so a short document does not collapse into
        # one chunk. Flush the running buffer first when it already stands on
        # its own (>= MIN_CHUNK_WORDS) or when adding this paragraph would
        # exceed MAX_CHUNK_WORDS; only genuinely short fragments get merged.
        if buffer_text and (
            buffer_words >= MIN_CHUNK_WORDS
            or buffer_words + len(words) > MAX_CHUNK_WORDS
        ):
            flush()
            buffer_source = passage.source_id
            buffer_filename = passage.filename
            buffer_page = passage.page

        buffer_text.append(text)
        buffer_words += len(words)

    flush()
    return chunks
