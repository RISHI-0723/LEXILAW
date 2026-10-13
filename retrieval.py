"""
retrieval.py
------------
Retrieval pipeline for LexiLaw:

  A. Jaccard similarity  -> remove near-duplicate chunks.
  B. Sentence-Transformer embeddings (all-MiniLM-L6-v2) -> local vectors.
  C. Semantic search     -> cosine similarity ranking, top-k retrieval.

Jaccard is used ONLY for duplicate detection. Semantic relevance uses
embedding cosine similarity.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from document_utils import Chunk


# ---------------------------------------------------------------------------
# A. Jaccard similarity (duplicate detection)
# ---------------------------------------------------------------------------

# Configurable near-duplicate threshold. If J(A, B) >= this, B is a duplicate.
JACCARD_THRESHOLD = 0.85

_WORD_RE = re.compile(r"[a-z0-9]+")


def _word_set(text: str) -> set:
    """Normalize text to a set of lowercase alphanumeric word tokens."""
    return set(_WORD_RE.findall(text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts: |A ∩ B| / |A ∪ B|.

    Returns 0.0 safely when both inputs are empty.
    """
    set_a = _word_set(text_a)
    set_b = _word_set(text_b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def deduplicate_chunks(
    chunks: List[Chunk], threshold: float = JACCARD_THRESHOLD
) -> List[Chunk]:
    """Remove near-duplicate chunks using sequential Jaccard comparison.

    Each new chunk is compared against already-kept chunks; if it is similar
    enough to one of them, it is skipped. The first (kept) chunk's text and
    metadata are preserved. This is O(n * kept) which is fine for a small
    college prototype.
    """
    kept: List[Chunk] = []
    for chunk in chunks:
        is_duplicate = any(
            jaccard_similarity(chunk.text, existing.text) >= threshold
            for existing in kept
        )
        if not is_duplicate:
            kept.append(chunk)
    return kept


# ---------------------------------------------------------------------------
# B. Sentence-Transformer embeddings
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The model is heavy to load, so cache a single instance per process.
_model = None


def get_model():
    """Load (once) and return the SentenceTransformer model.

    The model is downloaded from Hugging Face on first use and then cached
    locally by sentence-transformers.
    """
    global _model
    if _model is None:
        # Imported lazily so the rest of the module can be used/tested
        # without the heavy dependency installed.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts (shape: n x dim)."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Compute and attach a normalized embedding to each chunk (in place)."""
    if not chunks:
        return chunks
    embeddings = embed_texts([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, embeddings):
        chunk.embedding = vector
    return chunks


# ---------------------------------------------------------------------------
# C. Semantic search
# ---------------------------------------------------------------------------

# A soft floor on cosine similarity. Scores are NOT calibrated probabilities;
# this only helps the app decide when to abstain rather than guess.
DEFAULT_MIN_SCORE = 0.25


class SearchResult:
    """One retrieved chunk plus its retrieval (cosine) score."""

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = float(score)

    # Convenience passthroughs for the UI.
    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def page(self) -> Optional[int]:
        return self.chunk.page

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


def semantic_search(
    question: str,
    chunks: List[Chunk],
    top_k: int = 4,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[SearchResult]:
    """Rank chunks against a question by cosine similarity and return top-k.

    - Handles empty question or empty index by returning an empty list.
    - Chunks must already have embeddings attached (see embed_chunks).
    - Results below min_score are filtered out so the app can abstain.
    """
    if not question or not question.strip():
        return []
    if not chunks:
        return []

    # Build a matrix of chunk embeddings; skip any without an embedding.
    indexed = [c for c in chunks if c.embedding is not None]
    if not indexed:
        return []

    matrix = np.vstack([c.embedding for c in indexed])  # (n x dim), normalized
    query = embed_texts([question])[0]  # (dim,), normalized

    # Cosine similarity == dot product for unit vectors.
    scores = matrix @ query

    order = np.argsort(scores)[::-1][:top_k]
    results = [
        SearchResult(indexed[i], scores[i])
        for i in order
        if scores[i] >= min_score
    ]
    return results
