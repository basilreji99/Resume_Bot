import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse the same model as vector_store.py — loaded once at module level.
# Importing a second instance would waste ~80MB of memory.
from app.vector_store import _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures how similar two vectors are by the angle between them.

    Returns a value between -1 and 1:
      1.0  = identical direction (highly similar meaning)
      0.0  = perpendicular (unrelated)
     -1.0  = opposite directions (opposite meaning)

    For text embeddings in practice, scores stay between 0 and 1.
    A score above ~0.25 generally means the texts are topically related.

    Formula: cos(θ) = (A · B) / (|A| × |B|)
    np.dot   = dot product (A · B)
    np.linalg.norm = magnitude (|A|)
    """
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_query_relevant(
    query: str,
    document_text: str,
    threshold: float = 0.25
) -> tuple[bool, float]:
    """
    Checks whether a user question is semantically related to the document.

    Strategy: embed the query AND a sample of the document text, then
    compare them with cosine similarity. If the score is below the
    threshold, the question is probably off-topic.

    We use only the first 1000 words of the document as the comparison
    target — the intro/summary section is the richest in terms of topics
    and is fastest to embed. For a resume this is more than enough.

    Returns:
        (is_relevant: bool, score: float)
        The score is returned so it can be logged or shown in debug mode.
    """
    # Take a representative sample to keep embedding fast
    doc_sample = " ".join(document_text.split()[:1000])

    query_vec = _model.encode([query], convert_to_numpy=True)
    doc_vec = _model.encode([doc_sample], convert_to_numpy=True)

    score = cosine_similarity(query_vec, doc_vec)
    return score >= threshold, score


def check_query(query: str, document_text: str) -> tuple[bool, str]:
    """
    Main entry point for the guardrail layer.
    Returns (passed: bool, reason: str).

    'passed=True'  means the query should proceed to the LLM.
    'passed=False' means we should return a guardrail message directly.
    """
    # Check 1: minimum length
    if len(query.strip()) < 3:
        return False, "Please enter a valid question."

    # Check 2: semantic relevance
    relevant, score = is_query_relevant(query, document_text)
    if not relevant:
        return False, (
            "I can only answer questions about the information in this document. "
            "Your question doesn't appear to be related to its content."
        )

    return True, ""