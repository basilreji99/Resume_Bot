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
    threshold: float = 0.15
) -> tuple[bool, float]:
    """
    Checks whether a user question is semantically related to the document.

    Strategy: embed the query and compare it against SEVERAL short samples
    from the document (beginning, middle, end), then take the MAXIMUM score.
    This works better than comparing against the whole document because:
    - A question is short and stylistically different from a long document
    - Averaging the whole document into one vector dilutes specific topics
    - Taking the max score across samples catches questions about any section

    Threshold 0.15 is intentionally lenient — the system prompt already
    instructs the LLM to refuse off-topic questions. This layer only blocks
    clearly unrelated queries (weather, maths, general trivia etc.)

    Returns:
        (is_relevant: bool, score: float) — score is the best match found.
    """
    words = document_text.split()
    total = len(words)

    # Sample up to 3 sections: start, middle, end (each 300 words)
    # For short documents some samples will overlap — that's fine
    sample_size = 300
    starts = [0, max(0, total // 2 - sample_size // 2), max(0, total - sample_size)]
    samples = [" ".join(words[s: s + sample_size]) for s in starts]
    # Deduplicate in case of short documents
    samples = list(dict.fromkeys(samples))

    query_vec = _model.encode([query], convert_to_numpy=True)
    doc_vecs = _model.encode(samples, convert_to_numpy=True)

    # Take the best (maximum) similarity score across all samples
    scores = [cosine_similarity(query_vec, doc_vecs[i:i+1]) for i in range(len(samples))]
    best_score = max(scores)

    return best_score >= threshold, best_score


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