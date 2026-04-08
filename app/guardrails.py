import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns a value between -1 and 1. Higher = more similar.
    """
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_query(query: str, document_text: str) -> tuple[bool, str]:
    """
    Lightweight guardrail — blocks only empty or non-alphabetic input.

    Minimum length is 2 so short inputs like "Hi" pass through to the
    greeting handler in rag.py rather than being blocked here.

    Off-topic detection is delegated entirely to the LLM system prompt,
    which handles abstract phrasing ("summarise", "what stands out",
    "tell me more") far better than any word-matching or semantic check.
    """
    if len(query.strip()) < 2:
        return False, "Please enter a valid question."

    if not any(c.isalpha() for c in query):
        return False, "Please enter a valid question."

    return True, ""