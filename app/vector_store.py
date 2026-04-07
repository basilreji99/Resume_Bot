import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Model ─────────────────────────────────────────────────────
# We load the embedding model once at module level so it isn't
# reloaded on every function call. This model runs locally —
# no API key needed. It converts text → 384-dimensional vectors.
# "all-MiniLM-L6-v2" is small (80MB), fast, and very accurate
# for semantic similarity tasks.
_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Chunking ──────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Splits a long document into overlapping chunks of words.

    WHY OVERLAP? Imagine a sentence that straddles two chunks —
    without overlap, its meaning gets split and neither chunk
    captures it fully. Overlap ensures context isn't lost at
    chunk boundaries.

    chunk_size: number of words per chunk (300 ≈ 1-2 paragraphs)
    overlap:    how many words from the end of one chunk are
                repeated at the start of the next
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so the next chunk
        # starts overlap words before where this one ended
        start += chunk_size - overlap

    return chunks


# ── Embedding ─────────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Converts a list of text chunks into a 2D numpy array of vectors.

    Each chunk becomes one row of 384 numbers.
    Shape of output: (num_chunks, 384)

    show_progress_bar=True gives a nice progress indicator in
    the terminal while the model is processing.
    """
    embeddings = _model.encode(chunks, show_progress_bar=True)
    # FAISS requires float32 specifically — not float64
    return np.array(embeddings).astype("float32")


# ── FAISS index ───────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from the embedding matrix.

    IndexFlatL2 = brute-force search using L2 (Euclidean) distance.
    This is the simplest FAISS index — it checks every vector
    to find the closest ones. Fast enough for small documents
    (a resume has maybe 20-50 chunks). For millions of vectors
    you'd use an approximate index like IndexIVFFlat instead.

    d = the number of dimensions (384 for our model)
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)   # adds all chunk vectors to the index
    return index


# ── Search ────────────────────────────────────────────────────
def search(query: str, index: faiss.IndexFlatL2,
           chunks: list[str], top_k: int = 4) -> list[str]:
    """
    Given a user question, finds the most relevant chunks.

    Steps:
    1. Embed the question into a vector (same model, same space)
    2. Ask FAISS: "which stored vectors are closest to this?"
    3. Return the actual text of those top_k chunks

    top_k=4 means we return the 4 most relevant chunks.
    These will be passed to the LLM as context in Phase 4.

    I = indices of the nearest neighbours in the index
    D = their distances (lower = more similar for L2)
    """
    query_vec = _model.encode([query]).astype("float32")
    D, I = index.search(query_vec, top_k)

    # I[0] is the list of indices for the first (and only) query
    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return results


# ── Main pipeline entry point ─────────────────────────────────
def build_vector_store(text: str) -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Full pipeline: raw text → chunks → embeddings → FAISS index.
    Returns both the index and the original chunks list.
    We need to keep chunks because FAISS only stores vectors —
    we need the original text to pass to the LLM later.
    """
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    return index, chunks