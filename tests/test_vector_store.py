import pytest
import numpy as np
from app.vector_store import chunk_text, embed_chunks, build_faiss_index, search, build_vector_store


# ── Shared fixture ────────────────────────────────────────────
# @pytest.fixture means pytest runs this once and injects the
# result into any test that lists it as a parameter.
# This avoids rebuilding the vector store in every single test.
@pytest.fixture(scope="module")
def sample_store():
    """
    Builds a small vector store from a fake resume.
    scope="module" means it's built once per test file, not per test.
    """
    text = (
        "Basil Ahmed is a software engineer based in Boston. "
        "He has 5 years of experience in Python and machine learning. "
        "He worked at Acme Corp as a data engineer from 2019 to 2022. "
        "He holds a Masters degree in Computer Science from MIT. "
        "His skills include Python, SQL, TensorFlow, and AWS. "
        "He is fluent in English and Arabic. "
        "He can be reached at basil@example.com or +1-555-0100."
    )
    index, chunks = build_vector_store(text)
    return index, chunks, text


# ── chunk_text ────────────────────────────────────────────────
def test_chunk_text_basic():
    text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1, "Long text should produce multiple chunks"


def test_chunk_text_overlap():
    """Consecutive chunks should share words due to overlap."""
    text = " ".join([f"word{i}" for i in range(400)])
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    # Last 50 words of chunk 0 should appear at start of chunk 1
    end_of_first = chunks[0].split()[-50:]
    start_of_second = chunks[1].split()[:50]
    assert end_of_first == start_of_second


def test_chunk_text_short_document():
    """A short document should produce exactly one chunk."""
    text = "This is a short resume with only a few words."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 1


def test_chunk_text_empty():
    """Empty string should return one empty chunk (graceful, not a crash)."""
    chunks = chunk_text("", chunk_size=300, overlap=50)
    assert isinstance(chunks, list)


# ── embed_chunks ──────────────────────────────────────────────
def test_embed_chunks_shape():
    """Each chunk should produce a 384-dimensional vector."""
    chunks = ["Python developer with 5 years experience", "Masters in Computer Science"]
    embeddings = embed_chunks(chunks)
    assert embeddings.shape == (2, 384)


def test_embed_chunks_dtype():
    """FAISS requires float32 — make sure the dtype is correct."""
    chunks = ["test chunk"]
    embeddings = embed_chunks(chunks)
    assert embeddings.dtype == np.float32


# ── FAISS index ───────────────────────────────────────────────
def test_build_faiss_index(sample_store):
    index, chunks, _ = sample_store
    assert index.ntotal == len(chunks), \
        "Index should contain one vector per chunk"


# ── search ────────────────────────────────────────────────────
def test_search_returns_relevant_chunk(sample_store):
    """Searching for 'education' should surface the MIT chunk."""
    index, chunks, _ = sample_store
    results = search("What is his education background?", index, chunks, top_k=2)
    combined = " ".join(results).lower()
    assert "mit" in combined or "masters" in combined or "computer science" in combined


def test_search_returns_correct_count(sample_store):
    """search should return at most top_k results."""
    index, chunks, _ = sample_store
    results = search("skills", index, chunks, top_k=3)
    assert len(results) <= 3


def test_search_name_query(sample_store):
    """Querying for name should return the chunk containing Basil Ahmed."""
    index, chunks, _ = sample_store
    results = search("What is his name?", index, chunks, top_k=2)
    combined = " ".join(results).lower()
    assert "basil" in combined