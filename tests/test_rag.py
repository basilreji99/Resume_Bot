import pytest
from unittest.mock import MagicMock, patch
from app.vector_store import build_vector_store
from app.guardrails import cosine_similarity, is_query_relevant, check_query
from app.rag import chat
import numpy as np


# ── Shared fixture ────────────────────────────────────────────
RESUME_TEXT = (
    "Basil Ahmed is a software engineer based in Boston. "
    "He has 5 years of experience in Python and machine learning. "
    "He worked at Acme Corp as a data engineer from 2019 to 2022. "
    "He holds a Masters degree in Computer Science from MIT. "
    "His skills include Python, SQL, TensorFlow, and AWS."
)

@pytest.fixture(scope="module")
def vector_store():
    return build_vector_store(RESUME_TEXT)


# ── cosine_similarity ─────────────────────────────────────────
def test_cosine_similarity_identical():
    """A vector compared to itself should score 1.0."""
    v = np.array([[1.0, 0.5, -0.3]])
    score = cosine_similarity(v, v)
    assert abs(score - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    """Perpendicular vectors should score 0.0."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    score = cosine_similarity(a, b)
    assert abs(score) < 1e-5


def test_cosine_similarity_zero_vector():
    """Zero vector should not cause a crash — returns 0.0."""
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.5]])
    score = cosine_similarity(a, b)
    assert score == 0.0


# ── is_query_relevant ─────────────────────────────────────────
def test_relevant_query_passes():
    """A question about the resume should be flagged as relevant."""
    relevant, score = is_query_relevant(
        "What programming languages does he know?", RESUME_TEXT
    )
    assert relevant is True, f"Expected relevant but got score {score:.3f}"


def test_irrelevant_query_fails():
    """A clearly off-topic question should be flagged as irrelevant.
    We use a question with zero connection to a software resume."""
    relevant, score = is_query_relevant(
        "What is the boiling point of water in Kelvin?", RESUME_TEXT
    )
    assert relevant is False, f"Expected irrelevant but got score {score:.3f}"


# ── check_query ───────────────────────────────────────────────
def test_check_query_too_short():
    passed, msg = check_query("hi", RESUME_TEXT)
    assert passed is False
    assert "valid question" in msg


def test_check_query_on_topic():
    passed, _ = check_query("What is his work experience?", RESUME_TEXT)
    assert passed is True


def test_check_query_off_topic():
    passed, msg = check_query(
        "What is the capital of France?", RESUME_TEXT
    )
    assert passed is False
    assert "only answer questions" in msg


# ── chat() — mocking the Groq API ────────────────────────────
# We don't want to make real API calls in tests — they cost money,
# require a key, and are slow. We mock the Groq client so that
# any call to it returns a fake but realistic response object.

def make_mock_groq_response(content: str):
    """Helper to build a fake Groq API response object."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


def test_chat_returns_reply(vector_store):
    """chat() should return a string reply and updated history."""
    index, chunks = vector_store
    mock_response = make_mock_groq_response("Basil Ahmed is a software engineer.")

    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = mock_response
        reply, history = chat(
            user_message="What is his name?",
            chat_history=[],
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert history[-1]["role"] == "assistant"
    assert history[-2]["role"] == "user"


def test_chat_appends_history(vector_store):
    """Each chat() call should add exactly 2 items to history."""
    index, chunks = vector_store
    mock_response = make_mock_groq_response("He worked at Acme Corp.")
    history = [
        {"role": "user", "content": "What is his name?"},
        {"role": "assistant", "content": "His name is Basil Ahmed."}
    ]

    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = mock_response
        _, updated = chat(
            user_message="Where did he work?",
            chat_history=history,
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )

    assert len(updated) == 4  # 2 existing + 2 new


def test_chat_off_topic_blocked(vector_store):
    """Off-topic questions should be blocked before reaching the LLM."""
    index, chunks = vector_store

    with patch("app.rag.get_client") as mock_get_client:
        reply, _ = chat(
            user_message="What is the boiling point of water?",
            chat_history=[],
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )
        # The LLM should never have been called
        mock_get_client.return_value.chat.completions.create.assert_not_called()

    assert "only answer questions" in reply.lower() or "not" in reply.lower()