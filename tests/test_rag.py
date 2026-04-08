import pytest
from unittest.mock import MagicMock, patch
from app.vector_store import build_vector_store
from app.guardrails import cosine_similarity, check_query
from app.rag import chat, _extract_greeting_and_question, _rewrite_query
import numpy as np


# ── Shared fixture ────────────────────────────────────────────
RESUME_TEXT = (
    "Jane Smith is a software engineer based in New York. "
    "She has 5 years of experience in Python and machine learning. "
    "She worked at Acme Corp as a data engineer from 2019 to 2022. "
    "She holds a Masters degree in Computer Science from MIT. "
    "Her skills include Python, SQL, TensorFlow, and AWS."
)

@pytest.fixture(scope="module")
def vector_store():
    return build_vector_store(RESUME_TEXT)


# ── cosine_similarity ─────────────────────────────────────────
def test_cosine_similarity_identical():
    v = np.array([[1.0, 0.5, -0.3]])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

def test_cosine_similarity_orthogonal():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert abs(cosine_similarity(a, b)) < 1e-5

def test_cosine_similarity_zero_vector():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.5]])
    assert cosine_similarity(a, b) == 0.0


# ── check_query ───────────────────────────────────────────────
def test_check_query_empty():
    passed, msg = check_query("", RESUME_TEXT)
    assert passed is False
    assert "valid question" in msg

def test_check_query_single_char():
    passed, _ = check_query("a", RESUME_TEXT)
    assert passed is False

def test_check_query_hi_passes():
    """'Hi' is 2 chars and must pass — greeting handler deals with it."""
    passed, _ = check_query("Hi", RESUME_TEXT)
    assert passed is True

def test_check_query_numbers_only():
    passed, _ = check_query("12345", RESUME_TEXT)
    assert passed is False

def test_check_query_normal_question():
    passed, _ = check_query("What is her work experience?", RESUME_TEXT)
    assert passed is True

def test_check_query_abstract_question():
    """Abstract phrasing must pass — the LLM handles refusal."""
    passed, _ = check_query("What makes her stand out as a candidate?", RESUME_TEXT)
    assert passed is True


# ── Greeting detection ────────────────────────────────────────
def test_greeting_only():
    greeting, question = _extract_greeting_and_question("Hi!")
    assert greeting is not None
    assert question is None

def test_greeting_with_question():
    greeting, question = _extract_greeting_and_question("Hello! What are her skills?")
    assert greeting is not None
    assert question is not None
    assert "skills" in question.lower()

def test_no_greeting_just_question():
    greeting, question = _extract_greeting_and_question("What is her education background?")
    assert greeting is None

def test_how_are_you():
    greeting, question = _extract_greeting_and_question("How are you?")
    assert greeting is not None
    assert "doing well" in greeting.lower() or "ready" in greeting.lower()


# ── Query rewriter ────────────────────────────────────────────
def test_rewrite_skips_specific_question():
    """Specific questions should not be rewritten."""
    result = _rewrite_query("What university did she attend?", [])
    assert result == "What university did she attend?"

def test_rewrite_skips_without_history():
    """No history means nothing to rewrite from."""
    result = _rewrite_query("tell me more", [])
    assert result == "tell me more"

def test_rewrite_triggers_for_vague_followup():
    """Vague follow-up with history should trigger rewrite via LLM."""
    history = [
        {"role": "user", "content": "What is her education?"},
        {"role": "assistant", "content": "She holds a Masters from MIT."}
    ]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "What additional details exist about her education at MIT?"

    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = mock_response
        result = _rewrite_query("can you elaborate?", history)

    assert result != "can you elaborate?"
    assert len(result) > 10


# ── chat() ────────────────────────────────────────────────────
def make_mock_response(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


def test_chat_greeting_returns_without_llm_call(vector_store):
    """Pure greetings should not call the LLM at all."""
    index, chunks = vector_store
    with patch("app.rag.get_client") as mock_get_client:
        reply, history = chat(
            user_message="Hi",
            chat_history=[],
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )
        mock_get_client.return_value.chat.completions.create.assert_not_called()
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_chat_returns_reply(vector_store):
    """Normal question should return a string reply."""
    index, chunks = vector_store
    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = \
            make_mock_response("Jane Smith is a software engineer.")
        reply, history = chat(
            user_message="What is her name?",
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
    """Each chat() call adds exactly 2 items to history."""
    index, chunks = vector_store
    history = [
        {"role": "user", "content": "What is her name?"},
        {"role": "assistant", "content": "Her name is Jane Smith."}
    ]
    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = \
            make_mock_response("She worked at Acme Corp.")
        _, updated = chat(
            user_message="Where did she work?",
            chat_history=history,
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )
    assert len(updated) == 4


def test_chat_rate_limit_returns_friendly_message(vector_store):
    """Groq RateLimitError should return a friendly message, not crash."""
    from groq import RateLimitError
    index, chunks = vector_store
    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.side_effect = \
            RateLimitError("rate limit", response=MagicMock(status_code=429), body={})
        reply, _ = chat(
            user_message="What are her skills?",
            chat_history=[],
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )
    assert "wait" in reply.lower() or "try again" in reply.lower()


def test_chat_off_topic_reaches_llm(vector_store):
    """Off-topic questions pass the guardrail and are handled by the LLM."""
    index, chunks = vector_store
    with patch("app.rag.get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = \
            make_mock_response("I can only answer questions about the uploaded document.")
        reply, _ = chat(
            user_message="What is the capital of France?",
            chat_history=[],
            faiss_index=index,
            chunks=chunks,
            source_name="test_resume.pdf",
        )
        mock_get_client.return_value.chat.completions.create.assert_called_once()
    assert isinstance(reply, str)