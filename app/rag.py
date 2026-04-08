import os
import re
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
from app.vector_store import search
from app.guardrails import check_query

load_dotenv()

_client = None


def get_client() -> Groq:
    """Returns the shared Groq client, creating it on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        _client = Groq(api_key=api_key)
    return _client


MODEL = "llama-3.3-70b-versatile"


# ── System prompt ─────────────────────────────────────────────
# Resume_Bot is a general-purpose document Q&A tool — NOT a persona bot.
# The system prompt reflects this: it's professional, document-focused,
# and works for any uploaded resume or profile, not just Basil's.
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about a person \
based strictly on the information provided in their document: "{source_name}".

Rules you must always follow:
- Only use the context provided below each question to form your answer.
- If the context does not contain enough information to answer, say clearly: \
"I don't have enough information in this document to answer that."
- Never invent, assume, or infer details not explicitly stated in the context.
- Do not answer questions unrelated to the person or their document. \
If asked something off-topic, respond: \
"I can only answer questions about the uploaded document."
- For short social exchanges like "thanks" or "that's helpful" — respond briefly \
and warmly, then invite a follow-up question about the document.
- Be concise, factual, and professional.
"""


# ── Greeting detection ────────────────────────────────────────
GREETING_PATTERNS = [
    ({"how are you", "how are you doing", "how do you do",
      "how's it going", "how is it going"},
     "I'm doing well, thanks! Feel free to ask me anything about the uploaded document."),

    ({"what's up", "whats up", "sup"},
     "Not much — ready to help! Ask me anything about the document."),

    ({"hi", "hello", "hey", "howdy", "hiya", "greetings",
      "good morning", "good afternoon", "good evening"},
     "Hello! 👋 I'm ready to answer questions about the uploaded document. What would you like to know?"),
]


def _extract_greeting_and_question(message: str) -> tuple[str | None, str | None]:
    """
    Splits a message into a greeting response and a question part.
    e.g. "Hi! What are their skills?" → (greeting_reply, "What are their skills?")
    """
    parts = re.split(r'[.!?]+|,\s*(?=[A-Z])', message)
    parts = [p.strip() for p in parts if p.strip()]

    greeting_response = None
    remaining_parts = []

    for part in parts:
        cleaned = part.lower().strip("?.,!").strip()
        matched = False
        for patterns, response in GREETING_PATTERNS:
            if cleaned in patterns or any(cleaned.startswith(g) for g in patterns):
                greeting_response = response
                matched = True
                break
        if not matched:
            remaining_parts.append(part)

    question = " ".join(remaining_parts).strip() if remaining_parts else None
    return greeting_response, question


# ── Query rewriter ────────────────────────────────────────────
def _rewrite_query(user_message: str, chat_history: list[dict]) -> str:
    """
    Rewrites vague follow-ups into specific standalone questions so
    FAISS retrieval has enough semantic content to find relevant chunks.
    e.g. "Can you elaborate?" → "What additional details exist about their work experience?"
    Only triggers for short messages containing vague follow-up words.
    """
    if not chat_history:
        return user_message

    vague_triggers = {
        "elaborate", "more", "else", "expand", "continue", "go on",
        "tell me more", "what about", "also", "further",
        "can you", "could you", "explain", "detail"
    }
    words = set(user_message.lower().strip("?.,!").split())
    is_vague = len(user_message.split()) <= 8 and bool(words & vague_triggers)

    if not is_vague:
        return user_message

    recent = chat_history[-4:]
    history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:200]}"
        for m in recent
    )

    rewrite_prompt = (
        f"Given this conversation:\n{history_str}\n\n"
        f"The user then said: \"{user_message}\"\n\n"
        f"Rewrite the user's message as a specific, standalone question "
        f"that can be answered without seeing the conversation history. "
        f"Return only the rewritten question, nothing else."
    )

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        print(f"[Query rewrite] '{user_message}' -> '{rewritten}'")
        return rewritten
    except Exception:
        return user_message


# ── Context builder ───────────────────────────────────────────
def _build_context(chunks: list[str]) -> str:
    """Formats retrieved chunks into a numbered context block."""
    sections = [f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)]
    return "\n\n".join(sections)


# ── Main chat function ────────────────────────────────────────
def chat(
    user_message: str,
    chat_history: list[dict],
    faiss_index,
    chunks: list[str],
    source_name: str,
) -> tuple[str, list[dict]]:
    """
    Full RAG turn: greeting check → guardrail → rewrite → retrieve → LLM.

    Parameters:
        user_message  : the question the user just typed
        chat_history  : full conversation so far (list of role/content dicts)
        faiss_index   : the FAISS index built from the uploaded document
        chunks        : the original text chunks (parallel to the index)
        source_name   : filename or label, used in the system prompt

    Returns:
        (assistant_reply, updated_history)
    """

    # Step 1: Handle greetings — split into greeting + question if both present
    greeting_response, question_part = _extract_greeting_and_question(user_message)

    if greeting_response and not question_part:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": greeting_response})
        return greeting_response, chat_history

    if greeting_response and question_part:
        # Greeting + question — answer the question, prepend the greeting
        _, temp_history = chat(
            user_message=question_part,
            chat_history=list(chat_history),
            faiss_index=faiss_index,
            chunks=chunks,
            source_name=source_name,
        )
        question_answer = temp_history[-1]["content"]
        combined = f"{greeting_response}\n\n{question_answer}"
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": combined})
        return combined, chat_history

    # Step 2: Guardrail — block empty or nonsensical input
    passed, reason = check_query(user_message, " ".join(chunks))
    if not passed:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": reason})
        return reason, chat_history

    # Step 3: Rewrite vague follow-ups into specific standalone questions
    retrieval_query = _rewrite_query(user_message, chat_history)

    # Step 4: Retrieve chunks — more for follow-ups to get fresh material
    is_followup = retrieval_query != user_message
    top_k = 6 if is_followup else 4
    relevant_chunks = search(retrieval_query, faiss_index, chunks, top_k=top_k)

    # Step 5: Format context — tell LLM what was already said to avoid repeats
    context = _build_context(relevant_chunks)
    previous_reply = ""
    if chat_history and chat_history[-1]["role"] == "assistant":
        previous_reply = (
            f"\nYou already told the user this:\n"
            f"\"{chat_history[-1]['content'][:300]}\"\n"
            f"Do NOT repeat this. Provide new, additional information only.\n"
        )

    augmented_message = (
        f"Context from the document:\n\n{context}\n"
        f"{previous_reply}\n"
        f"Question: {user_message}"
    )

    # Step 6: Build full messages list — system + history + new message
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_name=source_name)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + chat_history
        + [{"role": "user", "content": augmented_message}]
    )

    # Step 7: Call the Groq API
    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,   # slightly factual for a document Q&A tool
            max_tokens=1024,
        )
        assistant_reply = response.choices[0].message.content
    except RateLimitError:
        assistant_reply = (
            "I'm receiving too many requests right now. "
            "Please wait a few seconds and try again."
        )
    except APIError as e:
        assistant_reply = "Something went wrong on my end. Please try again in a moment."
        print(f"[Groq APIError] {e}")

    # Step 8: Append both turns to history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, chat_history