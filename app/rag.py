import os
from groq import Groq
from dotenv import load_dotenv
from app.vector_store import search
from app.guardrails import check_query

load_dotenv()  # reads your .env file and loads GROQ_API_KEY into os.environ

# Initialise the Groq client once at module level.
# It automatically reads GROQ_API_KEY from the environment.
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# The model we're using. llama-3.3-70b-versatile is Groq's best
# free model as of 2026 — highly capable and very fast.
MODEL = "llama-3.3-70b-versatile"


# ── System prompt ─────────────────────────────────────────────
# This is the core of our guardrails. It tells the LLM:
#   1. What role it plays
#   2. What it's allowed to answer
#   3. How to behave when it doesn't know something
#   4. To never invent information
#
# {source_name} is a placeholder we fill in at runtime with the
# actual document name (e.g. "basil_resume.pdf")
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about a person \
based strictly on the information provided in their document: "{source_name}".

Rules you must always follow:
- Only use the context provided below each question to form your answer.
- If the context does not contain enough information to answer the question, \
say clearly: "I don't have enough information in the document to answer that."
- Never invent, assume, or infer details that are not explicitly stated in the context.
- Do not answer questions unrelated to the person or their document. \
If asked something off-topic (e.g. general knowledge, coding help, current events), \
respond: "I can only answer questions about the information in this document."
- Be concise, factual, and professional in your responses.
"""



# ── Build context string ──────────────────────────────────────
def _build_context(chunks: list[str]) -> str:
    """
    Formats the retrieved chunks into a clean context block
    that gets injected into the user's message.
    Numbering the chunks helps the LLM cite them implicitly.
    """
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
    Full RAG turn: retrieve → check relevance → build prompt → call LLM.

    Parameters:
        user_message  : the question the user just typed
        chat_history  : full conversation so far (list of role/content dicts)
        faiss_index   : the FAISS index built in Phase 3
        chunks        : the original text chunks (parallel to the index)
        source_name   : filename or URL label, used in the system prompt

    Returns:
        assistant_reply : the LLM's response string
        updated_history : chat_history with the new turn appended
    """

    # Step 1: Semantic guardrail — check query is relevant to document
    passed, reason = check_query(user_message, " ".join(chunks))
    if not passed:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": reason})
        return reason, chat_history

    # Step 2: Retrieve the most relevant chunks for this question
    relevant_chunks = search(user_message, faiss_index, chunks, top_k=4)

    # Step 3: Format context and inject it into the user message.
    context = _build_context(relevant_chunks)
    augmented_message = (
        f"Context from the document:\n\n{context}\n\n"
        f"Question: {user_message}"
    )

    # Step 4: Build the full messages list to send to the LLM.
    # Structure: [system] + [prior conversation] + [new user message]
    # The LLM sees the whole conversation each time — this is how
    # multi-turn memory works with stateless APIs.
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_name=source_name)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + chat_history
        + [{"role": "user", "content": augmented_message}]
    )

    # Step 5: Call the Groq API
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,     # low temperature = more factual, less creative
        max_tokens=1024,     # cap response length
    )
    assistant_reply = response.choices[0].message.content

    # Step 6: Append both turns to history for next round
    # Note: we store the original user_message (not the augmented one)
    # so the chat display looks clean to the user.
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, chat_history