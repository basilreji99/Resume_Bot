"""
FastAPI backend for the Basil persona chatbot.

Endpoints:
    GET  /          — health check
    POST /chat      — send a message, get a reply

Run locally:
    uvicorn api.persona_chat:app --reload --port 8000

The index is loaded ONCE at startup — not per request.
This keeps response times fast (no re-embedding on every call).
"""

import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Make app/ importable when running from project root
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import chat
from app.guardrails import check_query

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "basil_index.faiss"
CHUNKS_PATH = DATA_DIR / "basil_chunks.pkl"

# ── Load index at startup ─────────────────────────────────────
# This runs once when the server starts. Loading from disk is
# fast (~1 second). All requests then share the same index.
print("Loading persona index...")

if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
    raise RuntimeError(
        "Index files not found. Run 'python scripts/build_index.py' first."
    )

faiss_index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"✓ Index loaded — {faiss_index.ntotal} vectors, {len(chunks)} chunks")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Basil Persona Bot API",
    description="Ask questions about Basil Reji",
    version="1.0.0"
)

# CORS — allows your website's JavaScript to call this API.
# Replace the origins list with your actual website URL before deploying.
# "*" means any origin — fine for local testing, but tighten before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # local React dev server
        "http://localhost:5500",      # local VS Code Live Server
        "http://127.0.0.1:5500",
        "https://basilreji.com",      # your live website
        "https://www.basilreji.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
# Pydantic models validate incoming JSON automatically.
# If a required field is missing, FastAPI returns a 422 error.

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # optional — list of {role, content} dicts
                                # empty list means a fresh conversation

class ChatResponse(BaseModel):
    reply: str
    history: list[dict]


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Simple health check — confirms the server is running."""
    return {
        "status": "ok",
        "message": "Basil persona bot is running",
        "vectors": faiss_index.ntotal
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.

    Receives a message and optional conversation history.
    Returns the bot's reply and the updated history.

    The client is responsible for storing and re-sending history
    on each turn — this keeps the API stateless and simple.
    """
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if len(message) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Message too long. Please keep questions under 1000 characters."
        )

    # Guardrail check
    passed, reason = check_query(message, " ".join(chunks))
    if not passed:
        updated_history = request.history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reason}
        ]
        return ChatResponse(reply=reason, history=updated_history)

    # RAG pipeline
    try:
        reply, updated_history = chat(
            user_message=message,
            chat_history=request.history,
            faiss_index=faiss_index,
            chunks=chunks,
            source_name="Basil Reji's profile",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong generating a response: {str(e)}"
        )

    return ChatResponse(reply=reply, history=updated_history)