# Resume Bot

A general-purpose AI-powered document Q&A tool. Upload any resume, CV, or profile document (PDF or DOCX), or paste text directly, and ask questions about it in natural language. Built using Retrieval-Augmented Generation (RAG), Groq's LLM API, sentence-transformers for local embeddings, and Streamlit for the UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [How to Run Locally](#4-how-to-run-locally)
5. [Underlying Technologies and Concepts](#5-underlying-technologies-and-concepts)
6. [Owner's Guide — How to Maintain and Extend This Project](#6-owners-guide)

---

## 1. Project Overview

Resume Bot is a Streamlit web application that lets you upload any resume or profile document and have a conversation about its contents. The chatbot answers strictly from the uploaded document, refuses off-topic questions, and maintains multi-turn conversation memory within a session.

**Key features:**
- Upload PDF or DOCX files, or paste text directly
- Multi-turn conversation with memory across a session
- Handles greetings and small talk naturally
- Refuses off-topic questions politely via LLM system prompt
- Vague follow-up rewriting — "tell me more" retrieves fresh relevant chunks
- Graceful rate limit handling — friendly message instead of raw API error
- Reset button clears document and chat history completely
- Starter question buttons for quick exploration
- Runs fully locally — no deployment needed

---

## 2. Architecture

```
User (browser — Streamlit UI)
    │
    │  Types a question
    ▼
streamlit_app.py
    │
    ├── Greeting handler → canned response (no LLM call)
    │
    ├── Guardrail check → block empty/invalid input
    │
    ├── Query rewriter → rewrites vague follow-ups using Groq
    │        e.g. "elaborate" → "What additional details exist about their skills?"
    │
    ├── FAISS search → find top-k relevant chunks from uploaded document
    │        (using sentence-transformers embedding of the query)
    │
    ├── Context builder → format chunks + previous reply
    │
    └── Groq LLM (Llama 3.3 70B) → generate answer
            │
            ▼
        Answer displayed in Streamlit chat bubble
```

**Data flow when a document is uploaded:**
```
User uploads PDF / DOCX / pastes text
    │
    ▼
app/parser.py or app/scraper.py
    │  Extract raw text
    ▼
app/vector_store.py
    │
    ├── chunk_text()     — split into 300-word overlapping chunks
    ├── embed_chunks()   — sentence-transformers → 384-dimensional vectors
    └── build_faiss_index() — store vectors in FAISS IndexFlatL2
    │
    ▼
st.session_state["faiss_index"] + st.session_state["chunks"]
(stored in memory for the duration of the session)
```

---

## 3. Tech Stack

| Component | Technology | Why |
|---|---|---|
| UI framework | Streamlit | Build interactive web apps in pure Python |
| LLM | Groq (Llama 3.3 70B) | Free tier, extremely fast, OpenAI-compatible |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, no API key needed |
| Vector store | FAISS (faiss-cpu) | In-memory, fast, no server needed |
| PDF parsing | PyMuPDF (fitz) | Reliable text extraction from PDFs |
| DOCX parsing | python-docx | Extracts text from Word documents |
| Text cleaning | custom scraper.py | Cleans pasted text |
| Testing | pytest + pytest-mock | Unit tests for every component |

---

## 4. How to Run Locally

**Prerequisites:** Python 3.10+, Git

```bash
# Clone the repo
git clone https://github.com/basilreji99/Resume_Bot.git
cd Resume_Bot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create a file called .env in the project root:
# GROQ_API_KEY=your_key_here
# Get a free key at https://console.groq.com

# Run the app
streamlit run streamlit_app.py
```

The app opens automatically at `http://localhost:8501`.

**Run the test suite:**
```bash
pytest tests/ -v
```

---

## 5. Underlying Technologies and Concepts

This section explains every AI, ML, and systems concept used in this project in enough depth to answer technical interview questions confidently.

---

### 5.1 Large Language Models (LLMs)

**What is an LLM?**
A Large Language Model is a neural network trained on massive amounts of text to predict the next token in a sequence. Through this training it develops representations of language, facts, and reasoning. GPT-4, Llama 3, and Mistral are all LLMs.

**How does an LLM generate text?**
Given input tokens, the model outputs a probability distribution over its vocabulary for the next token. It samples from this distribution (controlled by temperature), appends the token, and repeats. This is called autoregressive generation.

**What is temperature?**
Controls how peaked or flat the probability distribution is before sampling. Temperature 0 = always pick the highest probability token (deterministic). Temperature 1 = sample proportionally to raw probabilities. Higher = more creative/random. This project uses 0.3 — factual and consistent, appropriate for a document Q&A tool.

**What is a context window?**
The maximum number of tokens an LLM can process in one call — both input and output combined. Llama 3.3 70B supports 128k tokens. All retrieved chunks plus conversation history must fit within this limit.

**What is a system prompt?**
A hidden instruction sent at the start of every conversation that shapes LLM behaviour. Users never see it. In this project it instructs the LLM to answer only from provided context, refuse off-topic questions, and handle small talk gracefully.

**What is prompt engineering?**
The practice of crafting LLM inputs to produce desired behaviour — writing clear instructions, setting constraints, defining persona, specifying fallback behaviour. The system prompt in this project is a form of prompt engineering.

**What is Groq?**
An AI inference company that built Language Processing Units (LPUs) — custom chips optimised for the sequential autoregressive decoding step of LLMs. Groq generates 500–800 tokens/second, roughly 10x faster than GPU-based providers. Their free tier requires no credit card.

---

### 5.2 Embeddings and Vector Representations

**What is an embedding?**
A dense vector of floating-point numbers representing the meaning of text in high-dimensional space. Texts with similar meanings produce vectors that are close together. This project uses 384-dimensional vectors — each piece of text becomes a list of 384 numbers.

**How are embedding models trained?**
Using contrastive learning. Pairs of semantically similar texts are trained to produce similar vectors (pulled together), while dissimilar pairs produce distant vectors (pushed apart). The model learns what "similar meaning" looks like across millions of examples.

**What model is used here?**
`all-MiniLM-L6-v2` from sentence-transformers. It is a distilled version of BERT, fine-tuned for semantic similarity. It produces 384-dimensional vectors, is only ~80MB, runs on CPU, and requires no API key. This makes it ideal for a local application.

**What is cosine similarity?**
Measures the angle between two vectors regardless of magnitude. Formula: `cos(θ) = (A · B) / (|A| × |B|)`. Returns 1.0 for identical direction (highly similar), 0.0 for perpendicular (unrelated). Used for semantic similarity because vector magnitude doesn't carry meaning — only direction does.

**What is the difference between symmetric and asymmetric embeddings?**
Symmetric: query and document are embedded the same way — used when they are similar in nature (e.g. duplicate question detection). Asymmetric: different representations for queries vs documents — used in search/RAG where a short question needs to match a longer paragraph. `all-MiniLM-L6-v2` is symmetric — it handles this use case adequately for a local tool, though asymmetric models like Cohere v3 give better retrieval quality.

---

### 5.3 Chunking

**What is chunking and why is it needed?**
LLM context windows are limited. Sending an entire document on every query wastes tokens and dilutes relevance. Chunking splits the document into smaller segments that can be individually embedded and retrieved — only the most relevant ones are sent to the LLM.

**What chunk size does this project use?**
300 words per chunk with 50-word overlap. This is roughly 1–2 paragraphs — enough context per chunk without being so broad that relevance scores are diluted.

**What is overlap?**
Consecutive chunks share words at their boundaries. Without overlap, a sentence that straddles a boundary gets split — neither chunk captures its full meaning. With 50-word overlap, every sentence is fully contained in at least one chunk.

**What chunking strategy is used?**
Word-count based chunking — simple and effective for prose documents like resumes. More advanced strategies include sentence-boundary chunking, semantic chunking (split when topic changes), and recursive character splitting.

---

### 5.4 Vector Stores and FAISS

**What is a vector store?**
A data structure that stores embedding vectors and supports efficient k-nearest-neighbour (kNN) search — given a query vector, find the k most similar vectors.

**What is FAISS?**
Facebook AI Similarity Search — an open-source library for efficient similarity search over dense vectors. It stores vectors in memory and supports multiple index types.

**What is IndexFlatL2?**
The simplest FAISS index. Brute-force search using L2 (Euclidean) distance — compares the query vector against every stored vector. Exact (not approximate), fast for small collections. A typical resume produces 10–30 chunks — IndexFlatL2 is ideal.

**Why does FAISS require float32?**
FAISS is written in C++ and uses SIMD CPU instructions optimised for 32-bit floats. Python's default float is 64-bit. Always call `.astype("float32")` before adding to or searching a FAISS index.

**When would you use a different FAISS index?**
For millions of vectors, brute-force becomes slow. IndexIVFFlat partitions vectors into clusters and searches only nearby ones. IndexHNSWFlat uses a graph structure for approximate search. Both trade a small accuracy loss for much faster retrieval at scale.

---

### 5.5 Retrieval-Augmented Generation (RAG)

**What is RAG?**
A pattern for making LLMs answer questions about specific data they were not trained on. Two steps: retrieval (find relevant document chunks) and augmented generation (use those chunks as context to generate an answer).

**Why RAG instead of fine-tuning?**
Fine-tuning bakes knowledge into model weights — expensive (GPU hours), static (must retrain when data changes), risks catastrophic forgetting. RAG is dynamic (swap the document without touching the model), cheap (no training), and transparent (you can inspect exactly what context was used).

**Why RAG instead of stuffing the whole document into the context?**
For small resumes you could stuff everything in. But RAG is better practice because: it scales to large documents, it is cheaper (fewer tokens per query), and LLMs attend less precisely to relevant parts of very long contexts (the "lost in the middle" problem).

**What is the full RAG pipeline in this project?**
1. Greeting check — canned response for greetings, no LLM needed
2. Guardrail — block empty or non-alphabetic input
3. Query rewriting — rewrite vague follow-ups into specific questions
4. Retrieval — embed query, search FAISS, get top-4 chunks (top-6 for follow-ups)
5. Context injection — format chunks, add "do not repeat" instruction if follow-up
6. Prompt construction — system prompt + history + augmented message
7. LLM generation — Groq generates answer from context
8. History update — append both turns, update Streamlit session_state

**What is the "lost in the middle" problem?**
LLMs are better at using information at the start and end of their context than in the middle. Retrieving fewer, highly relevant chunks (4–6) gives better answers than retrieving 20 chunks and stuffing them all in.

**What is query rewriting?**
Vague follow-ups like "tell me more" have almost no semantic content — they retrieve the same chunks as the previous question. Query rewriting uses the LLM to turn the vague message into a specific standalone question using conversation history, giving FAISS better signal to retrieve fresh, relevant chunks.

**What are guardrails?**
Checks that prevent the LLM from behaving in undesired ways. This project uses two layers: a pre-LLM guardrail (check_query — blocks empty/nonsensical input before any API call) and a prompt-level guardrail (system prompt — instructs the LLM to refuse off-topic questions and stay within the document).

---

### 5.6 Streamlit and Session State

**What is Streamlit?**
A Python library that converts Python scripts into interactive web apps. Every widget interaction (button click, file upload, text input) causes Streamlit to rerun the entire script from top to bottom. No HTML, CSS, or JavaScript needed.

**What is session_state?**
A dictionary that persists across Streamlit reruns within a session. Without it, every interaction would reset all variables. This project stores the FAISS index, chunks, chat history, document name, and uploader key in session_state so they survive reruns.

**Why does the file uploader need a key?**
Streamlit widgets have internal state that persists independently of session_state. Setting `key=f"uploader_{counter}"` and incrementing the counter on reset forces Streamlit to treat it as a new widget — clearing the selected file. Without this, the uploader shows the old file even after reset.

**What is st.rerun()?**
Forces Streamlit to immediately re-execute the script from the top. Used after processing a document (to collapse the expander and show the chat) and after a starter question click (to render the new message immediately).

---

### 5.7 Document Parsing

**How does PDF parsing work?**
PyMuPDF (imported as `fitz`) opens a PDF from raw bytes, iterates over each page, and calls `page.get_text()` to extract the text layer. PDFs store text as positioned character streams — PyMuPDF reassembles these into readable strings.

**How does DOCX parsing work?**
`python-docx` opens a `.docx` file (which is a ZIP archive containing XML files) and provides access to paragraphs, tables, and styles. We extract `paragraph.text` from each paragraph and join them.

**Why is `.doc` not supported?**
Old `.doc` format (pre-2007) is a binary format that `python-docx` cannot read. Users are prompted to save as `.docx` first.

**What is the difference between text extraction and OCR?**
Text extraction (what this project does) reads the actual text data stored in the PDF. OCR (Optical Character Recognition) analyses pixel images of text and recognises characters visually. Scanned PDFs have no text layer — they are images — and require OCR. This project does not support scanned PDFs.

---

### 5.8 Testing with pytest

**What is pytest?**
Python's standard testing framework. Functions starting with `test_` are automatically discovered and run. Each test calls a piece of code and asserts the output matches expectations.

**What is a fixture?**
Reusable setup code decorated with `@pytest.fixture`. pytest injects it into any test that lists it as a parameter. `scope="module"` builds it once per file rather than once per test — important here because building a FAISS index takes a few seconds.

**What is mocking?**
Replacing real functions with fake ones that return predictable values. Used to avoid making real Groq API calls in tests — saving money, removing network dependency, and keeping tests fast. `unittest.mock.patch` temporarily replaces the target during the test.

**What is side_effect in mocks?**
Instead of returning a value, `side_effect` makes the mock raise an exception when called. Used in `test_chat_rate_limit_returns_friendly_message` to simulate Groq's RateLimitError without actually hitting the rate limit.

---

### 5.9 Common Interview Questions

**Q: What is the difference between semantic search and keyword search?**
Keyword search matches exact words. Semantic search uses embeddings to find documents with similar meaning even if they use different words. "What programming languages does she know?" semantically matches "Python, SQL, TensorFlow" even though none appear in the query.

**Q: What is the difference between an embedding model and an LLM?**
An embedding model converts text to a fixed-size vector — output is numbers, not text. An LLM generates text token by token — output is natural language. Embedding models are smaller, faster, and cheaper. RAG needs both: embeddings for retrieval, LLM for generation.

**Q: How would you evaluate a RAG system?**
Key metrics: retrieval precision (are retrieved chunks relevant?), retrieval recall (are all relevant chunks found?), answer faithfulness (does the answer only use retrieved context?), answer relevance (does it actually address the question?). Tools like RAGAS automate this evaluation.

**Q: What are the main failure modes of RAG?**
Retrieval failure (wrong chunks found — wrong embedding model, bad chunk size, vague query). Context overflow (too many chunks exceed context window). Hallucination (LLM adds facts not in chunks — addressed by strict system prompt). Irrelevant retrieval (chunks match surface words but don't answer the question).

**Q: How would you handle a scanned PDF?**
Add an OCR step before text extraction using a library like `pytesseract` or a cloud service like AWS Textract. Detect if a PDF has no text layer (all pages return empty strings from `page.get_text()`) and route to OCR automatically.

**Q: How would you scale this to support multiple users simultaneously?**
The current architecture is single-user (one FAISS index in memory per session). For multi-user: move the vector store to a managed service (Pinecone, Weaviate), use a database to store session state, and deploy the backend as a REST API (FastAPI) with horizontal scaling.

**Q: Why use sentence-transformers locally instead of an API?**
For this use case: no API cost, no rate limits, no latency, and the model is small enough (80MB) to run on CPU. The tradeoff is ~5–10 seconds startup time to load the model and slightly lower retrieval quality than larger API-based models.

---

## 6. Owner's Guide

This section contains everything you need to maintain, extend, and debug this project.

---

### 6.1 Project structure

```
Resume_Bot/
├── app/
│   ├── __init__.py
│   ├── parser.py          — PDF and DOCX text extraction
│   ├── scraper.py         — paste text cleaning
│   ├── vector_store.py    — chunking, sentence-transformers, FAISS
│   ├── rag.py             — RAG pipeline, greeting handler, query rewriter
│   └── guardrails.py      — input validation
├── tests/
│   ├── __init__.py
│   ├── test_parser.py     — tests for file parsing
│   ├── test_scraper.py    — tests for URL scraping and text cleaning
│   ├── test_vector_store.py — tests for chunking, embedding, search
│   └── test_rag.py        — tests for RAG pipeline and guardrails
├── streamlit_app.py       — main Streamlit UI
├── requirements.txt       — Python dependencies
└── .env                   — local secrets (never committed)
```

---

### 6.2 How to run the app

```bash
# Activate virtual environment (always do this first)
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Run the app
streamlit run streamlit_app.py

# Run tests
pytest tests/ -v
```

---

### 6.3 How to add a new file input format

Currently supports PDF and DOCX. To add a new format (e.g. `.txt`):

1. Open `app/parser.py`
2. Add a new function `extract_text_from_txt(file_bytes: bytes) -> str`
3. Add an `elif filename.endswith(".txt"):` branch in `extract_text()`
4. Add `"txt"` to the `type=["pdf", "docx", "txt"]` list in `streamlit_app.py`
5. Add a test in `tests/test_parser.py`

---

### 6.4 How to change the LLM model

Find this line in `app/rag.py`:

```python
MODEL = "llama-3.3-70b-versatile"
```

Replace with any model available on Groq. Current options at console.groq.com include `llama-3.1-8b-instant` (faster, smaller), `mixtral-8x7b-32768` (large context), and others. When OpenAI access is available, change the client initialisation and base URL — the rest of the code is identical since Groq's API is OpenAI-compatible.

---

### 6.5 How to swap in OpenAI when you have a key

`app/rag.py` uses Groq's SDK which is OpenAI-compatible. To swap:

```python
# Change the import
from openai import OpenAI, RateLimitError, APIError

# Change get_client()
def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client

# Change the model
MODEL = "gpt-4o"
```

Add `OPENAI_API_KEY=your_key` to `.env`. Everything else stays the same.

---

### 6.6 How to adjust chunking settings

Find these values in `app/vector_store.py`:

```python
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50)
```

- Increase `chunk_size` if answers feel incomplete or miss surrounding context
- Decrease `chunk_size` if answers pull in irrelevant content
- Increase `overlap` if important sentences keep getting split at boundaries
- A good rule of thumb: `overlap` should be 15–20% of `chunk_size`

---

### 6.7 How to adjust retrieval count

Find this in `app/rag.py`:

```python
is_followup = retrieval_query != user_message
top_k = 6 if is_followup else 4
relevant_chunks = search(retrieval_query, faiss_index, chunks, top_k=top_k)
```

Increase `top_k` if answers feel incomplete. Decrease if answers include too much irrelevant content. The tradeoff: more chunks = more context but higher token cost and potential "lost in the middle" degradation.

---

### 6.8 How to change the system prompt

The bot's behaviour is controlled by `SYSTEM_PROMPT_TEMPLATE` in `app/rag.py`. Edit the rules inside this string to change how the bot behaves — tone, restrictions, fallback messages. No index rebuild needed. Restart the Streamlit app to see changes.

---

### 6.9 How to add a new starter question

Find this list in `streamlit_app.py`:

```python
starters = [
    "What is this person's educational background?",
    "What are their key technical skills?",
    "Summarise their work experience.",
    "What makes them stand out as a candidate?",
]
```

Add or change entries. Keep to 4–6 questions — they are displayed in a 2-column grid.

---

### 6.10 Environment variables

| Variable | Where to get it | What it does |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com → API Keys | Authenticates with Groq LLM API |

Add to `.env` in the project root:
```
GROQ_API_KEY=your_key_here
```

Never commit `.env` to GitHub. Confirm `.env` is in `.gitignore`.

---

### 6.11 Troubleshooting common issues

**"Hi" returns "Please enter a valid question"**
The guardrail minimum length check is set to `< 2`. If this is happening, the old version of `guardrails.py` is still running. Confirm `check_query` has `len(query.strip()) < 2` not `< 3`.

**Starter question buttons don't work after clicking**
This is a Streamlit rerun timing issue. The `pending_question` pattern in `streamlit_app.py` handles this — confirm it is present and hasn't been accidentally removed.

**File uploader doesn't clear after Reset**
The `uploader_key` counter mechanism must be in place. Confirm `reset_session()` increments `st.session_state["uploader_key"]` and the uploader widget uses `key=f"uploader_{st.session_state.get('uploader_key', 0)}"`.

**Answers repeat on follow-up questions**
The query rewriter should handle this. Check the terminal for `[Query rewrite]` log lines confirming rewrites are happening. If not, the vague trigger words in `_rewrite_query` may not be matching — the message may be too long (trigger only fires for messages ≤ 8 words).

**Rate limit error visible to user**
Confirm `RateLimitError` is imported from `groq` and caught in the `try/except` block in `app/rag.py`.

**Tests take over 60 seconds**
Normal — the `sentence-transformers` model (`all-MiniLM-L6-v2`, ~80MB) loads once per test session. Subsequent test runs are faster if the model is cached. The `scope="module"` fixture ensures the FAISS index is built only once per test file.

**ImportError: cannot import name 'is_query_relevant'**
The old `test_rag.py` is still on disk. Replace it with the current version — the new tests do not import `is_query_relevant` since that function was removed from `guardrails.py`.