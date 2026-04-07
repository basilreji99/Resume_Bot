from app.guardrails import is_query_relevant
import fitz, sys

# Load your actual resume
with open("basil.pdf", "rb") as f:
    doc = fitz.open(stream=f.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)

queries = [
    "What is his name?",
    "What is his education background?",
    "What are his skills?",
    "What is the capital of France?",
    "What is the boiling point of water?",
]

for q in queries:
    relevant, score = is_query_relevant(q, text)
    status = "PASS" if relevant else "BLOCK"
    print(f"[{status}] score={score:.3f}  {q}")