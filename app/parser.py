import fitz  # PyMuPDF
from docx import Document
import os


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Opens a PDF from raw bytes and extracts all text page by page.
    fitz is the underlying library name for PyMuPDF.
    """
    text = ""
    # fitz.open can open from bytes using a stream
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Opens a .docx file from raw bytes and extracts paragraph text.
    Word docs are structured as paragraphs — we join them with newlines.
    """
    import io
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text(uploaded_file) -> str:
    """
    Main entry point. Receives a Streamlit UploadedFile object,
    detects its type by extension, and routes to the right parser.
    Returns the raw extracted text as a string.
    """
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename.endswith(".doc"):
        # Older .doc format is not supported by python-docx
        # We'll show a friendly message instead of crashing
        raise ValueError(
            "Old .doc format is not supported. Please save as .docx and re-upload."
        )
    else:
        raise ValueError(
            f"Unsupported file type: {uploaded_file.name}. Please upload a PDF or DOCX."
        )