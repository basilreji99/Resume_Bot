import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from app.parser import extract_text
from app.scraper import clean_pasted_text
from app.vector_store import build_vector_store
from app.rag import chat

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Bot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Resume Bot")
st.caption("Upload a document, paste a URL, or paste text directly — then ask questions.")

# ── Helper: reset session for a fresh document ────────────────
def reset_session():
    for key in ["document_text", "document_name", "document_ready",
                "faiss_index", "chunks", "chat_history"]:
        st.session_state.pop(key, None)
    # Incrementing this key forces Streamlit to recreate the file
    # uploader widget from scratch, clearing the selected file.
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1

# ══════════════════════════════════════════════════════════════
# SECTION 1: Document input (always visible)
# ══════════════════════════════════════════════════════════════
with st.expander(
    "📂 Document source" + (" ✅" if st.session_state.get("document_ready") else ""),
    expanded=not st.session_state.get("document_ready", False)
):
    input_mode = st.radio(
        "How would you like to provide the information?",
        options=["Upload a file", "Paste text"],
        horizontal=True
    )

    st.divider()

    ready_to_submit = False

    if input_mode == "Upload a file":
        uploaded_file = st.file_uploader(
            "Upload a PDF or DOCX file", type=["pdf", "docx"],
            key=f"uploader_{st.session_state.get('uploader_key', 0)}"
        )
        if uploaded_file:
            st.write(f"📎 Selected: **{uploaded_file.name}**")
            ready_to_submit = True

    elif input_mode == "Paste text":
        paste_type = st.selectbox(
            "What is this text from?",
            ["LinkedIn profile (copied manually)", "Resume / CV",
             "Bio or about page", "GitHub bio", "Other"]
        )
        pasted_text = st.text_area(
            "Paste your text here",
            height=200,
            placeholder=(
                "Paste any profile text here — LinkedIn About section, "
                "resume content, website bio, GitHub profile, or a mix of these."
            )
        )
        if pasted_text.strip():
            st.caption(f"Word count: {len(pasted_text.split())}")
            ready_to_submit = True

    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("Analyse Document", disabled=not ready_to_submit, type="primary")
    with col2:
        if st.session_state.get("document_ready"):
            if st.button("🔄 Reset", help="Load a different document"):
                reset_session()
                st.rerun()

    # ── Processing ────────────────────────────────────────────
    if submit:
        try:
            with st.spinner("Reading document..."):
                if input_mode == "Upload a file":
                    raw_text = extract_text(uploaded_file)
                    source_label = uploaded_file.name
                elif input_mode == "Paste text":
                    raw_text = clean_pasted_text(pasted_text)
                    source_label = paste_type

            with st.spinner("Building vector index..."):
                index, chunks = build_vector_store(raw_text)

            st.session_state.update({
                "document_text": raw_text,
                "document_name": source_label,
                "document_ready": True,
                "faiss_index": index,
                "chunks": chunks,
                "chat_history": [],   # fresh history for each new document
            })
            st.success(f"✅ Loaded: **{source_label}** — {len(chunks)} chunks indexed.")
            st.rerun()  # collapse the expander and show the chat

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ══════════════════════════════════════════════════════════════
# SECTION 2: Chat interface (only shown when doc is ready)
# ══════════════════════════════════════════════════════════════
if st.session_state.get("document_ready"):

    st.divider()
    st.subheader(f"💬 Ask about: {st.session_state['document_name']}")

    # Initialise chat history if somehow not set
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Render conversation history ───────────────────────────
    # st.chat_message renders a styled bubble for each turn.
    # We replay the whole history on every rerun.
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Suggested starter questions ───────────────────────────
    if not st.session_state["chat_history"]:
        st.caption("Try asking:")
        cols = st.columns(2)
        starters = [
            "What is this person's educational background?",
            "What are their key technical skills?",
            "Summarise their work experience.",
            "What makes them stand out as a candidate?",
        ]
        for i, q in enumerate(starters):
            if cols[i % 2].button(q, key=f"starter_{i}"):
                # Treat a starter click exactly like a typed message
                st.session_state["pending_question"] = q
                st.rerun()

    # ── Handle pending question from starter buttons ──────────
    if "pending_question" in st.session_state:
        user_q = st.session_state.pop("pending_question")
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, updated_history = chat(
                    user_message=user_q,
                    chat_history=st.session_state["chat_history"],
                    faiss_index=st.session_state["faiss_index"],
                    chunks=st.session_state["chunks"],
                    source_name=st.session_state["document_name"],
                )
            st.markdown(reply)
        st.session_state["chat_history"] = updated_history
        st.rerun()

    # ── Main chat input ───────────────────────────────────────
    # st.chat_input renders the sticky input bar at the bottom.
    # It returns the typed text when the user presses Enter.
    user_input = st.chat_input("Ask a question about this document...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, updated_history = chat(
                    user_message=user_input,
                    chat_history=st.session_state["chat_history"],
                    faiss_index=st.session_state["faiss_index"],
                    chunks=st.session_state["chunks"],
                    source_name=st.session_state["document_name"],
                )
            st.markdown(reply)
        st.session_state["chat_history"] = updated_history

else:
    st.info("Upload or link a document above to start chatting.")