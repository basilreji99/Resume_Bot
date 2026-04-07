import streamlit as st
from app.parser import extract_text

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Bot",
    page_icon="📄",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("📄 Resume Bot")
st.caption("Upload a resume or profile document and ask questions about it.")

# ── File uploader ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload a PDF or DOCX file",
    type=["pdf", "docx"],
    help="Only PDF and DOCX formats are supported."
)

# ── Process the file once uploaded ───────────────────────────
if uploaded_file is not None:
    try:
        with st.spinner("Reading your file..."):
            raw_text = extract_text(uploaded_file)

        st.success(f"File read successfully: {uploaded_file.name}")

        # Show a preview of the extracted text (first 1000 characters)
        # This is a debug view — we'll remove it in a later phase
        with st.expander("Preview extracted text (debug view)"):
            st.text(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))

        # Store the text in session_state so other parts of the app can use it
        # session_state is Streamlit's way of remembering things across reruns
        st.session_state["document_text"] = raw_text
        st.session_state["document_name"] = uploaded_file.name

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong while reading the file: {e}")

else:
    # Remind the user what to do if nothing is uploaded yet
    st.info("Please upload a file to get started.")