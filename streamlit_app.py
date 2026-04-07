import streamlit as st
from app.parser import extract_text
from app.scraper import scrape_url, clean_pasted_text

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Bot",
    page_icon="📄",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("📄 Resume Bot")
st.caption("Upload a document, paste a URL, or paste text directly to get started.")

st.divider()

# ── Input mode selector ───────────────────────────────────────
# st.radio renders a set of mutually exclusive options.
# The user picks ONE input mode at a time.
input_mode = st.radio(
    label="How would you like to provide the information?",
    options=["Upload a file", "Enter a URL", "Paste text"],
    horizontal=True
)

st.divider()

# ── Variables we'll populate depending on mode ────────────────
raw_text = None       # will hold extracted text from any source
source_label = None   # human-readable description of the source
ready_to_submit = False

# ══════════════════════════════════════════════════════════════
# MODE 1: File upload
# ══════════════════════════════════════════════════════════════
if input_mode == "Upload a file":
    uploaded_file = st.file_uploader(
        label="Upload a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Only PDF and DOCX formats are supported."
    )
    if uploaded_file is not None:
        st.write(f"📎 Selected: **{uploaded_file.name}**")
        st.caption("Confirm this is the correct file before clicking Analyse.")
        ready_to_submit = True

# ══════════════════════════════════════════════════════════════
# MODE 2: URL input
# ══════════════════════════════════════════════════════════════
elif input_mode == "Enter a URL":

    # Source type dropdown — helps us label the content correctly
    # and in future phases could trigger source-specific handling
    source_type = st.selectbox(
        label="What type of link is this?",
        options=[
            "Portfolio website",
            "GitHub profile",
            "Personal blog",
            "Company page",
            "Other"
        ]
    )
    url_input = st.text_input(
        label="Enter the URL",
        placeholder="https://example.com/about"
    )

    # LinkedIn-specific warning shown proactively
    if "linkedin.com" in url_input.lower():
        st.warning(
            "LinkedIn blocks automated access. "
            "Please use the **Paste text** option and paste your profile text manually."
        )
        ready_to_submit = False
    elif url_input.strip():
        st.write(f"🔗 URL: **{url_input}**  |  Type: **{source_type}**")
        ready_to_submit = True

# ══════════════════════════════════════════════════════════════
# MODE 3: Paste text
# ══════════════════════════════════════════════════════════════
elif input_mode == "Paste text":

    paste_type = st.selectbox(
        label="What is this text from?",
        options=[
            "LinkedIn profile (copied manually)",
            "Resume / CV",
            "Bio or about page",
            "Other"
        ]
    )
    pasted_text = st.text_area(
        label="Paste your text here",
        height=250,
        placeholder="Paste your resume, LinkedIn profile, bio, or any profile text here..."
    )
    if pasted_text.strip():
        word_count = len(pasted_text.split())
        st.caption(f"Word count: {word_count}")
        ready_to_submit = True

# ══════════════════════════════════════════════════════════════
# Shared submit button — disabled until input is ready
# ══════════════════════════════════════════════════════════════
st.divider()
submit = st.button(
    label="Analyse Document",
    disabled=not ready_to_submit,
    type="primary"
)

# ══════════════════════════════════════════════════════════════
# Processing — runs only on explicit submit click
# ══════════════════════════════════════════════════════════════
if submit:
    try:
        with st.spinner("Processing..."):

            if input_mode == "Upload a file":
                raw_text = extract_text(uploaded_file)
                source_label = uploaded_file.name

            elif input_mode == "Enter a URL":
                raw_text = scrape_url(url_input.strip())
                source_label = f"{source_type}: {url_input}"

            elif input_mode == "Paste text":
                raw_text = clean_pasted_text(pasted_text)
                source_label = paste_type

        # Store everything in session_state for use in later phases
        st.session_state["document_text"] = raw_text
        st.session_state["document_name"] = source_label
        st.session_state["document_ready"] = True

        st.success(f"✅ Ready! Source loaded: **{source_label}**")

        # Debug preview — will be removed in a later phase
        with st.expander("Preview extracted text (debug view)"):
            preview = raw_text[:1000] + ("..." if len(raw_text) > 1000 else "")
            st.text(preview)

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ── Persistent ready state across reruns ─────────────────────
elif st.session_state.get("document_ready"):
    st.success(f"✅ Loaded: **{st.session_state['document_name']}**")
    st.caption("Switch input mode and click Analyse again to load a different source.")

else:
    st.info("Choose an input method above and click Analyse to get started.")