"""
Run this script once to build the FAISS index from your profile data files.
Run it again whenever you update any file in the data/ folder.

Usage:
    python scripts/build_index.py

Outputs (saved to data/):
    basil_index.faiss   — the FAISS vector index
    basil_chunks.pkl    — the original text chunks (parallel to the index)
"""

import os
import sys
import pickle
import faiss
from pathlib import Path

# Make sure app/ is importable from here
sys.path.append(str(Path(__file__).parent.parent))

from app.vector_store import chunk_text, embed_chunks, build_faiss_index

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "basil_index.faiss"
CHUNKS_PATH = DATA_DIR / "basil_chunks.pkl"


def load_all_markdown_files(data_dir: Path) -> str:
    """
    Reads every .md and .txt file in the data/ folder and
    concatenates them into one big string.

    We add a clear separator between files so chunks don't
    accidentally straddle two unrelated topics.
    """
    all_text = []
    files_loaded = []

    for filepath in sorted(data_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8").strip()
        if text:
            # Prepend the filename as context so the LLM knows
            # which section a chunk came from
            section_header = f"\n\n=== Source: {filepath.name} ===\n\n"
            all_text.append(section_header + text)
            files_loaded.append(filepath.name)

    # Also pick up plain .txt files if any
    for filepath in sorted(data_dir.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8").strip()
        if text:
            section_header = f"\n\n=== Source: {filepath.name} ===\n\n"
            all_text.append(section_header + text)
            files_loaded.append(filepath.name)

    if not all_text:
        raise FileNotFoundError(
            f"No .md or .txt files found in {data_dir}. "
            "Add your profile files and try again."
        )

    print(f"\nFiles loaded ({len(files_loaded)}):")
    for f in files_loaded:
        print(f"  ✓ {f}")

    return "\n\n".join(all_text)


def main():
    print("=== Building Basil persona index ===")

    # Step 1: Load all profile files
    print("\nStep 1: Loading profile data...")
    full_text = load_all_markdown_files(DATA_DIR)
    word_count = len(full_text.split())
    print(f"Total words loaded: {word_count:,}")

    # Step 2: Chunk the text
    print("\nStep 2: Chunking text...")
    chunks = chunk_text(full_text, chunk_size=300, overlap=50)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Embed the chunks
    print("\nStep 3: Embedding chunks (this may take 30–60 seconds)...")
    embeddings = embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 4: Build FAISS index
    print("\nStep 4: Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"Vectors in index: {index.ntotal}")

    # Step 5: Save to disk
    print("\nStep 5: Saving to disk...")
    DATA_DIR.mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"  ✓ Index saved to: {INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  ✓ Chunks saved to: {CHUNKS_PATH}")

    # Step 6: Quick sanity check
    print("\nStep 6: Sanity check — searching for 'name'...")
    from app.vector_store import search
    results = search("What is his name?", index, chunks, top_k=2)
    print("Top result preview:")
    print(f"  → {results[0][:120]}...")

    print("\n✅ Index build complete. You can now run the persona bot.")
    print("   Next: python -m uvicorn api.persona_chat:app --reload")


if __name__ == "__main__":
    main()