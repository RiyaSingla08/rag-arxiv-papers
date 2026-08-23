"""
Generates embeddings for every chunk (from data/processed/chunks.json) and
stores them in a local, persistent Chroma vector database. This is what
makes semantic search (Step 5) possible.

Run from the project root:
    python src/embed.py

The first run will download the embedding model (~90MB, one-time,
requires internet access to huggingface.co).
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# A small, fast, general-purpose embedding model. Runs on CPU, no API key needed.
# Produces 384-dimensional vectors.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "arxiv_rag_chunks"
BATCH_SIZE = 64  # encode chunks in batches for speed, rather than one at a time


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (downloads on first run)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Start fresh each time this script runs, so re-running after regenerating
    # chunks doesn't leave stale/duplicate entries behind.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, nothing to delete
    collection = client.create_collection(name=COLLECTION_NAME)

    print(f"Embedding and storing {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [str(c["chunk_id"]) for c in batch]
        metadatas = [
            {
                "arxiv_id": c["arxiv_id"],
                "paper_title": c["paper_title"],
                "chunk_index_in_paper": c["chunk_index_in_paper"],
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"\nDone. {collection.count()} chunks embedded and stored.")
    print(f"Vector store saved to: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
