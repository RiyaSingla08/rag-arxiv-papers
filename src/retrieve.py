"""
Semantic search over the paper chunks stored in Step 4's vector database.
Given a question, finds the most relevant chunks by meaning (not exact
keyword match).

Run from the project root:
    python src/retrieve.py "your question here"

Or import retrieve() from other scripts (Step 6 will do this).
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "arxiv_rag_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache so repeated calls to retrieve() (e.g. from the generation
# script in Step 6) don't reload the model or reconnect to the DB every time.
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def retrieve(question: str, top_k: int = 5):
    """
    Embed the question and return the top_k most similar chunks.
    Returns a list of dicts: {text, arxiv_id, paper_title, distance}
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "arxiv_id": results["metadatas"][0][i]["arxiv_id"],
            "paper_title": results["metadatas"][0][i]["paper_title"],
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return chunks


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    results = retrieve(question, top_k=5)

    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} (distance: {r['distance']:.3f}) ---")
        print(f"Paper: {r['paper_title'][:80]}")
        print(f"Text: {r['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()
