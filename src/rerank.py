"""
Adds a re-ranking step on top of Step 5's vector retrieval: fetch a larger
pool of candidates via fast embedding similarity, then use a slower but more
accurate cross-encoder to re-score and re-order them by true relevance.

Run from the project root:
    python src/rerank.py "your question here"

Or import rerank_retrieve() to use in generate.py.
"""

import sys

from sentence_transformers import CrossEncoder

from src.retrieve import retrieve

# A small, fast cross-encoder trained specifically for search relevance ranking.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank_retrieve(question: str, initial_k: int = 20, final_k: int = 5):
    """
    Two-stage retrieval:
      1. Fast embedding search pulls in `initial_k` candidates (wide net).
      2. Cross-encoder re-scores all of them directly against the question,
         and we keep the top `final_k` (precise cut).
    Returns the same chunk dict format as retrieve(), but sorted by the
    cross-encoder's relevance score instead of raw embedding distance.
    """
    candidates = retrieve(question, top_k=initial_k)

    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(question, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:final_k]


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/rerank.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    print("--- Before re-ranking (raw embedding similarity) ---")
    baseline = retrieve(question, top_k=5)
    for i, r in enumerate(baseline, start=1):
        print(f"{i}. [distance {r['distance']:.3f}] {r['paper_title'][:60]}")

    print("\n--- After re-ranking (cross-encoder relevance) ---")
    reranked = rerank_retrieve(question, initial_k=20, final_k=5)
    for i, r in enumerate(reranked, start=1):
        print(f"{i}. [score {r['rerank_score']:.3f}] {r['paper_title'][:60]}")


if __name__ == "__main__":
    main()
