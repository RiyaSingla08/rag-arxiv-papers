"""
Evaluates retrieval quality against the hand-labeled test set in
eval/qa_test_set.json. Computes Hit Rate@k and Mean Reciprocal Rank (MRR),
comparing raw embedding retrieval against cross-encoder re-ranked retrieval.

Run from the project root:
    python -m src.evaluate

This is what turns "the RAG system seems to work" into a real, defensible
claim about retrieval quality -- the metrics here are genuine evidence,
not vibes.
"""

import json
from pathlib import Path

from src.retrieve import retrieve
from src.rerank import rerank_retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = PROJECT_ROOT / "eval" / "qa_test_set.json"
RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"

TOP_K = 10  # how many chunks to retrieve/consider per question


def find_rank(chunks: list, expected_arxiv_id: str):
    """
    Returns the 1-indexed rank of the first chunk belonging to the expected
    paper, or None if that paper doesn't appear anywhere in the results.
    """
    for i, chunk in enumerate(chunks, start=1):
        if chunk["arxiv_id"] == expected_arxiv_id:
            return i
    return None


def evaluate_method(test_set: list, retrieval_fn, method_name: str):
    """
    Runs every test question through retrieval_fn and computes:
      - Hit Rate@5 and Hit Rate@10 (did the correct paper appear at all?)
      - MRR (Mean Reciprocal Rank -- rewards ranking the correct paper HIGHER,
        not just "somewhere in the top 10")
    """
    per_question_results = []

    for item in test_set:
        question = item["question"]
        expected = item["expected_arxiv_id"]

        chunks = retrieval_fn(question)
        rank = find_rank(chunks, expected)

        per_question_results.append({
            "question": question,
            "expected_arxiv_id": expected,
            "rank": rank,
            "reciprocal_rank": (1 / rank) if rank else 0,
            "hit_at_5": rank is not None and rank <= 5,
            "hit_at_10": rank is not None and rank <= 10,
        })

    n = len(per_question_results)
    hit_rate_5 = sum(r["hit_at_5"] for r in per_question_results) / n
    hit_rate_10 = sum(r["hit_at_10"] for r in per_question_results) / n
    mrr = sum(r["reciprocal_rank"] for r in per_question_results) / n

    return {
        "method": method_name,
        "hit_rate_at_5": hit_rate_5,
        "hit_rate_at_10": hit_rate_10,
        "mrr": mrr,
        "per_question": per_question_results,
    }


def main():
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    print(f"Evaluating retrieval quality on {len(test_set)} hand-labeled questions...\n")

    raw_results = evaluate_method(
        test_set,
        retrieval_fn=lambda q: retrieve(q, top_k=TOP_K),
        method_name="raw_embedding_retrieval",
    )

    reranked_results = evaluate_method(
        test_set,
        retrieval_fn=lambda q: rerank_retrieve(q, initial_k=20, final_k=TOP_K),
        method_name="cross_encoder_reranked",
    )

    print("=" * 60)
    print("SUMMARY: Raw retrieval vs. Re-ranked retrieval")
    print("=" * 60)
    print(f"{'Metric':<20}{'Raw retrieval':<20}{'Re-ranked':<20}")
    print(f"{'Hit Rate@5':<20}{raw_results['hit_rate_at_5']:<20.2%}{reranked_results['hit_rate_at_5']:<20.2%}")
    print(f"{'Hit Rate@10':<20}{raw_results['hit_rate_at_10']:<20.2%}{reranked_results['hit_rate_at_10']:<20.2%}")
    print(f"{'MRR':<20}{raw_results['mrr']:<20.3f}{reranked_results['mrr']:<20.3f}")

    print()
    print("=" * 60)
    print("PER-QUESTION BREAKDOWN (raw retrieval rank -> re-ranked rank)")
    print("=" * 60)
    for raw_q, rerank_q in zip(raw_results["per_question"], reranked_results["per_question"]):
        raw_rank = raw_q["rank"] if raw_q["rank"] else "not found"
        rerank_rank = rerank_q["rank"] if rerank_q["rank"] else "not found"
        flag = "  <-- changed" if raw_rank != rerank_rank else ""
        print(f"{raw_q['question'][:55]:<57} {str(raw_rank):>10} -> {str(rerank_rank):<10}{flag}")

    RESULTS_PATH.write_text(
        json.dumps({"raw": raw_results, "reranked": reranked_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
