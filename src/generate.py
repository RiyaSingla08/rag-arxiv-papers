"""
Full RAG pipeline: retrieve relevant chunks (Step 5), then feed them to a
local LLM via Ollama to generate a grounded, cited answer.

Run from the project root:
    python src/generate.py "your question here"

Requires Ollama running locally with the model pulled:
    ollama pull llama3.2
"""

import sys

import ollama

from src.retrieve import retrieve

OLLAMA_MODEL = "llama3.2"

RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions about academic papers on Retrieval-Augmented Generation (RAG).

Answer the question using ONLY the context provided below. Do not use any outside knowledge.
If the context does not contain enough information to answer the question, say so explicitly rather than guessing.
When you use information from the context, mention which paper it came from.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['paper_title']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, top_k: int = 5) -> dict:
    """
    Full RAG call: retrieve relevant chunks, build a prompt, generate an answer.
    Returns a dict with the answer text and the sources used, so the caller
    can display citations alongside the generated answer.
    """
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = response["message"]["content"]

    sources = list({c["paper_title"] for c in chunks})  # de-duplicated paper titles

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "chunks_used": chunks,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")
    print("Retrieving relevant chunks and generating answer...\n")

    result = generate_answer(question)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])
    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)
    for source in result["sources"]:
        print(f"  - {source}")


if __name__ == "__main__":
    main()
