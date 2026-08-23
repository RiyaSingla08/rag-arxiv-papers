"""
Splits each extracted paper's text (from data/processed/*.txt) into
overlapping chunks, ready for embedding in Step 4.

Run from the project root:
    python src/chunk.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_SIZE = 700       # target characters per chunk
CHUNK_OVERLAP = 100    # characters shared between consecutive chunks


def clean_text(text: str) -> str:
    """
    PDF extraction often leaves messy whitespace (stray line breaks mid-sentence,
    multiple spaces from column layouts). Collapse that down before chunking,
    so chunk boundaries land on sensible points rather than mid-word artifacts.
    """
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace runs to single spaces
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping chunks. Tries to break on a sentence boundary
    near the target size rather than cutting mid-sentence, when possible.
    """
    text = clean_text(text)
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If we're not at the very end of the text, try to extend to the next
        # sentence-ending punctuation so we don't cut off mid-sentence.
        if end < text_len:
            next_period = text.find(". ", end)
            # Only extend if a sentence boundary is reasonably close (within 150 chars)
            if next_period != -1 and next_period - end < 150:
                end = next_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap  # step forward, but overlap with the previous chunk

    return chunks


def main():
    metadata_path = PROCESSED_DIR / "papers_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    all_chunks = []
    chunk_id_counter = 0

    for paper in metadata:
        text_path = PROJECT_ROOT / paper["text_path"]
        raw_text = text_path.read_text(encoding="utf-8")

        chunks = chunk_text(raw_text)
        print(f"{paper['arxiv_id']}: {len(raw_text):,} chars -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": chunk_id_counter,
                "arxiv_id": paper["arxiv_id"],
                "paper_title": paper["title"],
                "chunk_index_in_paper": i,
                "text": chunk,
                "num_chars": len(chunk),
            })
            chunk_id_counter += 1

    output_path = PROCESSED_DIR / "chunks.json"
    output_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
