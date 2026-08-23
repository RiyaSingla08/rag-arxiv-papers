"""
Fetches papers from arXiv on a given topic, downloads their PDFs, and
extracts raw text from each one. This builds the raw knowledge base that
gets chunked and embedded in later steps.

Run from the project root:
    python src/ingest.py

Requires internet access to arxiv.org (the arXiv API).
"""

import json
from pathlib import Path

import arxiv
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ------------------------------------------------------------------
# What to search for. Change this to build a RAG system on a different
# topic later -- everything downstream just works off whatever text
# ends up in data/processed/.
# ------------------------------------------------------------------
SEARCH_QUERY = "retrieval augmented generation large language models"
MAX_PAPERS = 15


def fetch_papers(query: str, max_results: int):
    """Search arXiv and return a list of arxiv.Result objects."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def download_pdf(result: arxiv.Result, dest_dir: Path) -> Path:
    """Download one paper's PDF. Returns the path it was saved to."""
    # arxiv IDs look like "2005.11401v2" -- strip version suffix for a clean filename
    safe_id = result.get_short_id().split("v")[0]
    filename = f"{safe_id}.pdf"
    filepath = dest_dir / filename
    if not filepath.exists():
        result.download_pdf(dirpath=str(dest_dir), filename=filename)
    return filepath


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file, page by page."""
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def main():
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Searching arXiv for: '{SEARCH_QUERY}' (top {MAX_PAPERS} results)...")
    papers = fetch_papers(SEARCH_QUERY, MAX_PAPERS)
    print(f"Found {len(papers)} papers.")

    metadata = []

    for i, paper in enumerate(papers, start=1):
        short_id = paper.get_short_id().split("v")[0]
        print(f"[{i}/{len(papers)}] {short_id}: {paper.title[:70]}...")

        pdf_path = download_pdf(paper, RAW_PDF_DIR)

        try:
            text = extract_text(pdf_path)
        except Exception as e:
            print(f"  Failed to extract text: {e}")
            continue

        text_path = PROCESSED_DIR / f"{short_id}.txt"
        text_path.write_text(text, encoding="utf-8")

        metadata.append({
            "arxiv_id": short_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": paper.published.isoformat(),
            "abstract": paper.summary,
            "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
            "text_path": str(text_path.relative_to(PROJECT_ROOT)),
            "num_chars": len(text),
        })

    metadata_path = PROCESSED_DIR / "papers_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nDone. Extracted text from {len(metadata)} papers.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
