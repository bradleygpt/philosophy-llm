"""
Acquire philosophy-relevant papers from arXiv.

Pulls recent papers from selected arXiv categories, downloads PDFs,
extracts text, saves as cleaned .txt files with metadata headers.

Run: python scripts/acquire_arxiv.py
"""

import time
import json
import ssl
import certifi
from pathlib import Path
from datetime import datetime, timezone

import urllib.request
# Force urllib to use certifi's certificate bundle (fixes SSL on Windows)
ssl_context = ssl.create_default_context(cafile=certifi.where())
https_handler = urllib.request.HTTPSHandler(context=ssl_context)
opener = urllib.request.build_opener(https_handler)
urllib.request.install_opener(opener)

import arxiv
from pypdf import PdfReader


# Categories to pull from. Each entry: (category_id, max_papers, description)
CATEGORIES = [
    ("physics.hist-ph", 100, "History and philosophy of physics"),
    ("cs.CY", 100, "Computers and society (incl. AI ethics)"),
    ("cs.AI", 100, "AI (filtered for philosophical/alignment papers)"),
]

# Keywords used to filter cs.AI for philosophy-relevant papers.
# We don't want every AI paper — only those touching philosophy/alignment.
AI_PHILOSOPHY_KEYWORDS = [
    "alignment", "ethics", "moral", "philosophy", "consciousness",
    "agency", "value", "interpretab", "constitutional", "safety",
    "preference", "judgment", "explainab", "fair", "bias",
]


def is_philosophy_relevant_ai(paper) -> bool:
    """Check if a cs.AI paper is philosophy-adjacent based on title/abstract."""
    text = f"{paper.title} {paper.summary}".lower()
    return any(kw in text for kw in AI_PHILOSOPHY_KEYWORDS)


def safe_filename(paper) -> str:
    """Build a safe filename from paper id."""
    paper_id = paper.entry_id.split("/")[-1]
    # Clean up version suffix like 'v2'
    paper_id = paper_id.replace(".", "_")
    return f"arxiv_{paper_id}.txt"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file. Returns empty string on failure."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  [warn] PDF extraction failed: {e}")
        return ""


def format_paper_text(paper, body_text: str) -> str:
    """Combine metadata header with body text."""
    authors = ", ".join(str(a) for a in paper.authors)
    categories = ", ".join(paper.categories) if paper.categories else "n/a"
    
    header = f"""[ARXIV PAPER]
Title: {paper.title}
Authors: {authors}
ArXiv ID: {paper.entry_id}
Published: {paper.published.strftime("%Y-%m-%d") if paper.published else "n/a"}
Updated: {paper.updated.strftime("%Y-%m-%d") if paper.updated else "n/a"}
Categories: {categories}
URL: {paper.entry_id}

[ABSTRACT]
{paper.summary.strip()}

[FULL TEXT]
"""
    return header + body_text


def acquire_category(category: str, max_papers: int, description: str,
                     output_dir: Path, downloaded_ids: set) -> tuple[int, int]:
    """Acquire papers from a single category. Returns (successes, skips)."""
    print(f"\n=== {category}: {description} ===")
    print(f"Querying arXiv for up to {max_papers} most recent papers...")
    
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    
    successes = 0
    skips = 0
    filtered_out = 0
    
    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1].split("v")[0]
        
        if paper_id in downloaded_ids:
            skips += 1
            continue
        
        # Filter cs.AI for philosophy relevance
        if category == "cs.AI" and not is_philosophy_relevant_ai(paper):
            filtered_out += 1
            continue
        
        filename = safe_filename(paper)
        output_path = output_dir / filename
        
        if output_path.exists() and output_path.stat().st_size > 500:
            print(f"  [skip] {filename}")
            skips += 1
            downloaded_ids.add(paper_id)
            continue
        
        # Short title for log readability
        short_title = paper.title[:70].replace("\n", " ")
        print(f"  [get]  {paper_id}: {short_title}...")
        
        # Download PDF to a temp location
        try:
            pdf_filename = output_dir / f".tmp_{paper_id}.pdf"
            paper.download_pdf(dirpath=str(output_dir), filename=pdf_filename.name)
            
            body_text = extract_pdf_text(pdf_filename)
            
            if len(body_text) < 500:
                print(f"  [warn] PDF text suspiciously short ({len(body_text)} chars), skipping")
                pdf_filename.unlink(missing_ok=True)
                continue
            
            full_text = format_paper_text(paper, body_text)
            output_path.write_text(full_text, encoding="utf-8")
            
            # Clean up temp PDF
            pdf_filename.unlink(missing_ok=True)
            
            successes += 1
            downloaded_ids.add(paper_id)
            print(f"  [ok]   {filename} ({len(full_text):,} chars)")
            
            time.sleep(3.0)  # polite delay
            
        except Exception as e:
            print(f"  [fail] {paper_id} -- {e}")
            # Clean up any temp file
            try:
                (output_dir / f".tmp_{paper_id}.pdf").unlink(missing_ok=True)
            except Exception:
                pass
    
    print(f"\n{category} summary: {successes} new, {skips} skipped, {filtered_out} filtered out")
    return successes, skips


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "corpus" / "raw" / "arxiv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Track which paper ids we've seen (across categories — papers can be cross-listed)
    downloaded_ids = set()
    for existing in output_dir.glob("arxiv_*.txt"):
        # Extract id back from filename: arxiv_2401_12345.txt -> 2401.12345
        stem = existing.stem.replace("arxiv_", "")
        # Convert back: 2401_12345 -> 2401.12345
        parts = stem.split("_")
        if len(parts) >= 2:
            paper_id = ".".join(parts[:2])
            downloaded_ids.add(paper_id)
    
    print(f"Already-downloaded papers: {len(downloaded_ids)}")
    
    total_new = 0
    total_skipped = 0
    
    for category, max_papers, description in CATEGORIES:
        new, skipped = acquire_category(
            category, max_papers, description, output_dir, downloaded_ids
        )
        total_new += new
        total_skipped += skipped
    
    print(f"\n=== Done ===")
    print(f"Total new papers: {total_new}")
    print(f"Total skipped (already had): {total_skipped}")
    print(f"Total papers in corpus: {len(list(output_dir.glob('arxiv_*.txt')))}")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()