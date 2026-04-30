"""
Chunk the raw corpus into RAG-ready pieces with metadata.

For each source in corpus/sources.yaml, reads raw text files,
strips source-specific boilerplate, splits into ~500-token chunks
with 50-token overlap, and writes JSONL output to corpus/processed/.

Each output line is a JSON object representing one chunk:
{
  "id": "<source>:<file_stem>:<chunk_index>",
  "text": "...",
  "source": "gutenberg",
  "title": "The Republic",
  "url": "https://...",
  "license": "public-domain",
  "variants": ["personal", "publishable"],
  "chunk_index": 0,
  "total_chunks_in_doc": 142,
  "char_count": 1987,
  "token_estimate": 497
}

Usage:
  python scripts/chunk_corpus.py                # chunk all sources
  python scripts/chunk_corpus.py --source=iep   # chunk one source
  python scripts/chunk_corpus.py --sample       # chunk first 5 files per source (testing)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm


# Chunking parameters
TARGET_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
CHARS_PER_TOKEN_ESTIMATE = 4  # rough English heuristic
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE  # 2000
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE  # 200


def estimate_tokens(text: str) -> int:
    """Fast token-count estimate. Off by ~10-15% from real tokenizers, fine for chunking."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


# ============================================================================
# Source-specific cleaners
# ============================================================================

def clean_gutenberg(text: str) -> str:
    """Strip Project Gutenberg boilerplate, transcriber notes, and pre-content matter."""
    # Strip everything before the START marker
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    
    # Strip everything from the END marker onward
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    
    # Strip transcriber/producer credits (commonly between START marker and actual book)
    text = re.sub(
        r"(?:Produced by|Transcribed by|This e[Bb]ook was prepared by)[^\n]*(?:\n[^\n]*){0,15}?(?=\n\n)",
        "",
        text,
        count=1,
    )
    
    # Strip transcriber's note blocks
    text = re.sub(
        r"TRANSCRIBER['\u2019]?S?\s+NOTES?:[\s\S]*?(?=\n\n[A-Z][A-Z ]{3,}\n|\n\n\s*[Pp]reface|\n\n\s*[Cc]hapter|\n\n\s*[Bb]ook|\n\n\s*[A-Z][a-z]+\s+[A-Z]|\Z)",
        "",
        text,
        count=1,
    )
    
    # Strip publisher/printer blocks
    text = re.sub(
        r"(?:^|\n)\s*(?:LONDON|NEW YORK|PARIS|EDINBURGH|BOSTON|OXFORD|CAMBRIDGE)[:\s].*?(?:PRINTED BY|PRINTED AT|PUBLISHERS?)[^\n]*(?:\n[^\n]*){0,5}?(?=\n\n)",
        "\n\n",
        text,
        flags=re.IGNORECASE,
        count=1,
    )
    
    return text.strip()


def clean_arxiv(text: str) -> str:
    """Strip arXiv metadata header and references section."""
    # Strip our metadata header (we re-attach it via chunk metadata)
    if text.startswith("[ARXIV PAPER]"):
        full_text_marker = re.search(r"\[FULL TEXT\]\s*\n", text)
        if full_text_marker:
            text = text[full_text_marker.end():]
    
    # Strip References section if found (case-insensitive, common formats)
    refs_match = re.search(
        r"\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n",
        text
    )
    if refs_match:
        text = text[:refs_match.start()]
    
    return text.strip()


def clean_iep(text: str) -> str:
    """Strip IEP metadata header and end-of-entry sections."""
    if text.startswith("[IEP ENTRY]"):
        full_text_marker = re.search(r"\[FULL TEXT\]\s*\n", text)
        if full_text_marker:
            text = text[full_text_marker.end():]
    
    # IEP entries end with sections like "References and Further Reading", "Author Information"
    for marker in [
        "References and Further Reading",
        "Author Information",
        "References",
    ]:
        idx = text.find(marker)
        if idx > len(text) // 2:  # only strip if it's in the latter half (true ending)
            text = text[:idx]
            break
    
    return text.strip()


def clean_sep(text: str) -> str:
    """Strip SEP metadata header and end-of-entry sections."""
    if text.startswith("[SEP ENTRY]"):
        full_text_marker = re.search(r"\[FULL TEXT\]\s*\n", text)
        if full_text_marker:
            text = text[full_text_marker.end():]
    
    # SEP entries end with: Bibliography, Academic Tools, Other Internet Resources, Related Entries
    for marker in [
        "Bibliography",
        "Academic Tools",
        "Other Internet Resources",
        "Related Entries",
    ]:
        idx = text.find(marker)
        if idx > len(text) // 2:
            text = text[:idx]
            break
    
    return text.strip()


CLEANERS = {
    "gutenberg": clean_gutenberg,
    "arxiv": clean_arxiv,
    "iep": clean_iep,
    "sep": clean_sep,
}


# ============================================================================
# Metadata extraction
# ============================================================================

def extract_header_metadata(text: str, source: str) -> dict:
    """Extract title, url, etc. from the source-specific metadata header."""
    metadata = {"title": "", "url": "", "author": "", "pub_date": ""}
    
    if source == "arxiv" and text.startswith("[ARXIV PAPER]"):
        m = re.search(r"Title:\s*(.+)", text)
        if m: metadata["title"] = m.group(1).strip()
        m = re.search(r"Authors:\s*(.+)", text)
        if m: metadata["author"] = m.group(1).strip()
        m = re.search(r"ArXiv ID:\s*(.+)", text)
        if m: metadata["url"] = m.group(1).strip()
        m = re.search(r"Published:\s*(.+)", text)
        if m: metadata["pub_date"] = m.group(1).strip()
    
    elif source == "iep" and text.startswith("[IEP ENTRY]"):
        m = re.search(r"Title:\s*(.+)", text)
        if m: metadata["title"] = m.group(1).strip()
        m = re.search(r"URL:\s*(.+)", text)
        if m: metadata["url"] = m.group(1).strip()
    
    elif source == "sep" and text.startswith("[SEP ENTRY]"):
        m = re.search(r"Title:\s*(.+)", text)
        if m: metadata["title"] = m.group(1).strip()
        m = re.search(r"URL:\s*(.+)", text)
        if m: metadata["url"] = m.group(1).strip()
        m = re.search(r"Author:\s*(.+)", text)
        if m: metadata["author"] = m.group(1).strip()
        m = re.search(r"First published:\s*(.+)", text)
        if m: metadata["pub_date"] = m.group(1).strip()
    
    return metadata


def gutenberg_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from filename pattern like 'plato_republic.txt' (fallback)."""
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        author = parts[0].replace("_", " ").title()
        title = parts[1].replace("_", " ").title()
    else:
        author = ""
        title = stem.replace("_", " ").title()
    return {
        "title": title,
        "author": author,
        "url": "",
        "pub_date": "",
    }


def gutenberg_metadata_from_text(text: str, filename: str) -> dict:
    """Extract Gutenberg metadata from the file's actual header (Title:, Author:, etc.)."""
    meta = gutenberg_metadata_from_filename(filename)  # start with fallback
    
    # Look for "Title: ..." and "Author: ..." in the first 3000 chars
    head = text[:3000]
    
    title_match = re.search(r"^\s*Title:\s*(.+)", head, re.MULTILINE)
    if title_match:
        meta["title"] = title_match.group(1).strip()
    
    author_match = re.search(r"^\s*Author:\s*(.+)", head, re.MULTILINE)
    if author_match:
        meta["author"] = author_match.group(1).strip()
    
    return meta


# ============================================================================
# Chunking
# ============================================================================

def split_into_chunks(text: str) -> Iterator[str]:
    """
    Split text into overlapping chunks of approximately TARGET_CHUNK_CHARS.
    Prefers paragraph > sentence > word boundaries; never splits mid-word.
    """
    if not text:
        return
    
    if len(text) <= TARGET_CHUNK_CHARS:
        yield text
        return
    
    pos = 0
    text_len = len(text)
    
    while pos < text_len:
        end = pos + TARGET_CHUNK_CHARS
        
        if end >= text_len:
            chunk = text[pos:].strip()
            if chunk:
                yield chunk
            return
        
        # Find best split point in [pos + TARGET_CHUNK_CHARS - 200, pos + TARGET_CHUNK_CHARS + 200]
        search_start = max(pos + TARGET_CHUNK_CHARS - 200, pos + TARGET_CHUNK_CHARS // 2)
        search_end = min(end + 200, text_len)
        
        split_at = None
        
        # Prefer paragraph break (\n\n)
        para_split = text.rfind("\n\n", search_start, search_end)
        if para_split > pos:
            split_at = para_split + 2
        
        # Else sentence break
        if split_at is None:
            for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                sent_split = text.rfind(punct, search_start, search_end)
                if sent_split > pos:
                    split_at = sent_split + len(punct)
                    break
        
        # Else word break (space)
        if split_at is None:
            word_split = text.rfind(" ", search_start, search_end)
            if word_split > pos:
                split_at = word_split + 1
        
        # Last resort: hard cut
        if split_at is None:
            split_at = end
        
        chunk = text[pos:split_at].strip()
        if chunk:
            yield chunk
        
        # Advance with overlap
        pos = max(split_at - OVERLAP_CHARS, pos + 1)


# ============================================================================
# Per-source processing
# ============================================================================

def process_source(source_name: str, source_config: dict, project_root: Path,
                   sample_only: bool = False) -> dict:
    """Chunk all files for one source. Returns stats dict."""
    raw_dir = project_root / "corpus" / "raw" / source_name
    if not raw_dir.exists():
        print(f"  [skip] no raw dir for {source_name}")
        return {"files": 0, "chunks": 0, "skipped": True}
    
    output_dir = project_root / "corpus" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_name}.jsonl"
    
    cleaner = CLEANERS.get(source_name)
    if cleaner is None:
        print(f"  [warn] no cleaner for {source_name}, using identity")
        cleaner = lambda x: x.strip()
    
    files = sorted(raw_dir.glob("*.txt"))
    if sample_only:
        files = files[:5]
    
    license_str = source_config.get("license", "unknown")
    variants = source_config.get("variants", ["personal"])
    
    total_chunks = 0
    files_processed = 0
    files_skipped = 0
    
    with output_path.open("w", encoding="utf-8") as out:
        for file_path in tqdm(files, desc=f"  {source_name}", unit="file"):
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  [fail] {file_path.name}: {e}")
                files_skipped += 1
                continue
            
            # Get metadata
            if source_name == "gutenberg":
                meta = gutenberg_metadata_from_text(raw_text, file_path.name)
            else:
                meta = extract_header_metadata(raw_text, source_name)
            
            # Clean and chunk
            clean_text = cleaner(raw_text)
            if len(clean_text) < 200:
                files_skipped += 1
                continue
            
            chunks = list(split_into_chunks(clean_text))
            if not chunks:
                files_skipped += 1
                continue
            
            file_stem = file_path.stem
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "id": f"{source_name}:{file_stem}:{i:04d}",
                    "text": chunk_text,
                    "source": source_name,
                    "title": meta["title"],
                    "author": meta.get("author", ""),
                    "url": meta["url"],
                    "pub_date": meta.get("pub_date", ""),
                    "license": license_str,
                    "variants": variants,
                    "chunk_index": i,
                    "total_chunks_in_doc": len(chunks),
                    "char_count": len(chunk_text),
                    "token_estimate": estimate_tokens(chunk_text),
                    "source_file": file_path.name,
                }
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
            files_processed += 1
    
    return {
        "source": source_name,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "total_chunks": total_chunks,
        "output_path": str(output_path),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None,
                        help="Process only one source (gutenberg/arxiv/iep/sep)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample mode: chunk only first 5 files per source for testing")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    sources_yaml = project_root / "corpus" / "sources.yaml"
    
    if not sources_yaml.exists():
        print(f"Error: {sources_yaml} not found")
        sys.exit(1)
    
    with sources_yaml.open() as f:
        config = yaml.safe_load(f)
    
    sources = config.get("sources", {})
    
    if args.source:
        if args.source not in sources:
            print(f"Error: source '{args.source}' not in sources.yaml")
            sys.exit(1)
        sources = {args.source: sources[args.source]}
    
    print(f"Chunk parameters:")
    print(f"  target_tokens: {TARGET_CHUNK_TOKENS}")
    print(f"  overlap_tokens: {OVERLAP_TOKENS}")
    print(f"  chars_per_token (estimate): {CHARS_PER_TOKEN_ESTIMATE}")
    if args.sample:
        print(f"  SAMPLE MODE: 5 files per source\n")
    else:
        print()
    
    all_stats = []
    for source_name, source_config in sources.items():
        print(f"\n=== {source_name} ===")
        stats = process_source(source_name, source_config, project_root, sample_only=args.sample)
        all_stats.append(stats)
        if not stats.get("skipped"):
            print(f"  -> {stats['files_processed']} files processed, "
                  f"{stats['total_chunks']:,} chunks, "
                  f"{stats['files_skipped']} skipped")
            print(f"  -> {stats['output_path']}")
    
    # Summary
    print("\n=== Summary ===")
    total_chunks_all = sum(s.get("total_chunks", 0) for s in all_stats)
    total_files_all = sum(s.get("files_processed", 0) for s in all_stats)
    print(f"Total files processed: {total_files_all:,}")
    print(f"Total chunks produced: {total_chunks_all:,}")
    if total_files_all > 0:
        print(f"Average chunks per file: {total_chunks_all / total_files_all:.1f}")


if __name__ == "__main__":
    main()