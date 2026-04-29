"""
Acquire entries from the Stanford Encyclopedia of Philosophy (SEP).

Personal variant only — see docs/decisions/0001-corpus-strategy-and-distribution.md.
SEP terms of use explicitly permit crawling for indexing (subject to
reasonable network usage). We use a 5-second polite delay (more conservative
than IEP).

Entry index: https://plato.stanford.edu/contents.html
Each entry: https://plato.stanford.edu/entries/<slug>/

Usage:
  python scripts/acquire_sep.py --test       # crawl just one entry to verify parsing
  python scripts/acquire_sep.py              # crawl all entries (~2.5 hours)
  python scripts/acquire_sep.py --max=50     # crawl up to N entries (resumable)
"""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://plato.stanford.edu/"
TOC_URL = "https://plato.stanford.edu/contents.html"
USER_AGENT = "philosophy-llm/0.1 personal research project (respectful crawl, 5s delay)"
DELAY_SECONDS = 5.0
REQUEST_TIMEOUT = 30


def make_request(url: str) -> requests.Response:
    """Polite request with proper User-Agent and timeout."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def get_entry_urls() -> list[tuple[str, str]]:
    """Fetch the SEP table of contents and return list of (title, url) tuples."""
    print(f"Fetching entry index from {TOC_URL}")
    response = make_request(TOC_URL)
    soup = BeautifulSoup(response.content, "lxml")
    
    entries = []
    seen_urls = set()
    
    # SEP entry URLs follow the pattern /entries/<slug>/
    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(strip=True)
        
        # Resolve relative URLs
        full_url = urljoin(BASE_URL, href)
        
        # Filter: must be plato.stanford.edu, must be an /entries/ URL
        if not full_url.startswith("https://plato.stanford.edu/entries/"):
            continue
        if not title or len(title) < 2:
            continue
        
        # Skip URLs with anchors or query strings (not the canonical entry page)
        if "#" in full_url or "?" in full_url:
            full_url = full_url.split("#")[0].split("?")[0]
        
        # Strip trailing slash for consistency, then add it back
        full_url = full_url.rstrip("/") + "/"
        
        # Must be a single-segment entry path: /entries/SLUG/
        path_after_entries = full_url.replace("https://plato.stanford.edu/entries/", "")
        path_after_entries = path_after_entries.rstrip("/")
        if not path_after_entries or "/" in path_after_entries:
            continue
        
        if full_url not in seen_urls:
            seen_urls.add(full_url)
            entries.append((title, full_url))
    
    print(f"Found {len(entries)} unique entries")
    return entries


def safe_filename(url: str) -> str:
    """Build a safe filename from a SEP URL."""
    slug = url.replace("https://plato.stanford.edu/entries/", "").rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    return f"sep_{slug}.txt"


def extract_entry_text(html: bytes, url: str, title: str) -> tuple[str, str, str]:
    """
    Parse a SEP entry HTML and return (clean_text, author, pub_date).
    Strips navigation, footer, sidebar.
    """
    soup = BeautifulSoup(html, "lxml")
    
    # SEP main content is in <div id="article"> or <div id="main-text">
    content = (
        soup.find("div", id="article")
        or soup.find("div", id="main-text")
        or soup.find("article")
        or soup.find("main")
    )
    
    if content is None:
        # Fallback: use body
        content = soup.find("body")
    
    if content is None:
        return "", "", ""
    
    # Try to extract author and pub date metadata
    author = ""
    pub_date = ""
    
    # SEP has a "<div id="pubinfo">" with publication info
    pubinfo = soup.find("div", id="pubinfo")
    if pubinfo:
        pubinfo_text = pubinfo.get_text(separator=" ", strip=True)
        # Try to grab author from "Copyright © YYYY by Author Name <email>"
        m = re.search(r"Copyright\s+©?\s*\d{4}\s*by\s*([^<]+?)(?:<|$)", pubinfo_text)
        if m:
            author = m.group(1).strip()
        # Try to grab pub date — "First published Mon Jul 1, 2002"
        m = re.search(r"First published\s+(.*?)(?:;|$|substantive)", pubinfo_text)
        if m:
            pub_date = m.group(1).strip()
    
    # Remove navigation, scripts, styles, header, footer, sidebar
    for tag in content.find_all([
        "script", "style", "nav", "header", "footer", "aside",
        "noscript", "iframe", "form"
    ]):
        tag.decompose()
    
    # Remove SEP-specific navigation/footer elements
    for selector in [
        {"id": re.compile(r"(nav|menu|sidebar|footer|header|toc-link)", re.I)},
        {"class": re.compile(r"(nav|menu|sidebar|footer|header|toc-link)", re.I)},
    ]:
        for tag in content.find_all(attrs=selector):
            tag.decompose()
    
    # Also strip the academic-tools and pubinfo blocks (we extracted what we need)
    for div_id in ["academic-tools", "pubinfo", "related-entries"]:
        for tag in content.find_all("div", id=div_id):
            tag.decompose()
    
    text = content.get_text(separator="\n", strip=True)
    
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    return text, author, pub_date


def format_entry(title: str, url: str, body: str, author: str, pub_date: str) -> str:
    """Combine metadata header with body."""
    return f"""[SEP ENTRY]
Title: {title}
Source: Stanford Encyclopedia of Philosophy
URL: {url}
Author: {author or "see entry for full attribution"}
First published: {pub_date or "unknown"}
License: Stanford-restricted (personal variant only — see ADR-0001)

[FULL TEXT]
{body}
"""


def acquire_entry(title: str, url: str, output_dir: Path) -> bool:
    """Fetch one entry and save it. Returns True on success."""
    filename = safe_filename(url)
    output_path = output_dir / filename
    
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  [skip] {filename}")
        return True
    
    try:
        response = make_request(url)
        body, author, pub_date = extract_entry_text(response.content, url, title)
        
        if len(body) < 1000:
            print(f"  [warn] {filename} unexpectedly short ({len(body)} chars), skipping")
            return False
        
        formatted = format_entry(title, url, body, author, pub_date)
        output_path.write_text(formatted, encoding="utf-8")
        print(f"  [ok]   {filename} ({len(formatted):,} chars)")
        return True
        
    except Exception as e:
        print(f"  [fail] {filename} -- {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: fetch only one entry to verify parsing")
    parser.add_argument("--max", type=int, default=None,
                        help="Max entries to crawl (default: all)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "corpus" / "raw" / "sep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Polite delay: {DELAY_SECONDS} seconds between requests")
    
    entries = get_entry_urls()
    
    if args.test:
        if not entries:
            print("No entries found in TOC — parsing problem.")
            return
        # Pick a known well-formed entry to test against (Plato).
        # Fallback to first entry if not present.
        title, url = next(
            ((t, u) for t, u in entries if "plato" in u.lower()),
            entries[0]
        )
        print(f"\nTEST MODE: fetching one entry only")
        print(f"  Title: {title}")
        print(f"  URL: {url}\n")
        time.sleep(DELAY_SECONDS)
        success = acquire_entry(title, url, output_dir)
        if success:
            filename = safe_filename(url)
            print(f"\nTest succeeded. Open {output_dir / filename} to verify content quality.")
            print("If clean, run without --test to crawl all entries.")
        return
    
    if args.max:
        entries = entries[:args.max]
        print(f"Limited to first {args.max} entries\n")
    
    print(f"\nCrawling {len(entries)} entries (estimated time: {len(entries) * DELAY_SECONDS / 60:.0f} minutes)...\n")
    successes = 0
    failures = 0
    
    for i, (title, url) in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {title}")
        if acquire_entry(title, url, output_dir):
            successes += 1
        else:
            failures += 1
        time.sleep(DELAY_SECONDS)
    
    print(f"\nDone. {successes} succeeded, {failures} failed.")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()