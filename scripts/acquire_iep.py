"""
Acquire entries from the Internet Encyclopedia of Philosophy (IEP).

Open-access encyclopedia with attribution.
Crawls A-Z entry index, fetches each entry, strips navigation/footer,
saves clean text with metadata.

Usage:
  python scripts/acquire_iep.py --test       # crawl just one entry to verify parsing
  python scripts/acquire_iep.py              # crawl all entries
  python scripts/acquire_iep.py --max=50     # crawl up to N entries (resumable)
"""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://iep.utm.edu/"
LETTER_INDEX_URLS = [f"https://iep.utm.edu/{c}/" for c in "abcdefghijklmnopqrstuvwxyz"]
USER_AGENT = "philosophy-llm/0.1 personal research project (respectful crawl, 3s delay)"
DELAY_SECONDS = 3.0
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
    """Fetch all 26 IEP letter index pages and return list of (title, url) tuples."""
    print(f"Fetching entry index from {len(LETTER_INDEX_URLS)} letter pages")
    
    entries = []
    seen_urls = set()
    
    for letter_url in LETTER_INDEX_URLS:
        letter = letter_url.rstrip("/").split("/")[-1]
        try:
            response = make_request(letter_url)
        except Exception as e:
            print(f"  [warn] failed to fetch index for letter '{letter}': {e}")
            continue
        
        soup = BeautifulSoup(response.content, "lxml")
        
        letter_count = 0
        for link in soup.find_all("a", href=True):
            href = link["href"]
            title = link.get_text(strip=True)
            
            if not href.startswith("https://iep.utm.edu/"):
                continue
            if not title or len(title) < 2:
                continue
            
            # Skip non-entry pages
            skip_patterns = ["category", "tag", "author", "about", "contact",
                           "submissions", "feed", "wp-content", "wp-login",
                           "/page/", "#", "/search"]
            if any(pat in href.lower() for pat in skip_patterns):
                continue
            
            # Must be a single-segment path: /entryname/
            path = href.replace("https://iep.utm.edu/", "").rstrip("/")
            if not path or "/" in path:
                continue
            
            # Skip the letter index pages themselves
            if len(path) == 1:
                continue
            
            if href not in seen_urls:
                seen_urls.add(href)
                entries.append((title, href))
                letter_count += 1
        
        print(f"  [{letter}] found {letter_count} entries")
        time.sleep(DELAY_SECONDS)  # be polite during index crawl too
    
    print(f"\nTotal unique entries found: {len(entries)}")
    return entries


def safe_filename(url: str) -> str:
    """Build a safe filename from an IEP URL."""
    slug = url.replace("https://iep.utm.edu/", "").rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    return f"iep_{slug}.txt"


def extract_entry_text(html: bytes, url: str, title: str) -> str:
    """Parse an IEP entry HTML, strip navigation/footer/sidebar, return clean text."""
    soup = BeautifulSoup(html, "lxml")
    
    # Try to find the main content area. WordPress sites typically use:
    # <main>, <article>, or <div class="entry-content">
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="entry-content")
        or soup.find("div", class_="post")
        or soup.find("div", id="content")
    )
    
    if content is None:
        # Fallback: use body but warn
        content = soup.find("body")
    
    if content is None:
        return ""
    
    # Remove navigation, scripts, styles, comments, sidebars, footers
    for tag in content.find_all([
        "script", "style", "nav", "header", "footer", "aside", 
        "noscript", "iframe", "form"
    ]):
        tag.decompose()
    
    # Remove elements with classes/ids that suggest navigation/sidebar/footer
    for selector in [
        {"class": re.compile(r"(nav|menu|sidebar|widget|footer|comment|share|related)", re.I)},
        {"id": re.compile(r"(nav|menu|sidebar|widget|footer|comment)", re.I)},
    ]:
        for tag in content.find_all(attrs=selector):
            tag.decompose()
    
    text = content.get_text(separator="\n", strip=True)
    
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    return text


def format_entry(title: str, url: str, body: str) -> str:
    """Combine metadata header with body."""
    return f"""[IEP ENTRY]
Title: {title}
Source: Internet Encyclopedia of Philosophy
URL: {url}
License: Open access with attribution

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
        body = extract_entry_text(response.content, url, title)
        
        if len(body) < 1000:
            print(f"  [warn] {filename} unexpectedly short ({len(body)} chars), skipping")
            return False
        
        formatted = format_entry(title, url, body)
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
    output_dir = project_root / "corpus" / "raw" / "iep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Polite delay: {DELAY_SECONDS} seconds between requests")
    
    entries = get_entry_urls()
    
    if args.test:
        # Pick first entry, fetch it, exit
        if not entries:
            print("No entries found in index — parsing problem.")
            return
        title, url = entries[0]
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
    
    print(f"\nCrawling {len(entries)} entries...\n")
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