"""
Build a BM25 keyword retrieval index from the chunked corpus.

BM25 complements dense retrieval (BGE-M3) by catching exact keyword matches
that semantic embeddings sometimes miss. Stored as a pickle for fast reload.

Usage:
  python scripts/build_bm25.py                       # personal variant
  python scripts/build_bm25.py --variant=publishable # publishable variant
"""

import argparse
import json
import pickle
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    """Simple word-level tokenization. Lowercase, alphanumeric only."""
    text = text.lower()
    # Split on non-alphanumeric, keep word characters and apostrophes
    tokens = re.findall(r"[a-z0-9']+", text)
    # Drop very short tokens (single chars are usually noise)
    return [t for t in tokens if len(t) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["personal", "publishable"],
                        default="personal")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "corpus" / "processed"
    output_path = project_root / "retrieval" / f"bm25_{args.variant}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Building BM25 index for variant: {args.variant}")
    
    # Load all chunks for this variant
    print(f"\nLoading chunks...")
    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []
    
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    for jsonl_path in jsonl_files:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if args.variant not in chunk.get("variants", []):
                    continue
                chunk_ids.append(chunk["id"])
                chunk_texts.append(chunk["text"])
                chunk_metadata.append({
                    "source": chunk["source"],
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "license": chunk.get("license", ""),
                })
    
    print(f"  {len(chunk_ids):,} chunks loaded")
    
    # Tokenize
    print(f"\nTokenizing...")
    t0 = time.time()
    tokenized = [tokenize(text) for text in tqdm(chunk_texts, unit="chunk")]
    print(f"  Tokenization done in {time.time() - t0:.1f}s")
    
    # Build BM25 index
    print(f"\nBuilding BM25 index...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 built in {time.time() - t0:.1f}s")
    
    # Serialize
    print(f"\nSerializing to {output_path}...")
    payload = {
        "variant": args.variant,
        "chunk_ids": chunk_ids,
        "metadata": chunk_metadata,
        "bm25": bm25,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()