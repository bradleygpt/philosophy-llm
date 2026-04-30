"""
Generate a manifest summarizing the chunked corpus.

Reads all corpus/processed/*.jsonl files and produces
corpus/manifests/chunks.json with per-source statistics
and overall totals.

Usage: python scripts/generate_chunks_manifest.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "corpus" / "processed"
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist")
        sys.exit(1)
    
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: no .jsonl files in {processed_dir}")
        sys.exit(1)
    
    sources_summary = []
    grand_total_chunks = 0
    grand_total_chars = 0
    grand_total_tokens = 0
    
    # For variant breakdown
    variant_counts = defaultdict(int)
    
    for jsonl_path in jsonl_files:
        source_name = jsonl_path.stem
        chunk_count = 0
        char_total = 0
        token_total = 0
        chunk_sizes = []
        unique_docs = set()
        license_str = ""
        variants = []
        
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk_count += 1
                char_total += chunk["char_count"]
                token_total += chunk["token_estimate"]
                chunk_sizes.append(chunk["char_count"])
                unique_docs.add(chunk["source_file"])
                if not license_str:
                    license_str = chunk.get("license", "")
                if not variants:
                    variants = chunk.get("variants", [])
                
                # Track variant assignment
                for v in chunk.get("variants", []):
                    variant_counts[v] += 1
        
        if chunk_count == 0:
            continue
        
        avg_chars = char_total / chunk_count
        median_chars = median(chunk_sizes) if chunk_sizes else 0
        
        sources_summary.append({
            "source": source_name,
            "license": license_str,
            "variants": variants,
            "documents": len(unique_docs),
            "chunks": chunk_count,
            "avg_chunks_per_doc": round(chunk_count / len(unique_docs), 1) if unique_docs else 0,
            "total_chars": char_total,
            "total_tokens_estimate": token_total,
            "avg_chars_per_chunk": round(avg_chars, 1),
            "median_chars_per_chunk": median_chars,
            "jsonl_path": str(jsonl_path.relative_to(project_root)),
            "jsonl_size_mb": round(jsonl_path.stat().st_size / (1024 * 1024), 2),
        })
        
        grand_total_chunks += chunk_count
        grand_total_chars += char_total
        grand_total_tokens += token_total
    
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_parameters": {
            "target_tokens": 500,
            "overlap_tokens": 50,
            "chars_per_token_estimate": 4,
        },
        "totals": {
            "sources": len(sources_summary),
            "documents": sum(s["documents"] for s in sources_summary),
            "chunks": grand_total_chunks,
            "total_chars": grand_total_chars,
            "total_tokens_estimate": grand_total_tokens,
        },
        "variant_chunk_counts": dict(variant_counts),
        "sources": sources_summary,
    }
    
    manifest_dir = project_root / "corpus" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "chunks.json"
    
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    print(f"Manifest written: {manifest_path}")
    print(f"\nTotals:")
    print(f"  Documents:          {manifest['totals']['documents']:,}")
    print(f"  Chunks:             {manifest['totals']['chunks']:,}")
    print(f"  Total chars:        {manifest['totals']['total_chars']:,}")
    print(f"  Total tokens (est): {manifest['totals']['total_tokens_estimate']:,}")
    print(f"\nVariant chunk counts:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant:15s} {count:>10,}")
    print(f"\nPer source:")
    for s in sources_summary:
        print(f"  {s['source']:12s} {s['chunks']:>8,} chunks "
              f"({s['avg_chars_per_chunk']:.0f} avg chars, "
              f"{s['variants']})")


if __name__ == "__main__":
    main()