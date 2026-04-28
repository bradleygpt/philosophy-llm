"""
Generate a manifest of downloaded corpus files.

Reads all files in corpus/raw/<source>/ and produces
corpus/manifests/<source>.json with metadata about each file.

Run: python scripts/generate_manifest.py <source_name>
Example: python scripts/generate_manifest.py gutenberg
"""

import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    """SHA-256 hash of file content. Used to detect changes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(source_name: str) -> dict:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "corpus" / "raw" / source_name
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        sys.exit(1)
    
    files = []
    total_bytes = 0
    
    for file_path in sorted(raw_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        
        size_bytes = file_path.stat().st_size
        total_bytes += size_bytes
        
        files.append({
            "filename": file_path.name,
            "size_bytes": size_bytes,
            "size_chars_approx": size_bytes,  # close enough for ASCII-heavy text
            "sha256": file_hash(file_path),
        })
    
    manifest = {
        "source": source_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "files": files,
    }
    
    return manifest


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_manifest.py <source_name>")
        print("Example: python scripts/generate_manifest.py gutenberg")
        sys.exit(1)
    
    source_name = sys.argv[1]
    project_root = Path(__file__).resolve().parent.parent
    
    manifest = generate_manifest(source_name)
    
    manifest_dir = project_root / "corpus" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{source_name}.json"
    
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )
    
    print(f"Manifest written: {manifest_path}")
    print(f"  Source: {manifest['source']}")
    print(f"  Files: {manifest['file_count']}")
    print(f"  Total: {manifest['total_mb']} MB")


if __name__ == "__main__":
    main()