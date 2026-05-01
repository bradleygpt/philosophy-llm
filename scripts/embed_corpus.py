"""
Embed the chunked corpus into a Chroma vector database.

Reads corpus/processed/*.jsonl, runs each chunk through BGE-M3,
and writes embeddings + metadata to a local Chroma database.

By default, builds the personal variant (all chunks).
Use --variant=publishable to build the publishable variant (excludes SEP).

Usage:
  python scripts/embed_corpus.py                       # personal variant (default)
  python scripts/embed_corpus.py --variant=publishable # publishable variant
  python scripts/embed_corpus.py --sample              # first 500 chunks for testing
  python scripts/embed_corpus.py --batch-size=128      # tune for your GPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = 128
MAX_SEQ_LENGTH = 1024  # our chunks are ~500 tokens, well below this
CHROMA_BATCH_SIZE = 1000


def load_chunks(processed_dir: Path, variant: str, sample_n: int | None = None):
    """Stream chunks from JSONL files, filtered by variant."""
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files in {processed_dir}")
    
    chunks = []
    for jsonl_path in jsonl_files:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if variant in chunk.get("variants", []):
                    chunks.append(chunk)
                    if sample_n and len(chunks) >= sample_n:
                        return chunks
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["personal", "publishable"],
                        default="personal",
                        help="Which variant to build (default: personal)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample mode: embed first 500 chunks for testing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--reset", action="store_true",
                        help="Reset the Chroma database (delete and rebuild)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable fp16 (use fp32; slower but more compatible)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "corpus" / "processed"
    chroma_dir = project_root / "retrieval" / f"index_{args.variant}"
    
    if args.reset and chroma_dir.exists():
        import shutil
        print(f"Resetting Chroma DB at {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU detection
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("WARNING: CUDA not available, falling back to CPU (will be slow)")
    
    use_fp16 = (device == "cuda") and (not args.no_fp16)
    print(f"Precision: {'fp16' if use_fp16 else 'fp32'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Load chunks
    sample_n = 500 if args.sample else None
    print(f"\nLoading chunks for variant '{args.variant}'...")
    chunks = load_chunks(processed_dir, args.variant, sample_n)
    print(f"  {len(chunks):,} chunks to embed")
    
    if args.sample:
        print("  SAMPLE MODE: limited to first 500")
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    # Set max sequence length BEFORE inference
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # Convert to fp16 for speed (Tensor Cores)
    if use_fp16:
        model = model.half()
    
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    
    # Initialize Chroma
    print(f"\nInitializing Chroma DB at {chroma_dir}")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    
    collection_name = f"philosophy_{args.variant}"
    try:
        collection = client.get_collection(name=collection_name)
        existing_count = collection.count()
        print(f"  Collection '{collection_name}' exists with {existing_count:,} chunks")
        if existing_count > 0:
            print(f"  Loading existing IDs to support resumption...")
            existing_ids = set(collection.get()["ids"])
        else:
            existing_ids = set()
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "variant": args.variant,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
            },
        )
        existing_ids = set()
        print(f"  Created new collection '{collection_name}'")
    
    chunks_to_embed = [c for c in chunks if c["id"] not in existing_ids]
    skipped = len(chunks) - len(chunks_to_embed)
    if skipped > 0:
        print(f"  Skipping {skipped:,} already-embedded chunks")
    print(f"  {len(chunks_to_embed):,} chunks to embed in this run\n")
    
    if not chunks_to_embed:
        print("Nothing to embed. Exiting.")
        return
    
    # Embed and write in batches
    t0 = time.time()
    pbar = tqdm(total=len(chunks_to_embed), desc="Embedding", unit="chunk", smoothing=0.1)
    
    write_buffer_ids = []
    write_buffer_embeddings = []
    write_buffer_documents = []
    write_buffer_metadatas = []
    
    for i in range(0, len(chunks_to_embed), args.batch_size):
        batch = chunks_to_embed[i:i + args.batch_size]
        texts = [c["text"] for c in batch]
        
        # Embed batch on GPU
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        # Buffer for Chroma write
        for chunk, emb in zip(batch, embeddings):
            write_buffer_ids.append(chunk["id"])
            write_buffer_embeddings.append(emb.tolist())
            write_buffer_documents.append(chunk["text"])
            write_buffer_metadatas.append({
                "source": chunk["source"],
                "title": chunk.get("title", ""),
                "author": chunk.get("author", ""),
                "url": chunk.get("url", ""),
                "license": chunk.get("license", ""),
                "chunk_index": chunk["chunk_index"],
                "total_chunks_in_doc": chunk["total_chunks_in_doc"],
                "source_file": chunk.get("source_file", ""),
                "char_count": chunk["char_count"],
            })
        
        # Flush to Chroma when buffer full
        if len(write_buffer_ids) >= CHROMA_BATCH_SIZE:
            collection.add(
                ids=write_buffer_ids,
                embeddings=write_buffer_embeddings,
                documents=write_buffer_documents,
                metadatas=write_buffer_metadatas,
            )
            write_buffer_ids.clear()
            write_buffer_embeddings.clear()
            write_buffer_documents.clear()
            write_buffer_metadatas.clear()
        
        pbar.update(len(batch))
    
    # Flush remaining
    if write_buffer_ids:
        collection.add(
            ids=write_buffer_ids,
            embeddings=write_buffer_embeddings,
            documents=write_buffer_documents,
            metadatas=write_buffer_metadatas,
        )
    
    pbar.close()
    
    elapsed = time.time() - t0
    rate = len(chunks_to_embed) / elapsed if elapsed > 0 else 0
    
    print(f"\nDone.")
    print(f"  Embedded: {len(chunks_to_embed):,} chunks in {elapsed:.1f}s ({rate:.1f} chunks/sec)")
    print(f"  Total in collection: {collection.count():,}")
    print(f"  Database: {chroma_dir}")


if __name__ == "__main__":
    main()