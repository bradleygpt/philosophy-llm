"""
Hybrid retrieval module: dense (BGE-M3) + sparse (BM25) + RRF fusion + cross-encoder reranking.

Library usage:
    from retrieval.retrieve import HybridRetriever
    retriever = HybridRetriever(variant="personal")
    results = retriever.retrieve("what is the categorical imperative?", top_k=10)

CLI usage:
    python retrieval/retrieve.py --query "what is the categorical imperative?"
    python retrieval/retrieve.py --query "..." --no-rerank   # skip cross-encoder
    python retrieval/retrieve.py --query "..." --variant=publishable --top-k=20

Each result is a dict with:
    {
        "chunk_id": "sep:sep_kant-moral:0042",
        "text": "...",
        "score": 0.92,                  # rerank score if reranked, else RRF score
        "rrf_score": 0.0345,            # always present
        "rerank_score": 0.92 or None,
        "metadata": {"source": "sep", "title": "...", ...},
    }
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
DEFAULT_DENSE_TOP_K = 50
DEFAULT_SPARSE_TOP_K = 50
DEFAULT_RERANK_POOL = 50  # how many to send to the reranker
DEFAULT_FINAL_TOP_K = 10


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    return [t for t in tokens if len(t) > 1]


class HybridRetriever:
    def __init__(self, variant: str = "personal", verbose: bool = False,
                  use_reranker: bool = True):
        self.variant = variant
        self.verbose = verbose
        self.use_reranker = use_reranker
        
        chroma_dir = PROJECT_ROOT / "retrieval" / f"index_{variant}"
        bm25_path = PROJECT_ROOT / "retrieval" / f"bm25_{variant}.pkl"
        
        if not chroma_dir.exists():
            raise RuntimeError(f"Chroma index not found at {chroma_dir}. "
                             f"Run scripts/embed_corpus.py --variant={variant} first.")
        if not bm25_path.exists():
            raise RuntimeError(f"BM25 index not found at {bm25_path}. "
                             f"Run scripts/build_bm25.py --variant={variant} first.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load embedding model
        if verbose: print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        if device == "cuda":
            self.embedder = self.embedder.half()
        self.embedder.max_seq_length = 1024
        
        # Load reranker (if enabled)
        self.reranker = None
        if use_reranker:
            if verbose: print(f"Loading reranker ({RERANKER_MODEL_NAME})...")
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
            except Exception as e:
                print(f"  Reranker load failed: {e}")
                print(f"  Continuing without reranking.")
                self.reranker = None
        
        # Load Chroma
        if verbose: print(f"Loading Chroma index from {chroma_dir}...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(f"philosophy_{variant}")
        if verbose: print(f"  Chroma: {self.collection.count():,} chunks indexed")
        
        # Load BM25
        if verbose: print(f"Loading BM25 index from {bm25_path}...")
        with bm25_path.open("rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.bm25_chunk_ids = payload["chunk_ids"]
        self.bm25_metadata = payload["metadata"]
        self.bm25_id_to_idx = {cid: i for i, cid in enumerate(self.bm25_chunk_ids)}
        if verbose: print(f"  BM25: {len(self.bm25_chunk_ids):,} chunks indexed")
    
    def _dense_retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        with torch.no_grad():
            query_emb = self.embedder.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k,
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        scores = [1.0 - d for d in distances]
        return list(zip(ids, scores))
    
    def _sparse_retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i],
                            reverse=True)[:top_k]
        return [(self.bm25_chunk_ids[i], float(bm25_scores[i]))
                for i in top_indices if bm25_scores[i] > 0]
    
    def _reciprocal_rank_fusion(self, dense_results, sparse_results) -> dict[str, float]:
        rrf_scores: dict[str, float] = {}
        for rank, (chunk_id, _score) in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (chunk_id, _score) in enumerate(sparse_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return rrf_scores
    
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score candidates using the cross-encoder."""
        if not self.reranker or not candidates:
            return candidates
        
        pairs = [(query, c["text"]) for c in candidates]
        with torch.no_grad():
            scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Attach rerank scores
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        
        # Sort by rerank score, descending
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates
    
    def retrieve(self, query: str,
                  top_k: int = DEFAULT_FINAL_TOP_K,
                  dense_top_k: int = DEFAULT_DENSE_TOP_K,
                  sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
                  rerank_pool: int = DEFAULT_RERANK_POOL) -> list[dict]:
        """Run hybrid retrieval with optional reranking."""
        dense_results = self._dense_retrieve(query, dense_top_k)
        sparse_results = self._sparse_retrieve(query, sparse_top_k)
        rrf_scores = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Take top rerank_pool by RRF before reranking
        candidates_pool = rerank_pool if self.reranker else top_k
        sorted_ids = sorted(rrf_scores.keys(),
                            key=lambda i: rrf_scores[i],
                            reverse=True)[:candidates_pool]
        
        if not sorted_ids:
            return []
        
        chroma_data = self.collection.get(ids=sorted_ids)
        id_to_idx = {chroma_data["ids"][i]: i for i in range(len(chroma_data["ids"]))}
        
        candidates = []
        for chunk_id in sorted_ids:
            if chunk_id not in id_to_idx:
                continue
            idx = id_to_idx[chunk_id]
            candidates.append({
                "chunk_id": chunk_id,
                "text": chroma_data["documents"][idx],
                "rrf_score": rrf_scores[chunk_id],
                "rerank_score": None,
                "score": rrf_scores[chunk_id],  # may be overwritten by reranker
                "metadata": chroma_data["metadatas"][idx],
            })
        
        # Rerank if enabled
        if self.reranker:
            candidates = self._rerank(query, candidates)
            for c in candidates:
                c["score"] = c["rerank_score"]
        
        return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--variant", choices=["personal", "publishable"],
                        default="personal")
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip the cross-encoder reranking step")
    parser.add_argument("--show-text", action="store_true",
                        help="Show full chunk text (default: first 300 chars)")
    args = parser.parse_args()
    
    print(f"\nQuery: {args.query}")
    print(f"Variant: {args.variant}")
    print(f"Top K: {args.top_k}")
    print(f"Reranking: {'OFF' if args.no_rerank else 'ON'}\n")
    
    print("Loading retriever...")
    retriever = HybridRetriever(
        variant=args.variant,
        verbose=True,
        use_reranker=(not args.no_rerank),
    )
    
    print(f"\nRetrieving...")
    results = retriever.retrieve(args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Top {len(results)} results for: {args.query}")
    print('='*80)
    
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        text = r["text"] if args.show_text else r["text"][:300] + "..."
        score_label = "rerank" if r["rerank_score"] is not None else "rrf"
        
        print(f"\n[{i}] {score_label} score: {r['score']:.4f}")
        if r["rerank_score"] is not None:
            print(f"    (rrf: {r['rrf_score']:.4f})")
        print(f"    Source:  {meta.get('source', 'unknown')}")
        print(f"    Title:   {meta.get('title', '(no title)')}")
        if meta.get("author"):
            print(f"    Author:  {meta['author']}")
        if meta.get("url"):
            print(f"    URL:     {meta['url']}")
        print(f"    ID:      {r['chunk_id']}")
        print(f"    Text:    {text}")


if __name__ == "__main__":
    main()