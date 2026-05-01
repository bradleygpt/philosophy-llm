"""
Qualitative test suite for hybrid retrieval.

Runs ~15 hand-written queries through the retrieval pipeline and prints
the top results for visual inspection. This is a sanity check, not a
quantitative eval — we don't have ground-truth labels yet.

Usage:
  python scripts/test_retrieval.py                          # full test, with reranking
  python scripts/test_retrieval.py --variant=publishable    # test publishable variant
  python scripts/test_retrieval.py --no-rerank              # skip reranking for speed
  python scripts/test_retrieval.py --top-k=5                # show fewer results per query
"""

import argparse
import sys
from pathlib import Path
from time import time

# Add project root to path so we can import retrieval module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retrieve import HybridRetriever


# Test queries spanning the corpus.
# Each tuple: (query, expected_signals)
# expected_signals are keywords/sources we'd hope to see in good results.
TEST_QUERIES = [
    {
        "query": "What is Kant's categorical imperative?",
        "expected_signals": ["kant", "categorical", "imperative", "moral law"],
    },
    {
        "query": "Plato's theory of forms",
        "expected_signals": ["plato", "forms", "ideas", "republic"],
    },
    {
        "query": "Hume's argument against miracles",
        "expected_signals": ["hume", "miracle", "testimony", "enquiry"],
    },
    {
        "query": "What is consciousness?",
        "expected_signals": ["consciousness", "qualia", "phenomenal", "experience"],
    },
    {
        "query": "AI alignment problem",
        "expected_signals": ["alignment", "ai", "values", "safety"],
    },
    {
        "query": "Aristotle on virtue and the good life",
        "expected_signals": ["aristotle", "virtue", "eudaimonia", "ethics"],
    },
    {
        "query": "Wittgenstein's later philosophy of language",
        "expected_signals": ["wittgenstein", "language", "use", "investigations"],
    },
    {
        "query": "Quine's web of belief",
        "expected_signals": ["quine", "belief", "holism", "naturalism"],
    },
    {
        "query": "Nietzsche on the death of God",
        "expected_signals": ["nietzsche", "god", "death", "morality"],
    },
    {
        "query": "free will and determinism",
        "expected_signals": ["free will", "determinism", "compatibilism", "causation"],
    },
    {
        "query": "phenomenology of perception",
        "expected_signals": ["phenomenology", "perception", "experience", "husserl"],
    },
    {
        "query": "Spinoza's metaphysics of substance",
        "expected_signals": ["spinoza", "substance", "ethics", "monism"],
    },
    {
        "query": "fairness in machine learning",
        "expected_signals": ["fairness", "bias", "machine learning", "algorithm"],
    },
    {
        "query": "trolley problem",
        "expected_signals": ["trolley", "moral", "consequentialism", "deontology"],
    },
    {
        "query": "Descartes' cogito ergo sum",
        "expected_signals": ["descartes", "cogito", "doubt", "meditations"],
    },
]


def check_signals(text: str, signals: list[str]) -> tuple[int, list[str]]:
    """Count how many expected signal terms appear in the text. Case-insensitive."""
    text_lower = text.lower()
    hits = [s for s in signals if s.lower() in text_lower]
    return len(hits), hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["personal", "publishable"],
                        default="personal")
    parser.add_argument("--top-k", type=int, default=3,
                        help="How many results to show per query (default 3)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip the cross-encoder reranking step")
    parser.add_argument("--show-text", action="store_true",
                        help="Show full chunk text (default: first 200 chars)")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"RETRIEVAL TEST SUITE")
    print(f"{'='*80}")
    print(f"Variant:    {args.variant}")
    print(f"Top-K:      {args.top_k}")
    print(f"Reranking:  {'OFF' if args.no_rerank else 'ON'}")
    print(f"Test count: {len(TEST_QUERIES)}")
    print()
    
    print("Loading retriever (one-time setup)...")
    t0 = time()
    retriever = HybridRetriever(
        variant=args.variant,
        verbose=True,
        use_reranker=(not args.no_rerank),
    )
    print(f"  Setup complete in {time() - t0:.1f}s\n")
    
    # Track summary stats
    all_signal_hit_rates = []
    all_query_times = []
    source_distribution = {}
    
    for q_idx, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        signals = test["expected_signals"]
        
        print(f"\n{'─'*80}")
        print(f"[{q_idx}/{len(TEST_QUERIES)}] Query: {query}")
        print(f"            Expected signals: {signals}")
        print(f"{'─'*80}")
        
        t_query = time()
        results = retriever.retrieve(query, top_k=args.top_k)
        query_time = time() - t_query
        all_query_times.append(query_time)
        
        if not results:
            print("  No results.")
            all_signal_hit_rates.append(0.0)
            continue
        
        # Signal check across top results
        combined_top_text = " ".join(r["text"] for r in results)
        sig_count, sig_hits = check_signals(combined_top_text, signals)
        sig_rate = sig_count / len(signals) if signals else 0
        all_signal_hit_rates.append(sig_rate)
        
        print(f"\n  Signal hit rate: {sig_count}/{len(signals)} = {sig_rate:.0%}")
        print(f"  Hits: {sig_hits}")
        print(f"  Query time: {query_time:.2f}s\n")
        
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            text_preview = r["text"][:300] if not args.show_text else r["text"]
            text_preview = text_preview.replace("\n", " ")
            
            source = meta.get("source", "?")
            source_distribution[source] = source_distribution.get(source, 0) + 1
            
            score_label = "rerank" if r["rerank_score"] is not None else "rrf"
            
            print(f"  [{i}] {score_label}={r['score']:.3f}  source={source}  title={meta.get('title', '(no title)')[:60]}")
            print(f"      {text_preview[:300]}{'...' if len(r['text']) > 300 else ''}")
            print()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Queries tested:           {len(TEST_QUERIES)}")
    print(f"Average signal hit rate:  {sum(all_signal_hit_rates) / len(all_signal_hit_rates):.0%}")
    print(f"Average query time:       {sum(all_query_times) / len(all_query_times):.2f}s")
    print(f"\nSource distribution in top results:")
    for source, count in sorted(source_distribution.items(), key=lambda x: -x[1]):
        print(f"  {source:12s} {count} hits")
    
    print(f"\nQuality bar reminder (from SUCCESS_CRITERIA.md):")
    print(f"  Minimum viable: 70% retrieval accuracy")
    print(f"  Good:           85%")
    print(f"  Excellent:      92%+")
    print(f"\nNote: signal hit rate is a rough proxy. True retrieval accuracy")
    print(f"requires ground-truth labeled queries (coming in eval suite).")


if __name__ == "__main__":
    main()