# Philosophy LLM

A locally-hosted philosophical research assistant with retrieval-augmented generation, fine-tuned on a curated philosophy corpus.

**Status:** Phase 1 complete (corpus acquisition). Phase 2 (RAG infrastructure) next.

## Project goals

Build a personal philosophical reading and thinking tool with five archetypes:

- **Tutor** — Q&A on philosophical literature
- **Argument Mapper** — for/against argument trees
- **Synthesis Engine** — cross-cutting comparisons
- **Reading Companion** — paragraph-level analysis
- **Adversarial Interlocutor** — strongest opposing views

Hard constraints (non-negotiable):
- Zero ongoing cost (no paid APIs, no cloud GPUs, no subscriptions)
- Runs entirely locally on RTX 5050 8GB
- All corpus material from open / public / fair-use sources

See [SUCCESS_CRITERIA.md](SUCCESS_CRITERIA.md) for tiered metrics on retrieval, generation, and citation accuracy.

## Architecture

- **Base model:** Llama 3.2 3B Instruct
- **Fine-tuning:** QLoRA via Unsloth
- **Embeddings:** BGE-M3
- **Vector DB:** Chroma (local)
- **Retrieval:** Hybrid dense + BM25 + cross-encoder reranking
- **Serving:** Ollama (OpenAI-compatible API)
- **Frontend:** Open WebUI with five archetype presets
- **Alignment:** Constitutional AI + zero-cost RLAIF (local judge) + DPO

## Two-variant strategy

Two model variants are produced from one codebase:

- **Personal variant** — trained on full corpus including Stanford Encyclopedia of Philosophy. Local use only, never distributed.
- **Publishable variant** — trained on IEP + Gutenberg + arXiv only (no SEP). Shareable as artifact.

See [docs/decisions/0001-corpus-strategy-and-distribution.md](docs/decisions/0001-corpus-strategy-and-distribution.md) for the full reasoning.

## Phase 1 corpus

| Source | Files | Size | Variants |
|--------|-------|------|----------|
| Project Gutenberg | 37 | 18.8 MB | personal, publishable |
| arXiv philosophy | 240 | 17.7 MB | personal, publishable |
| Internet Encyclopedia of Philosophy | 889 | 56.6 MB | personal, publishable |
| Stanford Encyclopedia of Philosophy | 1,857 | 179.4 MB | personal only |
| **Total** | **3,023** | **272.5 MB** | |

Approximately 65 million tokens of philosophical text spanning Plato through contemporary AI ethics.

Acquisition scripts in [scripts/](scripts/), source manifests in [corpus/manifests/](corpus/manifests/), source-to-variant mapping in [corpus/sources.yaml](corpus/sources.yaml). Raw corpus files are gitignored — anyone with the repo can run the scripts to reproduce the corpus.

## Project phases

- [x] **Phase 0** — Foundation (folder structure, git, ADRs, success criteria)
- [x] **Phase 1** — Corpus acquisition
  - [x] 1.1 Configuration (`sources.yaml`)
  - [x] 1.2 Project Gutenberg
  - [x] 1.3 arXiv philosophy
  - [x] 1.4 Internet Encyclopedia of Philosophy
  - [x] 1.5 Stanford Encyclopedia of Philosophy
  - [x] 1.6 Final manifest pass
  - [x] 1.7 Spot-checks
- [ ] **Phase 2** — RAG infrastructure (chunking, embeddings, indexing)
- [ ] **Phase 3** — First archetype baseline (Reading Companion via Ollama)
- [ ] **Phase 4** — LoRA fine-tuning (personal + publishable variants)
- [ ] **Phase 5** — All five archetypes via Open WebUI presets
- [ ] **Phase 6** — Constitutional AI training + zero-cost RLAIF
- [ ] **Phase 7** — DPO refinement

## Reproducing the corpus

```bash
# Activate venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run acquisition (each script is resumable)
python scripts/acquire_gutenberg.py    # ~10 min
python scripts/acquire_arxiv.py        # ~30 min
python scripts/acquire_iep.py          # ~75 min
python scripts/acquire_sep.py          # ~155 min (personal variant only)

# Generate manifests
python scripts/generate_manifest.py gutenberg
python scripts/generate_manifest.py arxiv
python scripts/generate_manifest.py iep
python scripts/generate_manifest.py sep
```

All scripts use polite delays and respect publisher terms. SEP is acquired only for the personal variant per ADR-0001.

## License

Code: MIT (see LICENSE).

Corpus: each source retains its original license. See `corpus/sources.yaml` and per-source manifests for attribution and licensing details. SEP entries are subject to Stanford's terms of use and are excluded from the publishable variant.

Model weights (when produced): the publishable variant will be released under a permissive license; the personal variant is local-only and not distributed.

## Author

Bradley Hartnett