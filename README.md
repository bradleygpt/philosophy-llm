# Philosophy LLM (working title)

A locally-hosted philosophical research assistant: fine-tuned LLM with RAG retrieval over Stanford Encyclopedia of Philosophy and other open philosophical sources, five interaction archetypes, fully local and zero ongoing cost.

Final name TBD. Shortlist: maieutica, peripatos, protreptic, theoria, sullogismos.

## Stack

- **Base model:** Llama 3.2 3B Instruct
- **Fine-tuning:** QLoRA via Unsloth
- **Embeddings:** BGE-M3
- **Vector DB:** Chroma (local)
- **Retrieval:** Hybrid dense + BM25 + cross-encoder reranking
- **Serving:** Ollama (OpenAI-compatible API)
- **Frontend:** Open WebUI with five archetype presets
- **Alignment:** Constitutional AI + zero-cost RLAIF (local judge) + DPO

## Five archetypes

- **Tutor** — Q&A grounded in literature
- **Argument Mapper** — for/against argument trees
- **Synthesis Engine** — cross-cutting comparisons
- **Reading Companion** — paste paragraphs, get analysis
- **Adversarial Interlocutor** — strongest opposing views

## Hard constraints

- Zero ongoing cost
- Runs on RTX 5050 8GB VRAM
- All sources open / public domain / fair use respecting

## Status

In Phase 0: project skeleton, success criteria, license review.

See SUCCESS_CRITERIA.md for what 'done' means.