# ADR-0002: Serving Runtime and Base Model Contingencies

**Status:** Accepted  
**Date:** 2026-05-01

## Context

Phase 3 of the Philosophy LLM project requires:
1. A **serving runtime** to run language models locally
2. A **base model** for inference (and later fine-tuning in Phase 4)

The project's hard constraint is **zero ongoing cost**. Both choices must remain free in perpetuity, or we must have viable alternatives ready.

## Decision

**Primary serving runtime:** Ollama (MIT license)  
**Primary base model:** Llama 3.2 3B Instruct (Meta Community License)

## Rationale

**Ollama (MIT)**
- Permissive license; can be copied, modified, redistributed, and even sold
- Active community, broad model support
- OpenAI-compatible API → portable to other backends
- Phase 5 (Open WebUI) is built around Ollama-compatible APIs

**Llama 3.2 3B Instruct (Meta Community License)**
- Free for personal and commercial use under 700M MAU threshold (irrelevant for personal project)
- Fits in 8GB VRAM with headroom for inference
- Strong instruction-following baseline
- Same model family as future fine-tuning target (transfer of skills/code)

## Risks

**Risk 1: Meta changes Llama license.** Possible but unlikely in this direction (Meta has consistently moved *toward* permissive). Risk: revoked rights, new restrictions, or effective paywalling.

**Risk 2: Ollama project becomes commercial.** Less likely (MIT prevents license retraction on existing versions), but the maintained version could change.

**Risk 3: Hardware obsolescence.** If RTX 5050 inadequate for newer models. Less of a license risk, more of a capability risk.

## Contingency Plan

If Llama becomes restricted, **switch to one of these in priority order:**

1. **Qwen 2.5 3B Instruct** (Apache 2.0) — Same size class, open license, strong English performance. *Drop-in replacement.*
2. **Mistral 7B Instruct** (Apache 2.0) — Larger, slightly slower on RTX 5050, but unrestricted. Production-grade.
3. **Phi-3.5 Mini** (MIT) — 3.8B, fully open, Microsoft-backed. Solid fallback.
4. **Gemma 2 2B** (Gemma license; mostly permissive but with use restrictions) — Smallest, fastest, decent quality.

If Ollama becomes restricted, **switch to one of these:**

1. **llama.cpp** (MIT) — Direct C++ inference engine that Ollama wraps. Lower-level but no dependency on Ollama project.
2. **vLLM** (Apache 2.0) — Production-grade serving, more performant for batch workloads.
3. **Direct PyTorch** (BSD) — Most fundamental fallback; we can always load weights and run inference manually.

## Switching Cost

**Model switch (Llama → Qwen):** ~2-4 hours
- Pull new model via Ollama
- Update `serving/config.yaml` with model name
- Re-run Phase 4 fine-tuning on new base
- Possible prompt template adjustments

**Runtime switch (Ollama → llama.cpp):** ~4-8 hours
- Set up llama.cpp build environment
- Convert model weights to GGUF format (likely already are)
- Replace API calls in retrieval/generation pipeline
- Test for parity

Both switches are recoverable; neither blocks the project. The architecture intentionally talks to **OpenAI-compatible APIs**, which Ollama, llama.cpp server, and vLLM all expose. **The serving layer is interchangeable.**

## Watch Conditions

We will revisit this decision if:
- Meta announces commercial license changes for Llama
- Ollama relicenses or significantly changes pricing model
- A new model in the 3B parameter class significantly outperforms Llama 3.2 3B

## Decision Outcome

Proceed with Ollama + Llama 3.2 3B Instruct for Phase 3. Contingency stack documented above. ADR will be amended if the watch conditions trigger.