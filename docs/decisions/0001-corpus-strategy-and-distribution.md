# ADR 0001 — Corpus Strategy and Distribution Policy

**Status:** Accepted
**Date:** 2026-04-28
**Decision makers:** Bradley

## Context

The project trains a fine-tuned LLM with RAG retrieval over philosophical sources. The Stanford Encyclopedia of Philosophy (SEP) is the highest-quality general reference work in philosophy and is the best single corpus source. SEP's terms of use:

- Permit individual users to read, download, copy, search, and crawl entries for indexing (subject to reasonable network usage)
- Restrict redistribution: authors retain copyright; Stanford holds an exclusive license for online publication
- Do not directly address LLM training or model distribution (the terms predate this question)

Other high-quality corpus sources have permissive licenses:
- Internet Encyclopedia of Philosophy (IEP): open access with attribution
- Project Gutenberg: public domain
- arXiv philosophy papers: typically CC-licensed
- Open access journal archives

The project goal is twofold: a high-quality personal philosophical research tool, and the optional ability to share an artifact with a small circle (e.g., partner, friends).

## Decision

### Two model variants from one codebase

**Personal variant:** trained on SEP + IEP + Gutenberg + arXiv. Used only on Bradley's local hardware. Never distributed, never deployed externally.

**Publishable variant:** trained on IEP + Gutenberg + arXiv only (no SEP). Same code, same archetypes, same evaluation harness. Distributable as an artifact (model weights) and deployable to a free-tier service for controlled-access live demos.

A `corpus/sources.yaml` configuration file declares which sources belong to which variant. This makes provenance auditable.

### Distribution policy

**Code (GitHub repo):** public from day one. No restrictions.

**Personal variant weights:** private. Stored locally only. Never uploaded, never shared, never deployed remotely.

**Publishable variant weights:** shareable. Can be uploaded to Hugging Face, distributed via direct file transfer, or deployed to a free-tier hosting service.

**Live deployment of publishable variant:** permitted on free tiers only (Hugging Face Spaces, Oracle Cloud Free Tier, Cloudflare Tunnel + existing hardware, etc.). Open WebUI's built-in user authentication is the access control mechanism — admin approval required for new users. Paid hosting is excluded by the project's zero-cost rule.

### Phase 1 corpus acquisition constraints

Crawling of SEP and IEP follows these rules:

1. Maximum 1 request per 3 seconds (5 seconds preferred). Slow, polite crawling.
2. Respect `robots.txt`.
3. One-time corpus acquisition. No repeated re-crawls without cause.
4. Per-entry metadata preserved: author, title, version, source URL.
5. Raw and processed corpora stay in `corpus/raw/` and `corpus/processed/` (both gitignored).

### What is NOT decided here

- The decision to actually deploy the publishable variant publicly. That is deferred to a future Phase 8 if and when Bradley wants to share it. The infrastructure to do so is built; the choice to use it is separate.
- The legal status of distillation chains (training a publishable model on synthetic data from the SEP-trained model). Not pursued because Path C (separate corpora, separate variants) is cleaner and produces a stronger publishable artifact.
- Future commercial use. The project is non-commercial. If commercialization ever becomes a question, it requires a fresh legal review and likely direct permission from sources.

## Consequences

### Positive

- Personal variant is the strongest possible philosophical tool for Bradley's daily use.
- Publishable variant exists as a clean artifact, free of SEP licensing concerns.
- Code is fully open from day one — portfolio value preserved.
- Free-tier deployments are possible for sharing with girlfriend / small circle.
- Zero ongoing cost preserved.
- Provenance is auditable via `sources.yaml`.

### Negative

- Approximately 1.3x the work of a single-variant project (extra corpus pipeline, extra fine-tuning run, extra evaluation pass per variant). Most code is shared.
- Publishable variant has slightly less polished entries than personal variant (no SEP).
- Free-tier deployments are best-effort; if a platform changes terms, the live demo may go down.

### Mitigations

- The two variants share almost all infrastructure. Marginal cost is real but small.
- IEP + Gutenberg + arXiv is a genuinely strong corpus for philosophy. The publishable variant is not crippled.
- Live demo going down is acceptable — the artifact (model weights) and the local-run path always work regardless of hosting status.

## Implementation notes

- `corpus/sources.yaml` is the configuration file driving variant assignment.
- Two retrieval indices: `retrieval/index_personal/` and `retrieval/index_publishable/` (both gitignored).
- Two LoRA adapter sets in `training/`.
- Open WebUI configured with both models in the dropdown for local use.
- Model cards documenting each variant's training data and intended use.

## Revisiting this decision

This ADR should be revisited if:

- Stanford issues clarification on LLM training over SEP content.
- New publishable sources emerge that meaningfully strengthen the publishable corpus.
- The project moves toward commercialization (would require fresh analysis).
- Free-tier hosting options materially change.

Any change to this decision must be documented in a follow-up ADR (`0002-...md`).