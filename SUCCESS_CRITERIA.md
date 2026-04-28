# Success Criteria — Philosophy LLM Project

## What 'done' means

The project is 'done' when:

1. All five archetypes (Tutor, Argument Mapper, Synthesis Engine, Reading Companion, Adversarial Interlocutor) are accessible through Open WebUI as system-prompt presets.
2. RAG retrieval works: when asked about a specific philosopher or text, the model retrieves and cites the relevant source from the corpus.
3. The model has been fine-tuned on philosophy-specific examples and shows measurable improvement on the 100-question eval set vs. the base instruct model.
4. The Constitutional AI training loop has been run, and the model adheres to the constitution principles in spot-checked outputs.
5. The system runs entirely locally on the RTX 5050 with zero external API dependencies.
6. Phase 7 (DPO) is complete OR explicitly deferred.

## What 'good enough' means for each component

Each metric has three tiers: minimum viable (the project ships), good (the project is useful daily), and excellent (the project is genuinely strong).

The project ships at minimum viable. We tune toward 'good' as the active target. 'Excellent' is aspirational and we accept diminishing returns past that point.

### Retrieval (top-5 hit rate on eval questions)

- **Minimum viable:** 70%
- **Good:** 85%
- **Excellent:** 92%+

### Generation quality (constitution-aligned rubric vs. baseline)

- **Minimum viable:** beats baseline by 5%
- **Good:** beats baseline by 15%
- **Excellent:** beats baseline by 25%+

### Factual hallucination rate

- **Minimum viable:** under 10%
- **Good:** under 5%
- **Excellent:** under 2%

### Citation accuracy (when citing, the citation is real)

- **Minimum viable:** 85%
- **Good:** 92%
- **Excellent:** 97%+

## Which phases improve which metrics

These metrics are not all moved by the same techniques. Understanding this prevents wasted effort and false expectations.

- **Retrieval accuracy** is determined by the RAG pipeline (Phase 2) and changes little after that. We must hit our retrieval target during Phase 2, not through later fine-tuning. Improvements come from chunking strategy, embeddings, hybrid search (dense + BM25), reranking, and query rewriting.

- **Generation quality, hallucination rate, and citation accuracy** are improved by fine-tuning (Phase 4), Constitutional AI (Phase 6), and DPO (Phase 7). These touch how the model uses retrieved context, not whether the right context was retrieved in the first place.

We do not fine-tune our way out of bad retrieval. If retrieval is below target after Phase 2, we tune the retrieval pipeline further before proceeding to Phase 4. Bad retrieval combined with a strong model is worse than mediocre retrieval with a mediocre model, because the failure mode looks like success.

## What I am NOT trying to build

- A frontier-quality model. The goal is a useful personal tool, not a production system.
- A model that handles every philosophical question correctly. It will have gaps. That is fine.
- A commercial product. This is for personal use only.
- A model that operates without human oversight. The five archetypes assume an engaged human user.

## When I will know I'm done

The system gets used regularly for actual reading and thinking, and the friction is low enough that I reach for it instead of abandoning it. Concrete proxy: 30 distinct thoughtful interactions across two weeks without me wanting to throw it in the trash.

## Hard constraints (non-negotiable)

- Zero ongoing cost. No paid APIs. No cloud GPUs. No subscriptions.
- Runs on RTX 5050 8GB VRAM only.
- All corpus material from open / public / fair-use sources.

## On revising this document

This document will be revised as the project progresses. Conscious revisions based on new information are healthy. Unconscious drift toward whatever I happen to achieve is not. Any change to thresholds in this document must be accompanied by a brief note in `docs/decisions/` explaining what new information prompted the change.