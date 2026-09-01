---
title: "SCoRD: Semantic-Assisted Continual Retriever-Reranker Distillation for LLM-Based Recommendation"
category: "generative_retrieval"
slug: "scord_semantic_assisted_continual_retriever_reranker_distillation_for_llm_based_recommendation_summary"
catalogId: "paper-scord_semantic_assisted_continual_retriever_reranker_distillation_for_llm_based_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.19998"
---
> **Авторы:** Seunghyun Baek, Gyuseok Lee, Seunghan Lee, Wonbin Kweon, Dong Wang, SeongKu Kang.
>
> **Аффилиации:** Korea University; University of Illinois at Urbana-Champaign; Sungkyunkwan University. Explicitly listed in arXiv HTML; author-to-institution mapping was not fully machine-extractable..
>
> **Источник:** arXiv:2608.19998 от 2026-08-20.

## Коротко: о чем статья

SCoRD переносит intent-level guidance от LLM reranker к retriever и поддерживает их совместную continual adaptation без постоянных дорогостоящих LLM updates.

## Abstract

Recommendation systems increasingly adopt a two-stage pipeline, where an ID-based retriever retrieves candidates and an LLM-based reranker refines their rankings. To improve retrieval quality, reranker-to-retriever distillation is commonly used to transfer the reranker's knowledge to the retriever. For practical deployment, however, this pipeline must continually adapt to evolving interests and incoming interactions. A naive solution is to repeatedly update the LLM reranker and distill its latest knowledge, but this incurs prohibitive costs. Updating the retriever alone is cheaper, but its limited capacity makes adaptation from sparse data difficult. We propose SCoRD, a continual knowledge distillation framework for LLM-based reranking pipelines under a non-stationary data stream. SCoRD introduces a semantic reasoning assistant that distills the LLM's ability to infer underlying user intents into reusable intent-level guidance. It selectively distills reranker knowledge to the retriever on low-confidence sequences, guides retriever-only updates without repeated LLM inference, and feeds retriever-derived representations and intent-drift signals back to the reranker. Experiments on real-world datasets show that SCoRD enables effective and efficient retriever-reranker co-adaptation.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
