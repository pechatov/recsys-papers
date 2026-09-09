---
title: "When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems"
category: "generative_retrieval"
slug: "when_memory_takes_gradients_collaborative_vector_memory_for_agentic_recommender_systems_summary"
catalogId: "paper-when_memory_takes_gradients_collaborative_vector_memory_for_agentic_recommender_systems_summary"
paperUrl: "https://arxiv.org/abs/2608.26895"
---
> **Авторы:** Hanchong Chen, Xing Tang, Lingjie Li, Xiongfeng Shan, Xiuqiang He.
>
> **Аффилиации:** Could not be established from accessible arXiv HTML..
>
> **Источник:** arXiv:2608.26895 от 2026-08-27.

## Коротко: о чем статья

CoVeMem хранит collaborative memory как LightGCN vector states и подаёт релевантную историю LLM soft tokens, избегая LLM calls на переписывание памяти.

## Abstract

Agentic recommender systems ground each decision of a large language model (LLM) in a persistent memory of the user, and in existing agents that memory is text: a narrative written and maintained by further LLM calls. Text limits this memory in two ways. It is updated one rewrite at a time, so exploiting the full interaction history is prohibitively expensive; and collaborative evidence, graded similarity over an entire catalog, does not survive translation into sentences. We propose CoVeMem (Collaborative Vector Memory), which vectorizes the collaborative core of the agent's memory. Frozen LightGCN user and item states form the memory bank; at each decision, the candidate set itself retrieves the most relevant historical states, which enter the LLM's context as soft tokens alongside a light textual profile. Contrastive alignment to item-semantic anchors, followed by listwise co-training with masked candidates, teaches the model to read these states and to rank through them; a pointwise yes/no readout scores each candidate. Across four instruction-grounded recommendation benchmarks, CoVeMem matches or exceeds the strongest collaborative text-memory agent on 19 of 20 metric cells while requiring zero additional LLM calls for memory maintenance beyond the shared static profile, against per-interaction calls for text memory. The memory now takes gradients: the full interaction history, out of reach for text, becomes available as training data for what the agent remembers and for how it reads what it remembers.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
