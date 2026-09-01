---
title: "The Disconnect Between Better Descriptive Reasoning Trace Quality and Recommendation Effectiveness"
category: "semantic_ids_tokenization_indexing"
slug: "the_disconnect_between_better_descriptive_reasoning_trace_quality_and_recommendation_effectiveness_summary"
catalogId: "paper-the_disconnect_between_better_descriptive_reasoning_trace_quality_and_recommendation_effectiveness_summary"
paperUrl: "https://arxiv.org/abs/2608.23154"
---
> **Авторы:** Gustavo Penha, Juan Elenter, Claudia Hauff, Hugues Bouchard, Paul Bennett, Mounia Lalmas.
>
> **Аффилиации:** Could not be established from accessible arXiv HTML..
>
> **Источник:** arXiv:2608.23154 от 2026-08-24.

## Коротко: о чем статья

Контролируемое исследование показывает: более понятные descriptive reasoning traces над titles или SID не гарантируют лучшую offline recommendation effectiveness.

## Abstract

Recent work has focused on improving explicit natural-language descriptive reasoning traces for generative recommendation. This includes systems that augment semantic ID (SID) prediction with chain-of-thought reasoning. However, because SIDs are opaque learned identifiers rather than natural language, they require costly alignment before an LLM can reason over them. This provides a controlled experimental setting in which both item representation (Title vs. SID) and semantic grounding (minimal vs. extensive SID alignment) can be varied independently. We therefore present the first controlled comparison of descriptive reasoning trace quality across semantic IDs and natural-language titles in a 2 x 2 factorial study on three Amazon product domains using a shared Qwen3-1.7B backbone. We find that introducing explicit descriptive reasoning traces reduces traditional offline recommendation effectiveness under standard SFT and RL training, even though natural language titles produce substantially more grounded and interpretable traces. Extensive SID alignment improves descriptive trace quality but not traditional offline recommendation effectiveness, while a richer reward signal partially recovers performance. Overall, our results show that improving descriptive reasoning trace quality is not, by itself, sufficient to consistently improve traditional offline recommendation effectiveness under the training objectives and evaluation protocols studied here.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
