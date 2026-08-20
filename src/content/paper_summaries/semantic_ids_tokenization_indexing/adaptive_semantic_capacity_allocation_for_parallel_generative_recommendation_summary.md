---
title: "Adaptive Semantic Capacity Allocation for Parallel Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "adaptive_semantic_capacity_allocation_for_parallel_generative_recommendation_summary"
catalogId: "paper-adaptive_semantic_capacity_allocation_for_parallel_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.09685"
---
> **Авторы:** Chenxi Li, Yuchen Lu, Xu Yang.
>
> **Аффилиации:** University of the Chinese Academy of Sciences; Institute of Automation, Chinese Academy of Sciences.
>
> **Источник:** arXiv:2608.09685v1 от 2026-08-10. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

InforID автоматически распределяет budget между слотами semantic ID, выбирая длину идентификатора и размеры codebook вместо фиксированной однородной схемы.

## Abstract

Autoregressive semantic ID recommenders are constrained by expensive beam-search decoding, which limits the practical length of item identifiers. Parallel generation methods alleviate this bottleneck by predicting all semantic ID tokens simultaneously, enabling longer IDs. However, existing semantic ID methods still rely on manually predefined and homogeneous ID structures, where both the number of semantic slots and the codebook size of each slot are treated as fixed hyperparameters. This ignores the heterogeneous capacity demands of different semantic subspaces and may allocate prediction capacity to slots with limited utility. We show that uniformly expanding semantic slots can provide limited gains, indicating redundant capacity in homogeneous semantic IDs. We propose InforID, a lightweight adaptive semantic target construction framework for parallel generative recommendation. InforID allocates a fixed capacity budget across candidate semantic slots, thereby jointly determining the effective ID length and slot-specific codebook sizes. Experiments demonstrate improved recommendation accuracy under comparable capacity budgets while preserving one-step parallel prediction.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
