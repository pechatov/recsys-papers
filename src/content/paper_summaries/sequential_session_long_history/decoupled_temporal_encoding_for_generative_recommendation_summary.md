---
title: "Decoupled Temporal Encoding for Generative Recommendation"
category: "sequential_session_long_history"
slug: "decoupled_temporal_encoding_for_generative_recommendation_summary"
catalogId: "paper-decoupled_temporal_encoding_for_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.16274"
---
> **Авторы:** Pengfei Jia, Jingjian Wang, Jingmao Li, Ge Zhang, Feng Shi.
>
> **Аффилиации:** Unavailable: arXiv HTML conversion returned 404..
>
> **Источник:** arXiv:2608.16274 от 2026-08-17.

## Коротко: о чем статья

DTE разделяет macro-temporal dynamics и локальные order cues для parameter-efficient generative recommendation в сценариях с выраженной временной структурой.

## Abstract

Positional encoding is a fundamental component of Transformer-based generative recommendation models, where user histories are modeled as autoregressive item sequences. Most positional encoding methods are inherited from natural language processing and mainly represent discrete item order. However, recommendation sequences go beyond ordered lists, as timestamps and temporal effects also shape item relations. Our work is motivated by a real-world food delivery and instant retail recommendation system, where user behavior exhibits multi-level temporal regularities, including recency effects, meal-time peaks, weekday-weekend shifts, and promotion-driven traffic bursts. Existing methods partially address this issue through timestamp features, interval embeddings, decay functions, or attention biases, but they usually inject heterogeneous temporal signals through a unified representation or a single modeling pathway, making it difficult to distinguish broad temporal dynamics from local order cues. To address this limitation, we propose Decoupled Temporal Encoding, a lightweight framework for generative recommendation. DTE separates temporal dynamics from order information through two complementary modules: a personalized macro-temporal module that injects compact temporal primitives into item embeddings, and a time-gated micro-sequential module that introduces relative-order bias only when interactions are temporally dense. DTE is also parameter-efficient and deployment-friendly, allowing easy integration into existing systems.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
