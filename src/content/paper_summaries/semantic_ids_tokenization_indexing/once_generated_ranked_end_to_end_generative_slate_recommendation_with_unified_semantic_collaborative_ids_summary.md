---
title: "Once Generated, Ranked: End-to-End Generative Slate Recommendation with Unified Semantic-Collaborative IDs"
category: "semantic_ids_tokenization_indexing"
slug: "once_generated_ranked_end_to_end_generative_slate_recommendation_with_unified_semantic_collaborative_ids_summary"
catalogId: "paper-once_generated_ranked_end_to_end_generative_slate_recommendation_with_unified_semantic_collaborative_ids_summary"
paperUrl: "https://arxiv.org/abs/2608.17613"
---
> **Авторы:** Yang Hu, Jiayi Guo, Jingui Ma, Ning Li, Jiangling Qin, Yanming Li, Yang Deng, Xiaoshuang Chen, Kaiqiao Zhan.
>
> **Аффилиации:** Unavailable: arXiv HTML contains no accessible explicit affiliation block..
>
> **Источник:** arXiv:2608.17613 от 2026-08-18.

## Коротко: о чем статья

OGR генерирует упорядоченный slate end-to-end: semantic-collaborative IDs, list-wise planning и reward-guided оптимизация связывают generation со slate utility.

## Abstract

Slate recommendation treats a slate rather than an individual item as the recommendation unit, requiring joint optimization of item interactions and slate utility. Existing approaches typically separate candidate generation from ranking and restrict optimization to retrieved candidates. Generative recommendation with Semantic IDs (SIDs) offers a path to end-to-end recommendation, but existing SID construction often lacks recommendation-aware semantics and effective local collaborative signals, while next-token prediction is misaligned with slate-level objectives. We propose OGR, an end-to-end framework that directly generates ordered slates-"Once Generated, Ranked." OGR first introduces TUSID, which adaptively fuses item-specific semantic and local collaborative information into hierarchical SIDs. It then uses list-wise preference planning and pipelined position-wise SID decoding to model global preferences and inter-item dependencies while generating ordered slates. We further propose SPA, a reward-guided conservative policy optimization method that aligns generated slates with user preferences beyond likelihood imitation. Offline experiments show that OGR outperforms representative baselines, with 48.2% and 27.2% relative NDCG@5 gains on industrial and public datasets, respectively. Online A/B testing on Kuaishou further yields a 1.120% improvement in Effective Views.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
