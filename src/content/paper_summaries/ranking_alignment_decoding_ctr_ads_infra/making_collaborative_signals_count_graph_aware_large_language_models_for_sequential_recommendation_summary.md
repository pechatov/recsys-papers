---
title: "Making Collaborative Signals Count: Graph-Aware Large Language Models for Sequential Recommendation"
category: "ranking_alignment_decoding_ctr_ads_infra"
slug: "making_collaborative_signals_count_graph_aware_large_language_models_for_sequential_recommendation_summary"
catalogId: "paper-making_collaborative_signals_count_graph_aware_large_language_models_for_sequential_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.12184"
---
> **Авторы:** Fenglin Yan, Bohao Wang, Jian Zhang, Yu Cui, Tongya Zheng, Ye Feng, Can Wang, Jiawei Chen.
>
> **Аффилиации:** Не удалось установить по доступной HTML-версии arXiv.
>
> **Источник:** arXiv:2608.12184v1 от 2026-08-12. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

GALLM вводит обучаемые graph-derived biases в attention LLM, объединяя text--text, item--text и item--item collaborative relations без отдельного graph encoder.

## Abstract

Large language models (LLMs) have been widely adopted as backbones for recommender systems. However, their language-centric pretraining makes it difficult to capture collaborative signals implicit in user-item interactions, which are crucial for personalized recommendation. Existing methods either inject collaborative representations produced by external recommenders or model only intra-sequence dependencies, limiting their ability to exploit global collaborative patterns. To address this limitation, we propose GALLM, a graph-aware LLM framework for sequential recommendation. GALLM constructs a collaborative graph over text tokens and item tokens, and models three types of relations: Text--Text relations for preserving semantic dependencies, Item--Text relations for aligning item tokens with their textual descriptions, and Item--Item relations derived from global item co-occurrence patterns. These relations are transformed into lightweight learnable attention biases and incorporated into the LLM attention mechanism, enabling collaborative-aware token interactions without introducing an additional graph encoder. Experiments on four real-world benchmarks show that GALLM achieves the best performance among the compared baselines, improving over the strongest baseline by 9.76\% on average in HR@5.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
