---
title: "Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals"
category: "generative_retrieval"
slug: "training_free_llm_based_recommendation_with_post_llm_item_refinement_using_collaborative_signals_summary"
catalogId: "paper-training_free_llm_based_recommendation_with_post_llm_item_refinement_using_collaborative_signals_summary"
paperUrl: "https://arxiv.org/abs/2608.19665"
---
> **Авторы:** Kyungho Kim, Sunwoo Kim, Geon Lee, Shinhwan Kang, Sojeong Kim, Liam Collins, Bhuvesh Kumar, Donald Loveland, Kijung Shin.
>
> **Аффилиации:** KAIST; Snap Inc. Explicitly listed in arXiv HTML; author-to-institution mapping was not fully machine-extractable..
>
> **Источник:** arXiv:2608.19665 от 2026-08-20.

## Коротко: о чем статья

CoRRe уточняет LLM-представления товаров сигналами co-purchase graph и popularity уже после генерации, не требуя обучения или fine-tuning.

## Abstract

Large language models (LLMs) have shown promise for training-free recommendation, but LLM-generated user interests are often too broad for fine-grained item retrieval. Existing methods incorporate collaborative filtering (CF) signals in a pre-LLM manner through candidate reranking or prompt augmentation, yielding limited gains. We propose CoRRe, a training-free recommendation framework with a post-LLM paradigm that injects CF signals into LLM-generated item representations, which are later matched with LLM-generated user interests for ranking. Specifically, CoRRe refines the directions of item embeddings using an item-item co-purchase graph and their magnitudes using item popularity. Experiments on real-world datasets show that CoRRe consistently outperforms existing training-free methods and achieves competitive or superior performance compared with training-based methods, without requiring any model training or task-specific fine-tuning.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
