---
title: "FSGR: Mitigating Token Frequency Bias for Fair SID-Based Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "fsgr_mitigating_token_frequency_bias_for_fair_sid_based_generative_recommendation_summary"
catalogId: "paper-fsgr_mitigating_token_frequency_bias_for_fair_sid_based_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.12845"
---
> **Авторы:** Yuchen Zheng, Sihan Xu, Jingwen Yang, Xiangrui Cai, Haiwei Zhang, Xiaojie Yuan.
>
> **Аффилиации:** College of Computer Science, Nankai University; College of Cryptology and Cyber Science, Nankai University.
>
> **Источник:** arXiv:2608.12845v1 от 2026-08-13. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

FSGR балансирует частоты SID-токенов при построении идентификаторов и обучении, чтобы уменьшить exposure bias без заметной потери recommendation accuracy.

## Abstract

Semantic ID (SID)-based generative recommendation has recently achieved remarkable success. However, existing methods suffer from a previously overlooked fairness issue, which we term \textbf{Token Frequency Bias}, where high-frequency SID tokens are systematically over-predicted while low-frequency SID tokens are under-predicted. This bias originates from the combined effects of imbalanced semantic codebooks during SID construction, and popularity bias together with the maximum likelihood estimation objective during recommendation training, resulting in unfair exposure across item categories. Existing SID methods mainly focus on improving codebook quality and overlook the impact of token frequency imbalance on downstream recommendation fairness, while LLM debiasing methods often yield suboptimal results when directly applied to SID-based recommendation, due to the hierarchical semantics of SID tokens. To address this issue, we propose \textbf{FSGR}, a fairness optimization framework for SID-based generative recommendation. During SID construction, FSGR employs OT-based Assignment Optimization and Dual-Criteria Re-anchor mechanism to form a more balanced SID representation space. During recommendation training, it adopts a two-stage training strategy and introduces Hierarchical Frequency Calibration for layer-specific fairness fine-tuning. Experiments on three public datasets with three backbone models demonstrate that FSGR mitigates token frequency bias and delivers an average Gini fairness improvement of over 20\% while maintaining competitive recommendation accuracy.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
