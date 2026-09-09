---
title: "Adaptive Item-based Collaborative Structures via Noise Rescheduling in Diffusion for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "adaptive_item_based_collaborative_structures_via_noise_rescheduling_in_diffusion_for_generative_recommendation_summary"
catalogId: "paper-adaptive_item_based_collaborative_structures_via_noise_rescheduling_in_diffusion_for_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.23400"
---
> **Авторы:** Jiaqi Wang, Tianying Liu, Heng Chang, Jihong Guan, Wengen Li, Shuigeng Zhou.
>
> **Аффилиации:** Tongji University; Huawei Technologies Co., Ltd.; Fudan University..
>
> **Источник:** arXiv:2608.23400 от 2026-08-24.

## Коротко: о чем статья

ANR-DiffRec добавляет item co-occurrence в SID и динамически меняет diffusion noise schedule по локальной восстановимости и collaborative dependencies.

## Abstract

Discrete Diffusion Models (DDMs) have recently been introduced to recommendation systems, modeling user history as a token generation process via iterative denoising. However, while effective at capturing user-level sequential patterns, these methods often fail to explicitly integrate item-based collaborative filtering information, a critical component for accurate recommendation. This deficiency manifests in two key aspects: (1) the item representation is often semantic-focused, lacking collaborative priors for diffusion training; and (2) the denoising process employs a uniform noise schedule, treating all tokens indiscriminately and ignoring item-level adaptive structural dependencies. To bridge this gap, we propose ANR-DiffRec, a unified framework designed to encode item-based collaborative structures into discrete diffusion for generative recommendation. First, we explicitly incorporate an item co-occurrence matrix to guide semantic ID generation, providing a structured collaborative prior for discrete diffusion training. Second, we introduce an item-based adaptive noise rescheduling mechanism that dynamically adjusts denoising weights according to both local contextual recoverability and behavior-aware item dependencies. Specifically, the proposed strategy jointly models intra-item structural context and inter-item collaborative signals, enabling structure-aware denoising during diffusion training. Extensive experiments on multiple benchmarks demonstrate that our method consistently outperforms state-of-the-art generative recommendation models. Code: https://github.com/CalmaQi/ANR-DiffRec.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
