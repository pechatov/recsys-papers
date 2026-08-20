---
title: "HCGRec: Hint-Conditioned Generative Recommendation with Semantic IDs"
category: "semantic_ids_tokenization_indexing"
slug: "hcgrec_hint_conditioned_generative_recommendation_with_semantic_ids_summary"
catalogId: "paper-hcgrec_hint_conditioned_generative_recommendation_with_semantic_ids_summary"
paperUrl: "https://arxiv.org/abs/2608.11980"
---
> **Авторы:** Kangning Zhang, Haotian Fang, Xukun Luo, Hao Yin, Yang Gao, Peng Yan, Weiwen Liu, Weinan Zhang, Yong Yu.
>
> **Аффилиации:** Shanghai Jiao Tong University; Meituan (как указано в HTML-версии arXiv).
>
> **Источник:** arXiv:2608.11980v1 от 2026-08-12. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

HCGRec возвращает сигнал обучения для SID-префиксов, с которых rollouts не достигают target item: подсказанный префикс отделяется от оптимизируемого суффикса.

## Abstract

Semantic-ID generative recommenders represent each item as a short sequence of discrete semantic tokens and predict the next item by autoregressively generating this token sequence. This paradigm enables a unified generation interface for item IDs, histories, and item text, but it also creates a structured optimization bottleneck during reward-based post-training: when an early semantic token enters the wrong branch of the item-token space, finite rollout groups rarely reach the ground-truth item, so group-relative optimization receives identical zero rewards and produces no useful advantage. We propose Hint-Conditioned Generative Recommendation (HCGRec), a semantic-ID generative recommendation framework that recovers learning signal for such hard training instances. HCGRec diagnoses each instance with checkpoint rollouts and supplies a minimal target-prefix hint only when the current generator cannot reach the correct item. The model then generates the unhinted suffix under the hinted semantic branch, turning zero-reward groups into informative comparisons over item-token completions. Hinting also changes token identity: hinted prefix tokens are oracle-provided item context, while unhinted suffix tokens are sampled generation actions. We therefore introduce hint-aware credit decomposition, using supervised learning to preserve item-semantic and prefix-structure alignment for hinted tokens and GRPO to optimize the sampled suffix. Experiments on sequential recommendation benchmarks show that HCGRec substantially improves over supervised fine-tuning and vanilla reward-based post-training, while reducing zero-advantage training samples from over 70% to below 20%. The code is accessible at https://github.com/WncFht/GRec.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
