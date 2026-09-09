---
title: "A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation"
category: "sequential_session_long_history"
slug: "a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cross_domain_sequential_recommendation_summary"
catalogId: "paper-a_dual_expert_strategy_integrating_llms_to_mitigate_negative_transfer_in_cross_domain_sequential_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.23131"
---
> **Авторы:** Hyeongjun Yun, Kihyuk Song, Jaegul Choo, Chung Park.
>
> **Аффилиации:** SK Telecom; KAIST..
>
> **Источник:** arXiv:2608.23131 от 2026-08-24.

## Коротко: о чем статья

DuELRec сочетает domain-gated single- и cross-domain experts с item-level contrastive learning, чтобы уменьшать negative transfer в CDSR.

## Abstract

Cross-Domain Sequential Recommendation (CDSR) predicts the next item a user will interact with based on their historical interaction sequences across multiple domains. Recent approaches leverage Large Language Models (LLMs) finetuned on textual representations of cross-domain user sequences to retrieve the recommended items, referred to as LLMRec. However, LLMRec primarily models the autoregressive patterns of token-level item texts, while overlooking item-level collaborative signals. This semantic misalignment often leads to distorted knowledge transfer across domains-termed negative transfer degrading performance in the CDSR task. To address this issue, we propose a novel LLM-based CDSR model, DuELRec: Domain-Gated Dual Experts with LLMs for Cross-Domain Sequential Recommendation. We propose a domain-gated dual-expert framework, equipped with an item-aware attention transformation module, which aggregates textual subtokens into item-level representations and enforces block-level attention masking. The single-domain expert restricts autoregressive attention to items within the same domain, while the cross-domain expert allows it across all domains. A gating mechanism adaptively fuses their outputs, using single-domain signals to reduce cross-domain noise that causes negative transfer. Second, we introduce a dual-sampling token-to-item contrastive learning objective that allows LLMs to capture the item-level collaborative signals from both single- and cross-domains. This is achieved by transforming token-level item texts into item-level representations and applying stochastic negative sampling from both single- and cross-domain item pools for contrastive learning. Extensive experiments on two real-world datasets across ten domains show that our model outperforms 26 state-of-the-art methods in recommendation performance.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
