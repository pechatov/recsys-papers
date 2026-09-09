---
title: "An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation"
category: "generative_retrieval"
slug: "an_event_is_worth_one_token_event_tokenization_for_industrial_scale_llm_recommendation_summary"
catalogId: "paper-an_event_is_worth_one_token_event_tokenization_for_industrial_scale_llm_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.25546"
---
> **Авторы:** Fan Xia, Zhaoheng Zheng, Iman Setayesh, Ruogu Lin, Yiqin Pan, Samarth Mittal, Wentao Bao, Vinti Pandey, Sachin Patil, Jianpeng Cheng, Jun Xiao, Zhuang Wang, Xiangjun Fan, Sri Reddy, Minghai Chen.
>
> **Аффилиации:** AI at Meta..
>
> **Источник:** arXiv:2608.25546 от 2026-08-26.

## Коротко: о чем статья

AMBER сжимает полный temporal snapshot взаимодействия в кешируемый Event Token, вводя snapshot resolution как новое измерение масштаба LLM recommendation.

## Abstract

LLM-based recommendation has scaled along model capacity and sequence length, yet each position encodes only text, semantic IDs, or a few categorical features, discarding rich user, item, context, and outcome signals available at each event. Under autoregressive modeling, this yields weak queries at each position and, since each position becomes context for the next, the degradation compounds across the sequence. We propose an event-centric paradigm that represents each interaction by its full temporal snapshot, and identify a new scaling dimension we term snapshot resolution: the amount of information encoded per event. To efficiently scale snapshot resolution, we introduce AMBER (Autoregressive Modeling via Bottlenecked Event Representation), which compresses each temporal snapshot into a compact Event Token, a new LLM input modality. The representation is learned end-to-end, while Event Tokens are pre-computed and cached for serving, decoupling snapshot resolution from real-time serving compute. On industrial-scale ranking and retrieval benchmarks, AMBER advances the compute-quality Pareto frontier relative to alternative recommendation paradigms. At sufficient capacity, a single unified tokenizer even outperforms dedicated per-entity tokenizers, demonstrating positive transfer across structurally different entity types. AMBER's Event Tokens also transfer across model architectures: when integrated into a heavily optimized non-LLM ranker as serving-time historical features, they yield statistically significant improvements. Further scaling Event Tokenizer capacity provides additional improvements.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
