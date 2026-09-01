---
title: "TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising"
category: "semantic_ids_tokenization_indexing"
slug: "tagr_temporally_adaptive_generative_recommendation_for_industrial_live_streaming_advertising_summary"
catalogId: "paper-tagr_temporally_adaptive_generative_recommendation_for_industrial_live_streaming_advertising_summary"
paperUrl: "https://arxiv.org/abs/2608.24034"
---
> **Авторы:** Wencai Ye, Guangyi Liu, Chaoyi Wang, Wenbin Luo, Shengyu Wang, Mingjie Sun, Peng Wang, Quanming Yao, Wenjin Wu, Peng Jiang.
>
> **Аффилиации:** Could not be established from accessible arXiv HTML..
>
> **Источник:** arXiv:2608.24034 от 2026-08-25.

## Коротко: о чем статья

TAGR обновляет идентификаторы live ads, моделирует multi-scale intent и чередует fresh on-policy preference optimization с NTP maintenance для production live-stream advertising.

## Abstract

Live-streaming advertising is an important monetization channel on short-video and e-commerce platforms, where rapidly changing live content, promoted products, and user feedback impose strong freshness requirements on recommendation models. Existing generative recommenders designed for static domains fail at three levels: static semantic IDs (SID) cannot track evolving live ads; single-scale behavior modeling misses shifting intent; preference optimization conflicts between fresh on-policy feedback and training stability. We propose TAGR, a generative recommendation framework with temporal adaptation at three levels: live-ad tokenization, user intent modeling, and preference alignment. At the token level, Live Semantic-Collaborative ID (LSID) periodically refreshes each active ad's SID based on its current live scene and promoted products, while retaining a stable hierarchical token vocabulary for autoregressive generation. At the intent level, Intent-Aware Generation (IAG) models live-room entry histories at multiple temporal granularities as the primary intent sequence, keeps auxiliary behaviors as separate inputs, and weights next-token prediction (NTP) using post-request intent evidence and business value. At the alignment level, Intermittent On-Policy Preference Optimization (IOPO) periodically samples fresh candidate groups from the current policy and performs behavior- and value-aligned preference updates interleaved with supervised NTP maintenance to preserve learned behavior distribution. Deployed on a large-scale e-commerce live-stream advertising platform, TAGR improves live-room entry and shopping-cart click rates by 8.5% and 7.4%, respectively, and achieves a 16.1% revenue lift over the production baseline. These results demonstrate the effectiveness and industrial viability of temporally adaptive generative recommendation for live-stream advertising.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
