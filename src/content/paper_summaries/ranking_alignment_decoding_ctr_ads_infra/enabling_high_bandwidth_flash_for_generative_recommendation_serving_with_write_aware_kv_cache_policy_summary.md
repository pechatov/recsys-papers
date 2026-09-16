---
title: "Enabling High-Bandwidth Flash for Generative Recommendation Serving with Write-Aware KV Cache Policy"
category: "ranking_alignment_decoding_ctr_ads_infra"
slug: "enabling_high_bandwidth_flash_for_generative_recommendation_serving_with_write_aware_kv_cache_policy_summary"
catalogId: "paper-enabling_high_bandwidth_flash_for_generative_recommendation_serving_with_write_aware_kv_cache_policy_summary"
paperUrl: "https://arxiv.org/abs/2609.07175"
---
> **Авторы:** Danni Peng, Kai Wu, Tianyu Zuo, Pengfei Xia, Hui Zang.
>
> **Аффилиации:** Huawei Technologies Co., Ltd..
>
> **Источник:** arXiv:2609.07175 v1 от 2026-09-07.

## Коротко: о чем статья

Работа исследует инфраструктуру serving для генеративных рекомендаций: High-Bandwidth Flash хранит больше пользовательских KV-кэшей, а admission-controlled LRU-K не записывает в кэш пользователей с низким повторным использованием. В экспериментах HBF даёт в 3,8–4,7 раза большую пропускную способность, чем HBM-only, а LRU-K увеличивает расчётный срок службы flash примерно с одного года до более чем шести лет при K=10 без потери throughput.

## Abstract

Generative recommendation (GR) systems increasingly leverage user-level KV cache reuse to avoid recomputing long user histories. However, the growing KV cache capacity and bandwidth requirements introduce new challenges for memory system. High-Bandwidth Flash (HBF) provides a promising solution by offering substantially higher capacity than HBM while approaching HBM-class read bandwidth, enabling larger scale KV cache retention and improved serving throughput. Yet conventional Least-Recently-Used (LRU) KV cache management tightly couples KV cache writes with cache misses, generating excessive write traffic that rapidly exhausts flash endurance. In this work, we evaluate a write-aware KV cache policy based on admission-controlled LRU-K for HBF-based GR serving. By filtering low-reuse users before cache admission, LRU-K decouples KV cache writes from misses and significantly reduces unnecessary writes. We develop an analytical model to characterize GR serving performance, KV cache write traffic, and HBF lifetime, and evaluate performance across diverse memory systems and GR workloads. Our results show that HBF-based systems achieve 3.8 to 4.7 times higher throughput than HBM-only systems. Moreover, LRU-K extends HBF lifetime from about one year under conventional LRU to over six years with a moderate K=10, while maintaining comparable or even slightly improved throughput. These results highlight the importance of write aware KV cache policy for sustainable HBF-based GR serving.

## Ограничения

Это свежий препринт; заявленные результаты необходимо интерпретировать в рамках раскрытых авторами workload, данных, оборудования и evaluation protocol.
