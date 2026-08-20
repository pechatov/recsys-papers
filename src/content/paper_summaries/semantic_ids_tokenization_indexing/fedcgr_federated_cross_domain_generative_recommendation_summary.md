---
title: "FedCGR: Federated Cross-Domain Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "fedcgr_federated_cross_domain_generative_recommendation_summary"
catalogId: "paper-fedcgr_federated_cross_domain_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.10929"
---
> **Авторы:** Zhuodong Liu, Hugen Lv, Xiangyu Li, Bohan Guo, Peiyu Hu.
>
> **Аффилиации:** Beijing Jiaotong University; Shanghai Jiao Tong University; University of Malaya; Xi’an Jiaotong-Liverpool University.
>
> **Источник:** arXiv:2608.10929v1 от 2026-08-11. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

FedCGR использует публичные item metadata для общего semantic-ID языка между доменами и reliability-aware aggregation без передачи приватных interactions.

## Abstract

Cross-domain recommendation (CDR) transfers preference knowledge across related domains, but federated deployment makes cross-domain alignment difficult because the behavioral anchors that align item spaces, such as overlapping users and shared interaction signals, are often sparse, unavailable, or privacy-sensitive across clients. To address this tension, we revisit federated CDR as generation over a stable semantic item language. By representing items as discrete semantic ID (SID) sequences derived from public item-side metadata, cross-domain item alignment is induced by a shared vocabulary rather than by exchanging private interactions or aligning domain-specific embeddings. Directly federating SID-based generators, however, introduces two design constraints: the SID tokenizer must remain fixed to preserve cross-client token consistency, which creates a semantic-only bottleneck because local collaborative filtering (CF) signals cannot be globally shared or aligned; meanwhile, standard federated averaging can cause negative transfer under domain heterogeneity. To overcome these constraints, we propose FedCGR, a federated generative CDR framework that keeps the item language stable and makes adaptation explicit. FedCGR injects local CF evidence through a reliability-aware semantic interface and trains a prototype-personalized generator that selectively aggregates shared parameters according to domain relatedness while keeping domain-specific quantities local. Experiments on six Amazon cross-domain scenarios show that FedCGR consistently outperforms federated generative baselines and achieves competitive performance against strong sequential and federated CDR methods under both full-ranking and sampled evaluation protocols.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
