---
title: "Token-Level Credit Assignment Optimization for Generative Document Retrieval"
category: "generative_retrieval"
slug: "token_level_credit_assignment_optimization_for_generative_document_retrieval_summary"
catalogId: "paper-token_level_credit_assignment_optimization_for_generative_document_retrieval_summary"
paperUrl: "https://arxiv.org/abs/2608.12049"
---
> **Авторы:** Xinpeng Zhao, Yang Liu, Ran Chen, Xinyu Ma, Daiting Shi, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Xin Xin.
>
> **Аффилиации:** Shandong University; Peking University; Baidu Inc.; Leiden University.
>
> **Источник:** arXiv:2608.12049v1 от 2026-08-12. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

Метод заменяет одинаковый sequence-level reward для всех токенов DocID пошаговыми оценками вклада каждого решения в retrieval quality.

## Abstract

Generative retrieval models perform document retrieval by autoregressively generating document identifiers (DocIDs). This process naturally forms a sequential decision problem, where each decoding step selects a DocID token and the complete token sequence determines the retrieved document. However, retrieval effectiveness is typically evaluated only after the full DocID is generated, creating a mismatch between token-level generation and document-level relevance supervision. As a result, existing reinforcement learning methods for generative retrieval mostly rely on sequence-level rewards, where the same document-level feedback is propagated to all decoding steps. Such coarse-grained feedback makes it difficult to identify which token decisions are responsible for successful or failed retrieval. In this work, we propose a fine-grained reinforcement learning framework for generative retrieval with token-level relevance rewards. Instead of assigning a single reward to the entire generated DocID, we estimate step-wise rewards by measuring how each token decision changes the expected retrieval quality of the corresponding generation trajectory. This enables more precise credit assignment and encourages the policy to favor token decisions that contribute more directly to document-level relevance. We further develop practical reward estimation strategies tailored to the DocID generation process and incorporate them into a policy optimization framework. Experiments on retrieval benchmarks show that our method consistently outperforms sequence-level reward baselines, demonstrating the effectiveness of fine-grained supervision for aligning autoregressive DocID generation with retrieval objectives.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
