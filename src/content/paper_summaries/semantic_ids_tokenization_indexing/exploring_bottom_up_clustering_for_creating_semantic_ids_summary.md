---
title: "Exploring Bottom-Up Clustering for Creating Semantic IDs"
category: "semantic_ids_tokenization_indexing"
slug: "exploring_bottom_up_clustering_for_creating_semantic_ids_summary"
catalogId: "paper-exploring_bottom_up_clustering_for_creating_semantic_ids_summary"
paperUrl: "https://arxiv.org/abs/2609.08310"
---
> **Авторы:** Leah Woldemariam, Sudhanshu Garg, Taha Belkhouja, Charles Kim-Yip, Ali Sahami.
>
> **Аффилиации:** Cornell Tech, Cornell University; PayPal.
>
> **Источник:** arXiv:2609.08310 v1 от 2026-09-08.

## Коротко: о чем статья

Авторы предлагают строить Semantic IDs с помощью bottom-up clustering: сначала сохраняется локальная структура пространства item-эмбеддингов, затем кластеры объединяются в иерархию. Метод нацелен одновременно на уникальность идентификаторов и сохранение полезной семантики, улучшая качество кластеризации SID и их пригодность для downstream generative retrieval.

## Abstract

The success of generative retrieval has largely been attributed to the use of Semantic IDs, which improve over arbitrary item-level identifiers such as hashes by capturing the semantics of items. The main challenges faced when constructing Semantic IDs, however, is in mapping each identifier to a unique product and capturing information valuable to downstream tasks. Past works have appended additional codewords to de-duplicate item identifiers and utilized residual quantization to create hierarchical clusters. In this work, we present an algorithm for generating Semantic IDs that ensure the identifiers are both unique and preserve the structure of the original embedding. Key to our work is the use of bottom-up clustering to preserve local structure in the embedding space, improving the clustering quality of the resulting Semantic IDs and their utility for downstream generative retrieval.

## Ограничения

Это свежий препринт; заявленные результаты необходимо интерпретировать в рамках раскрытых авторами workload, данных, оборудования и evaluation protocol.
