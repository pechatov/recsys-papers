---
title: "PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "prq_kmeans_projection_residual_quantization_for_semantic_id_tokenization_summary"
catalogId: "paper-prq_kmeans_projection_residual_quantization_for_semantic_id_tokenization_summary"
paperUrl: "https://arxiv.org/abs/2608.24207"
---
> **Авторы:** Yunxiao Luo, Siyuan Wang, Ben Chen, Chenyi Lei.
>
> **Аффилиации:** Could not be established from accessible arXiv HTML..
>
> **Источник:** arXiv:2608.24207 от 2026-08-25.

## Коротко: о чем статья

PRQ-KMeans улучшает residual quantization SID: вычитает global mean, уточняет centroids weighted top-k similarities и использует projection residual.

## Abstract

Semantic identifiers (SIDs) represent entities as hierarchical token sequences for generative retrieval and recommendation. Residual-quantization tokenizers construct these sequences by selecting a codeword at each level and passing a residual to the next. We view this process as progressive commonality removal: each token captures a component shared within its group, while later tokens should model the remaining differences. This view reveals three limitations: a corpus-wide shared component can consume first-level capacity, hard assignment ignores graded similarities to nearby codewords, and full-codeword subtraction can leave variation along the selected-codeword direction in the next residual. We therefore develop our solution in the post-hoc setting, where residual construction is not constrained by input reconstruction. Specifically, we propose PRQ-KMeans, which removes the global-mean component, refines centroids with Top-k similarity-weighted updates, and replaces full-codeword subtraction with a projection residual that removes each representation's selected-centroid component. Experiments on a large-scale industrial search dataset and four public recommendation benchmarks show that PRQ-KMeans achieves the strongest overall performance among the evaluated tokenizers, including gains of up to 7.4% in HitRate and 11.8% in MRR on the industrial dataset.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
