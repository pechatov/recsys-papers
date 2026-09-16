---
title: "REDSI: Addressing the Reproducibility and Evaluation Consistency of Differentiable Search Indexing for Document Retrieval"
category: "generative_retrieval"
slug: "redsi_addressing_the_reproducibility_and_evaluation_consistency_of_differentiable_search_indexing_for_document_retrieval_summary"
catalogId: "paper-redsi_addressing_the_reproducibility_and_evaluation_consistency_of_differentiable_search_indexing_for_document_retrieval_summary"
paperUrl: "https://arxiv.org/abs/2609.08860"
---
> **Авторы:** Vivien Nicolas, Hicham Randrianarivo, Pascale Sébillot, Caio Corro.
>
> **Аффилиации:** Artefact Research Center; INSA Rennes, IRISA, CNRS, Université de Rennes; MICS, CentraleSupélec, Université Paris-Saclay.
>
> **Источник:** arXiv:2609.08860 v1 от 2026-09-08.

## Коротко: о чем статья

ReDSI закрывает проблему воспроизводимости Differentiable Search Indexing: это открытая реализация DSI с atomic, naive и semantic document IDs и параметризуемым, документированным pipeline построения NQ320K. Авторы получают конкурентные результаты и систематически сравнивают качество retrieval, parameter efficiency, методы обучения и стратегии decoding при уменьшении масштаба модели.

## Abstract

The differentiable search index (DSI) framework (Tay et al., 2022) has become the de facto baseline for generative retrieval. However, DSI is hard to reproduce: no public implementation covers all three original document identifier types (atomic, naive, semantic), reported results vary widely, and the ubiquitous NQ320K dataset is built from Natural Questions through diverse and underspecified preprocessing. We introduce ReDSI, the first open-source DSI implementation supporting all three identifier types, together with a parameterizable and well-documented NQ320K construction pipeline. Experimentally, we achieve results that are competitive with or stronger than previous DSI baselines. Moreover, we conduct extensive experiments under model downscaling, covering retrieval effectiveness, parameter efficiency, training methods and decoding strategies, opening novel directions for future research.

## Ограничения

Это свежий препринт; заявленные результаты необходимо интерпретировать в рамках раскрытых авторами workload, данных, оборудования и evaluation protocol.
