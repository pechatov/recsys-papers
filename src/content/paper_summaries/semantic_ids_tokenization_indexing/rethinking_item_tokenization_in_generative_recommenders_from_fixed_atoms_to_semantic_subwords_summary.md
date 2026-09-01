---
title: "Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords"
category: "semantic_ids_tokenization_indexing"
slug: "rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to_semantic_subwords_summary"
catalogId: "paper-rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to_semantic_subwords_summary"
paperUrl: "https://arxiv.org/abs/2608.22734"
---
> **Авторы:** Xinrui Miao, Mingjia Yin, Jiaqing Zhang, Wei Guo, Yong Liu, Yuyang Ye, Hao Wang, Enhong Chen.
>
> **Аффилиации:** University of Science and Technology of China; Huawei Technologies..
>
> **Источник:** arXiv:2608.22734 от 2026-08-24.

## Коротко: о чем статья

SST заменяет fixed SID atoms в истории на variable-length semantic subwords, уменьшая intra-item attention overload и высвобождая capacity для behavioral transitions.

## Abstract

In generative recommender systems, items are typically tokenized into fixed-length semantic ID sequences for autoregressive next-item prediction. However, for user-context modeling, this fine-grained representation triggers Intra-item Attention Overload: excessive attention is spent on low-level intra-item dependencies rather than high-level inter-item behavioral transitions. To address this, we propose Semantic Subword Tokenization (SST), which represents historical items as variable-length semantic subwords while preserving fixed-length target decoding. SST first applies Item-level Subword Tokenization (IST) to merge stable adjacent atom tokens into compact semantic subword tokens, thereby reducing intra-item reassembly in the encoder. It then introduces Behavior-induced Co-occurrence Augmentation (BCA) to inject coarse-grained semantic prefix transition signals, guiding the freed modeling capacity toward inter-item behavioral regularities. Extensive experiments on three public datasets and three generative recommender backbones show empirical improvements of SST over fixed-length and transferable variable-length SID baselines. Code is available at https://github.com/mxrcandy/Semantic-Subword-Tokenization.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
