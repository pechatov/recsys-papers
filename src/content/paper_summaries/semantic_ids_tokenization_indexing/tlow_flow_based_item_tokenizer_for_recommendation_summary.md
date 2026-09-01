---
title: "Tlow: Flow-based Item Tokenizer for Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "tlow_flow_based_item_tokenizer_for_recommendation_summary"
catalogId: "paper-tlow_flow_based_item_tokenizer_for_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.24176"
---
> **Авторы:** Nian Li, Chonggang Song, Jingtao Ding, Lingling Yi, Yong Li, Qingmin Liao.
>
> **Аффилиации:** Tsinghua University; Tencent Inc.; Shenzhen International Graduate School, Tsinghua University..
>
> **Источник:** arXiv:2608.24176 от 2026-08-25.

## Коротко: о чем статья

Tlow normalizes item embeddings flow-моделью перед независимой tokenization и связывает codebook с token embedding space для более чётких token IDs.

## Abstract

Item tokenizer encodes semantic embeddings into token IDs to replace the randomly assigned item IDs used in traditional recommendation models, fundamentally addressing the problems of excessive parameters and cold starts. However, the most common tokenizer, RQ-VAE, suffers from low decoding efficiency due to the inherent dependencies among its codebooks. Meanwhile, efficient independent tokenizers such as optimized product quantization (OPQ) still struggle with dimensional correlations and distribution complexity of semantic embeddings. In this work, we propose a f\underline{low}-based item \underline{T}okenizer (Tlow) to transform raw semantic embeddings into a latent space where embeddings conform to a unified standard normal distribution, achieving dual advantages of dimensional independence and distributional simplicity. Independent tokenization performed on these latent embeddings yields semantically clear token IDs. Additionally, we introduce a novel codebook guidance to align the codebook space with the token embedding space, further aiding the learning of more semantically distinct token embeddings. Offline experiments on four public datasets demonstrate that Tlow's tokenization and codebook guidance significantly improve recommendation performance. The improvement on cross-domain and multi-modal recommendations also proves the effectiveness of item tokenization in a simplified embedding space. Online experiments for a multi-modal retrieval task on China's largest social media platform WeChat validate Tlow's powerful distribution transformation capability. The retrieval model based on token IDs improves user CTR by 10.32\% globally and by 11.64\% for new items. Our codes are available at https://github.com/wjjln/Tlow.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
