---
title: "PailitaoGR: Latent Think-with-Images for Generative Image Retrieval"
category: "generative_retrieval"
slug: "pailitaogr_latent_think_with_images_for_generative_image_retrieval_summary"
catalogId: "paper-pailitaogr_latent_think_with_images_for_generative_image_retrieval_summary"
paperUrl: "https://arxiv.org/abs/2608.26658"
---
> **Авторы:** Xiaomeng Fan, Yueran Liu, Shengyu Zhou, Chenghan Fu, Wanxian Guan, Feng Li, Chuan Yu, Jian Xu, Bo Zheng.
>
> **Аффилиации:** Could not be established from accessible arXiv HTML..
>
> **Источник:** arXiv:2608.26658 от 2026-08-27.

## Коротко: о чем статья

PailitaoGR обучает генеративный image retriever выделять target region и избирательно использовать auxiliary visual evidence без crop или OCR.

## Abstract

Generative retrieval has demonstrated strong performance by directly generating product semantic identifiers (SIDs). Extending this paradigm to image search, however, is nontrivial because real-world query images contain diverse information, including the search target, useful auxiliary evidence, and irrelevant visual content. This requires the model to identify and focus on the search target while selectively utilizing auxiliary evidence. In this paper, we propose \textbf{PailitaoGR}, a \emph{Latent Think-with-Images} method for generative image retrieval, which internalizes target-focused perception and selective auxiliary-evidence utilization into a the generative retrieval model, enabling \textit{Zooming without Cropping} and \textit{Reading without OCR}. Specifically, we design a target-focused perception mechanism that identifies and enhances visual tokens of the search target, consisting of a target Enhancer and a learning strategy based on on-policy distillation and attention guidance loss, enabling the model to focus on search-target regions. We also design a selective auxiliary-evidence utilization mechanism that identifies and enhances visual tokens of auxiliary evidence, including an auxiliary enhancer and an in-capacity incremental contrastive distillation strategy, enabling the model to exploit auxiliary evidence. We construct training and validation sets sampled from real-world online image-search logs. Experiments show that our method outperforms existing baselines by an average of 13.8\%, validating its effectiveness.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
